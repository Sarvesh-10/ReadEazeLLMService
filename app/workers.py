import json
import time
import psycopg2
from config import redis_client
import os
import logging
import requests
# LangChain PDF loader and vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
# LangChain text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# Qdrant client
from qdrant_client import QdrantClient

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("worker")

# ----------------- Config -----------------
QUEUE_NAME = "book_indexing_queue"
FAILED_QUEUE = "failed_jobs_queue"
SUCCESS_QUEUE = "successful_jobs_queue"
DB_URL = os.getenv("DB_URL", "postgres://user:pass@localhost/dbname")
QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION = "books_collection"
HF_EMBEDDING_URL = os.getenv("HF_EMBEDDING_URL", "http://huggingface-embeddings:80")

# ----------------- DB Helpers -----------------
def get_db_connection():
    return psycopg2.connect(DB_URL)

def fetch_pdf_from_db(book_id: int, output_path: str) -> str:
    """Fetch PDF binary from DB and save to local file."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT file_data FROM books WHERE id = %s", (book_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No book found with id={book_id}")
            
            pdf_bytes = row[0]
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            
            return output_path
    finally:
        conn.close()

def update_job_status(job_id: int, status: str):
    """Update job status in DB."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE book_indexing_jobs SET status = %s, updated_at = NOW() WHERE id = %s",
                (status, job_id),
            )
            conn.commit()
    finally:
        conn.close()

# ----------------- HF Embedding Client -----------------
class HFEmbeddingClient:
    def __init__(self, service_url=HF_EMBEDDING_URL):
        self.service_url = service_url.rstrip("/")
        self.session = requests.Session()

    def get_embeddings(self, texts: list):
        """
        texts: List[str]
        returns: List[List[float]] embeddings
        """
        payload = {"text": texts}
        response = self.session.post(f"{self.service_url}/v1/embeddings", json=payload)
        response.raise_for_status()
        return response.json()["embedding"]  # adjust key based on HF image

embedding_client = HFEmbeddingClient()

# ----------------- Job Processing -----------------
def process_job(job_data: dict):
    job_id = job_data["id"]
    book_id = job_data["book_id"]
    user_id = job_data.get("user_id")

    try:
        # Fetch PDF
        temp_pdf_path = f"/tmp/book_{book_id}.pdf"
        pdf_file = fetch_pdf_from_db(book_id, temp_pdf_path)
        logger.info(f"Fetched PDF for book_id={book_id} to {pdf_file}")
        update_job_status(job_id, "IN_PROGRESS")

        # Load PDF
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        if not documents:
            raise ValueError("No text extracted from PDF.")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = []
        for doc in documents:
            split_docs.extend(splitter.create_documents([doc.page_content], metadatas=[doc.metadata]))

        # Add metadata
        for d in split_docs:
            d.metadata["book_id"] = book_id
            d.metadata["job_id"] = job_id
            d.metadata["user_id"] = user_id

        # ----------------- Embed using HF embedding pod -----------------
        texts = [d.page_content for d in split_docs]
        embeddings = embedding_client.get_embeddings(texts)

        # Push to Qdrant
        vectorstore = Qdrant.from_documents(
            split_docs,
            embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )

        logger.info(f"Indexed book_id={book_id} with {len(split_docs)} chunks.")
        update_job_status(job_id, "COMPLETED")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        update_job_status(job_id, "FAILED")
        redis_client.rpush(FAILED_QUEUE, json.dumps(job_data))

# ----------------- Worker Loop -----------------
def worker_loop():
    logger.info("Worker started. Waiting for jobs...")
    try:
        client = QdrantClient(QDRANT_URL)
        collections = client.get_collections()
        logger.info(f"Qdrant reachable. Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.error(f"Qdrant connectivity test failed: {e}")
        return

    while True:
        try:
            logger.info("Waiting for jobs...")
            job = redis_client.blpop(QUEUE_NAME, timeout=0)
            _, job_json = job
            job_data = json.loads(job_json)
            process_job(job_data)
        except Exception as e:
            logger.error(f"Worker loop error: {e}")

if __name__ == "__main__":
    worker_loop()
