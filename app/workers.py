import json
import psycopg2
from config import redis_client
import os
import logging
import requests
# LangChain PDF loader and vector store
from langchain_community.document_loaders import PyPDFLoader
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
from qdrant_client.http import models
import uuid
import math
# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("worker")

# ----------------- Config -----------------
QUEUE_NAME = "book_indexing_queue"
OUTPUT_JOB_QUEUE = "output_jobs_queue"
FAILED_QUEUE = "failed_jobs_queue"
SUCCESS_QUEUE = "successful_jobs_queue"
DB_URL = os.getenv("DB_URL", "postgres://user:pass@localhost/dbname")
QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION = "books_collection"
HF_EMBEDDING_URL = os.getenv("HF_EMBEDDING_URL", "http://huggingface-embeddings:80/predict")
API_URL = 'https://sarvesh92-sentence-transformers-all-minilm-l6-v2.hf.space/embed'


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


client = QdrantClient(QDRANT_URL)

def embed_texts_in_batches(texts, batch_size=32):
    """
    Batches text and sends to the HuggingFace embedding API.
    Returns a list of embeddings for all texts.
    """
    all_embeddings = []
    total_batches = math.ceil(len(texts) / batch_size)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = requests.post(API_URL, json={"texts": batch})
        if response.status_code == 200:
            embeddings = response.json()["embedding"]
            # API returns [{"embedding": [...], "index": idx}, ...]
            all_embeddings.extend([item["embedding"] for item in embeddings])
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")

        logger.info(f"Processed batch {i//batch_size + 1} of {total_batches}")

    return all_embeddings

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

        
        texts = [d.page_content for d in split_docs]
        embeddings = embed_texts_in_batches(texts, batch_size=32)
        points = [
        models.PointStruct(
        id=str(uuid.uuid4()),
        vector=emb,
        payload={
            "text": text,
            "book_id": book_id,
            "job_id": job_id,
            "user_id": user_id
        }
        )
    for text, emb in zip(texts, embeddings)
]
        # Push to Qdrant
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

        logger.info(f"Indexed book_id={book_id} with {len(split_docs)} chunks.")
        update_job_status(job_id, "COMPLETED")
        return True

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        update_job_status(job_id, "FAILED")
        redis_client.rpush(OUTPUT_JOB_QUEUE, json.dumps(job_data))

# ----------------- Worker Loop -----------------
def worker_loop():
    logger.info("Worker started. Waiting for jobs...")
    try:
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
            isIndexed = process_job(job_data)
            if isIndexed:
                logger.info(f"Job {job_data['id']} completed successfully.")
                redis_client.rpush(OUTPUT_JOB_QUEUE, json.dumps(job_data))
            else:
                logger.error(f"Job {job_data['id']} failed during processing.")
        except Exception as e:
            logger.error(f"Worker loop error: {e}")

if __name__ == "__main__":
    worker_loop()
