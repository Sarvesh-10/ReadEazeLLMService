
from fastapi import APIRouter,HTTPException,Request
from ..services.generator_factory import ImageGeneratorFactory
from pydantic import BaseModel
from ..customLogging import logger

class ImageRequest(BaseModel):
    prompt: str
    model: str = "gemini"
router = APIRouter(prefix='/image',tags=['image'])
@router.post("/generate-image")
async def generate_image(request: Request):
    
    logger.info("Received request to generate image", extra={"request": request.headers, "body": await request.body()})
    try:
        data = await request.json()
        prompt = data.get('prompt')
        print("prompt is:", prompt)
        generator = ImageGeneratorFactory.get_generator("gemini")
        return generator.generate_image(prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

