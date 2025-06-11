
from fastapi import APIRouter,HTTPException,Request
from ..services.generator_factory import ImageGeneratorFactory
from pydantic import BaseModel

class ImageRequest(BaseModel):
    prompt: str
    model: str = "gemini"
router = APIRouter(prefix='/image',tags=['image'])
@router.post("/generate-image")
async def generate_image(request: Request):
    try:
        data = await request.json()
        prompt = data.get('prompt')
        print("prompt is:", prompt)
        generator = ImageGeneratorFactory.get_generator("gemini")
        return generator.generate_image(prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

