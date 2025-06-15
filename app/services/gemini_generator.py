from fastapi import Response
from google import genai
from google.genai import types
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse
from .image_generator import ImageGenerator
import os

class GeminiGenerator(ImageGenerator):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiGenerator, cls).__new__(cls)
            api_key = os.getenv("GEMINI_API_KEY")
            cls._instance.client = genai.Client(api_key=api_key)
            if not cls._instance.client:
                raise ValueError("Gemini client initialization failed. Please check your API key.")
            print("Gemini generator initialized with API key:", api_key)
        return cls._instance

    def generate_image(self, prompt):
        print("gemini generator generating image")
        response = self.client.models.generate_content(
            model=os.getenv("GEMINI_MODEL_NAME"),
            contents=(prompt),
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image']),
        )
        
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                img_io = BytesIO()
                image.save(img_io, format="PNG")
                print(image.size)
                return Response(img_io.getvalue(), media_type="image/png")

        return {"error": "No image generated"}
