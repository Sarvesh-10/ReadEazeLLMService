from .gemini_generator import GeminiGenerator

class ImageGeneratorFactory:
    _instances = {}
    
    @staticmethod
    def get_generator(model_name: str):
        print("Getting generator for model", model_name)
        print("Instances:", ImageGeneratorFactory._instances)
        
        # Check if the model is supported
        if model_name == "gemini":
            if model_name not in ImageGeneratorFactory._instances:
                ImageGeneratorFactory._instances[model_name] = GeminiGenerator()
            return ImageGeneratorFactory._instances[model_name]
        
        # Raise an error for unsupported models
        raise ValueError(f"Model {model_name} not supported")