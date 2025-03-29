from abc import ABC, abstractmethod

class ImageGenerator(ABC):
    @abstractmethod
    def generate_image(self,prompt:str):
        pass