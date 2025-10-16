from openai import OpenAI
from together import Together
from abc import ABC, abstractmethod


class ModelClient(ABC):
    @abstractmethod
    def generate_completion(self, messages: list) -> str:
        pass


class TogetherClient(ModelClient):
    def __init__(self, model_name, temperature, max_tokens):
        self.client = Together()
        self.model_name = model_name
        self.temperature = temperature  
        self.max_tokens = max_tokens

    def generate_completion(self, messages: list) -> str:
        if 'qwen3' in self.model_name.lower():
            response = self.client.chat.completions.create( 
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                chat_template_kwargs={"enable_thinking": True},
                max_tokens=self.max_tokens
            )  
        else: 
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
                stream=False,
                max_tokens=self.max_tokens,
            )
        return response.choices[0].message.content


class OpenAIClient(ModelClient):
    def __init__(self, model_name, temperature, max_tokens):
        self.client = OpenAI()
        self.model_name = model_name.split("/")[-1]
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_completion(self, messages: list) -> str:
        if self.model_name == "o4-mini" or self.model_name == "o3-mini":
            response = self.client.responses.create(
                model=self.model_name,
                reasoning={"effort": self.temperature},
                input=messages,
                max_output_tokens=self.max_tokens
            )
        else: 
            response = self.client.responses.create(
                model=self.model_name,
                temperature=self.temperature,
                input=messages, 
                max_output_tokens=self.max_tokens
            )
        return response.output_text
    

def get_model_client(model_name, temperature, max_tokens):
    if "openai" in model_name.lower():
        return OpenAIClient(model_name, temperature, max_tokens)
    else:
        return TogetherClient(model_name, temperature, max_tokens)