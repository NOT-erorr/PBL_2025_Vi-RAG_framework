import os
from typing import List
from google import genai
from .prompt import PromptBuilder


class GeminiLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-1.5-flash",
        prompt_builder: PromptBuilder | None = None
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(
        self,
        query: str,
        contexts: List[str]
    ) -> str:
        if not contexts:
            return "Không có đủ thông tin để trả lời câu hỏi."

        prompt = self.prompt_builder.build(query, contexts)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text.strip()
