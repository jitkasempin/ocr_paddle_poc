from openai import AsyncOpenAI
from typing import List, Dict


class SchematronClient:
    def __init__(self, base_url="https://api.inference.net/v1"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="inference-66489c8266a84025a2c150297a34c5ce",
        )
        self.model_name = "inference-net/schematron-8b"

    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Async chat completion with Schematron-8B"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0,
            stream=False,
        )
        return response.choices[0].message.content
