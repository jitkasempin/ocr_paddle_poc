from openai import AsyncOpenAI
from typing import List, Dict
# from app.config import default_config


class Qwen3VLLMClient:
    def __init__(self, base_url="https://7543zwihjsv4co-8000.proxy.runpod.net/v1"):
        # print("Runpod API Key")
        # print(default_config.runpod_api_key)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )
        self.model_name = "./Qwen3-14B-FP8-Dynamic"

    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Async chat completion with Qwen3-14B"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=8000
            # extra_body={
            #     "top_k": kwargs.get('top_k', 20),
            #     "repetition_penalty": kwargs.get('repetition_penalty', 1.05),
            #     "chat_template_kwargs": {
            #         "enable_thinking": kwargs.get('enable_thinking', False)
            #     }
            # }
        )
        return response.choices[0].message.content
