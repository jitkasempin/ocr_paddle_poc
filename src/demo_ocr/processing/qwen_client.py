from openai import AsyncOpenAI
from typing import List, Dict
# from app.config import default_config


class Qwen3VLLMClient:
    def __init__(self, base_url="https://api.runpod.ai/v2/em2h41xp8ytr67/openai/v1"):
        # print("Runpod API Key")
        # print(default_config.runpod_api_key)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c"
        )
        self.model_name = "Qwen/Qwen3-14B"

    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Async chat completion with Qwen3-14B"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=10000
            # extra_body={
            #     "top_k": kwargs.get('top_k', 20),
            #     "repetition_penalty": kwargs.get('repetition_penalty', 1.05),
            #     "chat_template_kwargs": {
            #         "enable_thinking": kwargs.get('enable_thinking', False)
            #     }
            # }
        )
        return response.choices[0].message.content
