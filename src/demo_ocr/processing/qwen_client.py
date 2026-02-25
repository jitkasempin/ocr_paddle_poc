from openai import AsyncOpenAI
from typing import List, Dict
from loguru import logger
# from app.config import default_config


class Qwen3VLLMClient:
    def __init__(self, base_url="https://api.runpod.ai/v2/em2h41xp8ytr67/openai/v1"):
        # print("Runpod API Key")
        # print(default_config.runpod_api_key)
        self.client = AsyncOpenAI(
            base_url=base_url,
            # api_key="0"
            api_key="rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c"
        )
        self.model_name = "Qwen/Qwen3-14B" # "./Qwen3-14B-FP8-Dynamic"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0

    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Async chat completion with Qwen3-14B"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=10000,
            # extra_body={
            #     "top_k": kwargs.get('top_k', 20),
                # "repetition_penalty": 1.05,
                # "chat_template_kwargs": {
                    # "enable_thinking": False
                # }
            # }
        )

        # Track token usage
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total = response.usage.total_tokens

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total
            self.request_count += 1

            logger.info(
                f"[Qwen3-14B] Token usage — "
                f"prompt: {prompt_tokens}, "
                f"completion: {completion_tokens}, "
                f"total: {total} | "
                f"Cumulative — "
                f"prompt: {self.total_prompt_tokens}, "
                f"completion: {self.total_completion_tokens}, "
                f"total: {self.total_tokens}, "
                f"requests: {self.request_count}"
            )

        return response.choices[0].message.content

    def get_token_usage(self) -> Dict[str, int]:
        """Return cumulative token usage stats."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }

    def reset_token_usage(self):
        """Reset cumulative token counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
