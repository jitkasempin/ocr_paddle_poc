from openai import AsyncOpenAI
from typing import List, Dict
from loguru import logger
from pydantic import BaseModel
# from app.config import default_config


class Qwen3VLLMClient:
    def __init__(self, base_url="https://9trsuv1opxh8oh-8000.proxy.runpod.net/v1"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="0"
            # api_key="rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c"
        )
        self.model_name = "Qwen/Qwen3.5-9B"
        # self.model_name = "Qwen/Qwen3-14B"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0

    async def chat_completion(self, messages: List[Dict[str, str]], js_schema: BaseModel) -> str:
        """Async chat completion with Qwen3.5-27B-FP8 via SGLang OpenAI-compatible API"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.6,
            # max_tokens=10000,
            extra_body={
                "top_k": 20,
                "top_p": 0.95,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "min_p": 0.0,
                "structured_outputs": {"json": js_schema.model_json_schema()}
            }
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

            logger.info("The return data below: ")

            logger.info(response.choices[0].message.content)

            logger.info(
                f"[Qwen3.5-27B-FP8] Token usage — "
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
