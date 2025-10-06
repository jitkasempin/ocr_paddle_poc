from __future__ import annotations

import json
import os
from collections.abc import Generator

import requests
from loguru import logger

from core.utils import convert_files_to_images
from core.utils import encode_image
from core.utils import resize_images
from core.utils import validate_file_paths


def stream_request(
    messages: list[dict],
    model_name: str,
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> Generator[str]:
    """
    Make a streaming request to the vLLM server running on localhost:8000
    """
    vlm_url ="https://api.runpod.ai/v2/gcbh5q33h2r7g2/openai/v1"  # os.getenv("VLM_MODEL_URL", "")
    if vlm_url == "":
        raise ValueError(
            "VLM_MODEL_URL is not set. Please set it to the URL of the VLM model."
        )

    # Prepare the request payload
    payload = {
        "model": model_name.replace("hosted_vllm/", ""),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,  # Enable streaming
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c",
    }

    # Make streaming request
    url = f"{vlm_url}/chat/completions"

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data)
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                choice = json_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            continue
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making streaming request: {e}")
        raise


def convert_to_markdown_stream(
    file_inputs, model_name, max_img_size, concurrency_limit, max_gen_tokens
):
    """
    Generator function that yields streaming markdown conversion results
    Processes images one by one and concatenates results
    """
    file_paths: list[str] = [
        file_input[0] if isinstance(file_input, tuple) else file_input
        for file_input in file_inputs
    ]
    validate_file_paths(file_paths)
    file_paths = convert_files_to_images(file_paths)
    resize_images(file_paths, max_img_size)

    # Create system prompt for PDF to markdown conversion
    user_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

    logger.info(
        f"Converting {len(file_paths)} image(s) to markdown using {model_name} (processing one by one)"
    )

    # Accumulate results from all pages
    full_markdown_content = ""

    # Process each image individually
    for i, file_path in enumerate(file_paths):
        logger.info(f"Processing page {i + 1} of {len(file_paths)}: {file_path}")

        # Build messages for this single image
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(file_path)}"
                },
            },
            {"type": "text", "text": user_prompt},
        ]

        messages = [{"role": "user", "content": content}]

        # Stream this individual page
        page_content = ""
        try:
            raise Exception("Force to use sync request")
        
            for chunk in stream_request(
                messages=messages,
                model_name=model_name,
                max_tokens=max_gen_tokens,
            ):
                page_content += chunk
                # Yield accumulated content from all pages processed so far + current page
                current_total = (
                    full_markdown_content
                    + f"Page {i + 1} of {len(file_paths)}\n"
                    + page_content
                )
                yield current_total

            # Process the completed page content and add it to the full content
            full_markdown_content += (
                f"Page {i + 1} of {len(file_paths)}\n" + page_content
            )
            logger.info(f"Successfully converted page {i + 1}")

        except Exception as e:
            logger.error(f"Error during streaming conversion of page {i + 1}: {e}")
            # Fallback to non-streaming for this page
            logger.info(f"Falling back to non-streaming request for page {i + 1}")
            try:
                from core.client import sync_request

                tmp_model_name = f"hosted_vllm/{model_name}" 

                response = sync_request(
                    messages=messages, model_name=tmp_model_name, max_tokens=max_gen_tokens
                )
                page_content = response["choices"][0]["message"]["content"]
                full_markdown_content += (
                    f"Page {i + 1} of {len(file_paths)}\n" + page_content
                )
                yield full_markdown_content
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for page {i + 1}: {fallback_error}")
                error_content = (
                    f"\n\n**Error processing page {i + 1}: {str(fallback_error)}**\n\n"
                )
                full_markdown_content += (
                    f"Page {i + 1} of {len(file_paths)}\n" + error_content
                )
                yield full_markdown_content

    # print raw model response
    logger.info(f"Raw model response:\n {full_markdown_content}")
    logger.info("Successfully completed document conversion")


def convert_to_markdown(
    file_inputs, model_name, max_img_size, concurrency_limit, max_gen_tokens
):
    """
    Non-streaming version for backward compatibility
    """
    # Get the final result from the streaming generator
    final_result = ""
    for result in convert_to_markdown_stream(
        file_inputs, model_name, max_img_size, concurrency_limit, max_gen_tokens
    ):
        final_result = result
    return final_result
