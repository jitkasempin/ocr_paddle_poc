# from .fast_vision import FastVLLM
import torch
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_ollama import ChatOllama
# import outlines
import re
from olmocr.pipeline import build_page_query
import httpx
# from outlines import Generator, Template
from fastmrz import FastMRZ
import mimetypes
# import mimetypes
# import json
from typing import Optional, List, Dict, Any
from google import genai
import os
# #region agent log
def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        import json, time
        payload = {
            "sessionId": "debug-session",
            "runId": os.getenv("AGENT_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        os.makedirs("/home/jitkasem/.cursor", exist_ok=True)
        with open("/home/jitkasem/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion

# from typhoon_ocr import prepare_ocr_messages
from openai import OpenAI, AsyncOpenAI
import json
import logging
from PIL import Image
from io import BytesIO
import base64
from dots_ocr.utils import dict_promptmode_to_prompt

# Configure logging for token usage - outputs to stdout for Docker logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("typhoon_ocr")
# from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.image_utils import PILimage_to_base64
# from typhoon_ocr import ocr_document
from typhoon_ocr import ocr_document, prepare_ocr_messages
import asyncio
# Requires `pip install docling-surya`
# See https://pypi.org/project/docling-surya/
_agent_log(
    "H1",
    "processing/ocr_no_flash.py:docling_surya_import",
    "Attempting import docling_surya",
    {"pythonpath": os.getenv("PYTHONPATH")},
)
try:
    from docling_surya import SuryaOcrOptions
    _agent_log(
        "H1",
        "processing/ocr_no_flash.py:docling_surya_import",
        "Imported docling_surya successfully",
        {"has_docling_surya": True},
    )
except Exception as e:
    import sys
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            docling_surya_version = version("docling-surya")
        except PackageNotFoundError:
            docling_surya_version = None
        try:
            surya_ocr_version = version("surya-ocr")
        except PackageNotFoundError:
            surya_ocr_version = None
        try:
            docling_version = version("docling")
        except PackageNotFoundError:
            docling_version = None
    except Exception:
        docling_surya_version = None
        surya_ocr_version = None
        docling_version = None

    _agent_log(
        "H1",
        "processing/ocr_no_flash.py:docling_surya_import",
        "Failed to import docling_surya",
        {
            "error_repr": repr(e),
            "python_version": sys.version,
            "executable": sys.executable,
            "docling_surya_version": docling_surya_version,
            "surya_ocr_version": surya_ocr_version,
            "docling_version": docling_version,
            "sys_path_head": sys.path[:8],
        },
    )
    raise

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Disable flash_attn to avoid import errors
import os
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

# Import transformers after setting the environment variable
from transformers import DonutProcessor, VisionEncoderDecoderModel
import fitz  # PyMuPDF
from .qwen_client import Qwen3VLLMClient

# class InvoicePaymentDate(BaseModel):
    # payment_term_date: datetime = Field(
        # default=None, description="date of the payment term if available in iso format"
    # )

class InvoiceIssueDate(BaseModel):
    invoice_day: int = Field(
        default=None, description="The day part of the invoice issued date. Must have the value between 1 and 31"
    )
    invoice_month: int = Field(
        default=None, description="The month part of the invoice issued date. Must have the value between 1 and 12. Convert the month name in English or month name in three english letters abbreviation (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec) or month name in Thai (มกราคม, กุมภาพันธ์, มีนาคม, เมษายน, พฤษภาคม, มิถุนายน, กรกฎาคม, สิงหาคม, กันยายน, ตุลาคม, พฤศจิกายน, ธันวาคม) to the number 1 - 12"
    )
    invoice_year: int = Field(
        default=None, description="The year part of the invoice issued date. The year can be Common Era (CE) or Buddhist Era (BE) and can be in 2 or 4 digits. The year must be in the range of 2010 - 2025 (or 10 - 25) for CE year and 2543 - 2568 (or 43 - 68) for BE year"
    )

class InvoicePaymentDate(BaseModel):
    payment_day: int = Field(
        default=None, description="The day part of the payment term date. Must have the value between 1 and 31"
    )
    payment_month: int = Field(
        default=None, description="The month part of the payment term date. Must have the value between 1 and 12. Convert the month name in English or month name in three english letters abbreviation (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec) or month name in Thai (มกราคม, กุมภาพันธ์, มีนาคม, เมษายน, พฤษภาคม, มิถุนายน, กรกฎาคม, สิงหาคม, กันยายน, ตุลาคม, พฤศจิกายน, ธันวาคม) to the number 1 - 12"
    )
    payment_year: int = Field(
        default=None, description="The year part of the payment term date. The year can be Common Era (CE) or Buddhist Era (BE) and can be in 2 or 4 digits. The year must be in the range of 2010 - 2025 (or 10 - 25) for CE year and 2543 - 2568 (or 43 - 68) for BE year"
    )   


class OCR:
    def __init__(self,ocr_model="FILM6912/typhoon-ocr-7b",llm_model="qwen3:14b",**kwargs):
        # self.model=FastVLLM()
        # self.model.load_model(ocr_model,**kwargs)
        # self.model.model= torch.compile(self.model.model)

        # Load the pre-trained Donut model and processor
        self.donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        self.donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

        
        # Configure httpx timeout
        timeout_config = httpx.Timeout(
            timeout=330.0,
            connect=15.0,
            read=300.0,
            write=15.0
        )


         # Create httpx client with retry logic
        http_client = httpx.AsyncClient(
            timeout=timeout_config
            # limits=httpx.Limits(
                # max_keepalive_connections=1,
                # max_connections=1
            # ),
            # transport=httpx.AsyncHTTPTransport(retries=2)
        )
        
        

        self.llm=ChatOllama(
            model=llm_model,
            temperature=0,
            base_url="https://ml.weaverbase.com/ollama"
        )

        self.my_openai = AsyncOpenAI(base_url="https://veejutidvzi7xy-8000.proxy.runpod.net/v1", api_key="rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c", http_client=http_client)
        # self.my_openai = OpenAI(base_url="https://8000-01jv6gbqesg14ne3mavgm9acm7.cloudspaces.litng.ai/v1", api_key="api-key")
        self.olm_ocr_openai = AsyncOpenAI(base_url="https://api.runpod.ai/v2/ajplyymntb6f54/openai/v1", api_key="rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c")

        self.nanonet_client = AsyncOpenAI(base_url="https://ifp0ig0mslclt9-8000.proxy.runpod.net/v1", api_key="0")

        self.q_client = Qwen3VLLMClient()

        self.fast_mrz = FastMRZ()

        self.hunyuan_ocr = OpenAI(base_url="http://86.38.238.11:8000/v1", api_key="EMPTY", timeout=3600)

        # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # self.ext_model = outlines.from_gemini(genai.Client(api_key="AIzaSyAaJcvCSi4s9FvVi5JGSzkEQ8uP_45tttw"), "gemini-2.0-flash")

        # self.prompt_template = Template.from_string(
        #     """
        #     Today's date is {{ now }}
        #     Given a user message, extract information in the message like payment term date in iso format.
        #     If the payment term date is relative, think step by step to find the right date.
        #     Here is the message:
        #     {{ message }}
        #     """
        # )

        # self.invoice_prompt_template = Template.from_string(
        #     """
        #     Today's date is {{ now }}
        #     Given a user message, extract information in the message like invoice issued date in iso format.
        #     If the invoice issued date is relative, think step by step to find the right date.
        #     Here is the message:
        #     {{ message }}
        #     """
        # )
    async def olmocr_runpod_predict(self, pdf_file_path, page_number):
        query = await build_page_query(pdf_file_path, page=page_number, target_longest_image_dim=2048)
        query['model'] = 'Adun/olmOCR-7B-thai-v3.2'
        response = await self.olm_ocr_openai.chat.completions.create(**query) 

        return response.choices[0].message.content


    async def ocr_page_with_nanonets_s(self, img_file_path):
        
        img_base64 = None
        with open(img_file_path, "rb") as image_file:
            img_base64 =base64.b64encode(image_file.read()).decode("utf-8")


        response = await self.nanonet_client.chat.completions.create(
            model="nanonets/Nanonets-OCR2-3B",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {
                            "type": "text",
                            "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=6000
        )

        return response.choices[0].message.content


    async def dotsocr_runpod_predict(self, f_path):
        prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]
        image = Image.open(f_path)
        # https://vjavkcdqrgqyq5-8000.proxy.runpod.net/
        addr = "https://veejutidvzi7xy-8000.proxy.runpod.net/v1" 
        
        # "https://en3mvx70t92s25-8000.proxy.runpod.net/v1"
        # "https://vjavkcdqrgqyq5-8000.proxy.runpod.net/v1"
        dots_ocr_client = AsyncOpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr)
        messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url":  PILimage_to_base64(image)},
                    },
                    {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
                ],
            }
        )
        try:
            response = await dots_ocr_client.chat.completions.create(
                messages=messages, 
                model="rednote-hilab/dots.ocr", 
                max_completion_tokens=8000,
                temperature=0,
                top_p=0.9)
            
            response = response.choices[0].message.content
            return response

        except Exception as e:
            print(f"request error: {e}")
            return None


        # pass
    async def docling_with_surya(self, image_path):

        source = image_path

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=SuryaOcrOptions(lang=["en"]),
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        # result = converter.convert(image_path)
        # print("The result is complete")

        result = await asyncio.to_thread(converter.convert, source)
        print(result.document.export_to_markdown())



        return result.document.export_to_markdown() 





        # pass
    
    async def parsing_mrz_passport(self, image_file_path):
        passport_mrz = await asyncio.to_thread(self.fast_mrz.get_details, image_file_path)

        print("JSON:")
        print(json.dumps(passport_mrz, indent=4))

        return passport_mrz


    def encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_chat_messages(self, image_path: str, prompt: str) -> List[Dict]:
        """
        Create chat messages with image and prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for the model
            
        Returns:
            List of message dictionaries
        """

        # Detect MIME type (jpg/png/webp/etc)
        mime, _ = mimetypes.guess_type(image_path)
        if mime is None:
            mime = "image/jpeg"  # fallback

        return [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(image_path)}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    def process_single_item(self, data: Dict) -> Dict:
        """
        Process a single data item through the VLLM API.
        
        Args:
            client: OpenAI client instance
            data: Input data dictionary
            
        Returns:
            Updated data dictionary with model response
        """
        # Extract image path and prompt
        img_path = data['image_path']
        prompt = data['question']
        
        # Create chat messages
        messages = self.create_chat_messages(img_path, prompt)
        
        # Get model response
        response = self.hunyuan_ocr.chat.completions.create(
            model="tencent/HunyuanOCR",
            messages=messages,
            temperature=0.0,
            top_p=0.95,
            seed=1234,
            stream=False,
            extra_body={
                "top_k": 1,
                "repetition_penalty": 1.0
            }
        )
        
        # Update data with model response
        data["vllm_answer"] = response.choices[0].message.content
        return data



    async def run_hunyuan_predict(self, image_file_path):
        the_message = """
            • Identify the formula in the image and represent it using LaTeX format.

            • Parse the table in the image into HTML.

            • Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.

            • Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.
        """

        my_data : Dict = {
            "image_path": image_file_path,
            "question": the_message
        }

        result_text = await asyncio.to_thread(self.process_single_item, my_data)

        return result_text["vllm_answer"]


    async def numarkdown_runpod_predict(self, image_file_path):

        openai_api_key = "rpa_FPEGQAATGI03GTAQJ94I7I7V1X21UXY3UDXSL7OE610y7c"
        openai_api_base = "https://api.runpod.ai/v2/g21qhyibbeg9jm/openai/v1"

        async_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        # def encode_image(image_path):
        """
        Encode the image file to base64 string
        """

        base64_image = None
        with open(image_file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # base64_image = encode_image("image.jpg")
        data_url = f"data:image/jpeg;base64,{base64_image}"

        chat_response = await async_client.chat.completions.create(
            model="numind/NuMarkdown-8B-Thinking",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {"url": data_url},
                            "min_pixels": 100 * 28 * 28,
                            "max_pixels": 5000 * 28 * 28,
                        },
                    ],
                },
            ]
        )

        result = chat_response.choices[0].message.content
        reasoning = result.split("<think>")[1].split("</think>")[0]
        answer  = result.split("<answer>")[1].split("</answer>")[0]
        print(answer)

        return answer
        # pass

    
    async def typhoon_runpod_predict(self, orig_filename, task_type, page_number):
        """
        Run OCR prediction using Typhoon model deployed on RunPod via vLLM.
        Logs token usage for monitoring.

        Args:
            orig_filename: Path to the PDF or image file
            task_type: OCR task type (e.g., "v1.5")
            page_number: Page number to process

        Returns:
            Extracted markdown text from the document
        """
        # Prepare OCR messages using typhoon_ocr utility
        messages = prepare_ocr_messages(
            pdf_or_image_path=orig_filename,
            task_type="v1.5",
            target_image_dim=1800,
            target_text_length=8000,
            page_num=page_number if page_number else 1,
            figure_language="Thai"
        )

        # Create async client for RunPod vLLM endpoint
        typhoon_client = AsyncOpenAI(
            base_url='https://05j4jhk4yupj58-8000.proxy.runpod.net/v1',
            api_key='0'
        )

        try:
            # Send request to vLLM endpoint
            response = await typhoon_client.chat.completions.create(
                model="typhoon-ocr-1-5",
                messages=messages,
                max_tokens=8000,
                extra_body={
                    "repetition_penalty": 1.1,
                    "temperature": 0,
                    "presence_penalty": 1.5,
                    "top_p": 0.6,
                },
            )

            # Log token usage for monitoring via Docker logs
            if response.usage:
                usage = response.usage
                logger.info(
                    f"[TYPHOON_OCR_TOKEN_USAGE] "
                    f"file={orig_filename} | "
                    f"prompt_tokens={usage.prompt_tokens} | "
                    f"completion_tokens={usage.completion_tokens} | "
                    f"total_tokens={usage.total_tokens}"
                )
                # Also print for immediate visibility in Docker logs
                print(
                    f"[TYPHOON_OCR_TOKEN_USAGE] "
                    f"file={orig_filename} | "
                    f"prompt_tokens={usage.prompt_tokens} | "
                    f"completion_tokens={usage.completion_tokens} | "
                    f"total_tokens={usage.total_tokens}",
                    flush=True
                )
            else:
                logger.warning(f"[TYPHOON_OCR_TOKEN_USAGE] No usage data returned for file={orig_filename}")

            # Extract text content
            text_output = response.choices[0].message.content
            return text_output

        except Exception as e:
            logger.error(f"[TYPHOON_OCR_ERROR] file={orig_filename} | error={str(e)}")
            raise

    def predict(self,image,
            max_new_tokens=8192,
            temperature=0.1,
            max_images_size=3000
            ):
        buffer = ""
        text=""
        for i in self.model.generate(
            """Below is an image of a document page along with its dimensions.
            Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.
            If the document contains images, use a placeholder like dummy.png for each image.
            Your final output must be in JSON format with a single key `natural_text` containing the response.
            Please reply in Markdown format only.
            """,
            image,
            stream=True,
            history_save=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_images_size=max_images_size
            ):
            text+=i
            if ': "' in text and '"}' not in text:
                buffer += i
                if buffer.endswith("\\n"):
                    yield "\n"
                    buffer = "" 
                elif len(buffer) == 2:
                    if not buffer.startswith("\\"):
                        yield buffer[0]
                        buffer = buffer[1]
                elif len(buffer) == 1 and buffer not in "\\":
                    yield buffer
                    buffer = ""

    async def structured_output(self,markdown:str,schema:BaseModel):
        prompt=f"""
        Convert the given markdown text into JSON that align with the following schema:\n {schema.model_json_schema()} 
        Wrap the output in `json` tags. Do not hallucinate. Do not use any information that is not in the markdown text.
        If the information is not available, use empty string.

        Markdown text: {markdown}
        /no_think
        """

        # Single request with thinking mode
        response = await self.q_client.chat_completion([
            {"role": "system", "content": "You are a helpful assistant that converts Markdown to JSON format according to the given schema."},
            {"role": "user", "content": prompt}
        ])

        return response

    def classify_document(self, image: Image.Image) -> str:
        """
        Classifies a document image into 'Invoice', 'Quotation', or 'Other'.

        Args:
            image: The document image as a PIL Image object.
            processor: The DonutProcessor instance.
            model: The VisionEncoderDecoderModel instance.

        Returns:
            A string representing the predicted category.
        """
        # Prepare inputs for the model
        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = self.donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.donut_processor(image, return_tensors="pt").pixel_values

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.donut_model.to(device)
        pixel_values = pixel_values.to(device)
        decoder_input_ids = decoder_input_ids.to(device)

        # Generate output sequence
        outputs = self.donut_model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.donut_model.decoder.config.max_position_embeddings,
            pad_token_id=self.donut_processor.tokenizer.pad_token_id,
            eos_token_id=self.donut_processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.donut_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode and parse the output
        sequence = self.donut_processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.donut_processor.tokenizer.eos_token, "").replace(self.donut_processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        try:
            result = self.donut_processor.token2json(sequence)
            predicted_class = result.get('class', 'unknown').lower()
            
            # Map the model's output to the desired categories
            if "invoice" in predicted_class:
                return "Invoice"
            # The base model does not have a 'quotation' class.
            # This logic can be expanded after fine-tuning.
            # For now, we map common related classes.
            elif predicted_class in ["form", "letter", "specification"]:
                # A simple heuristic: check for keywords if the class is ambiguous
                image.save("temp_for_ocr.png") # Save for text check
                try:
                    import pytesseract
                    text_content = pytesseract.image_to_string(Image.open("temp_for_ocr.png")).lower()
                    if "quotation" in text_content or "quote" in text_content:
                        return "Quotation"
                    elif "analysis" in text_content:
                        return "Analysis Report"
                    
                except ImportError:
                    # If pytesseract is not installed, we can't do the keyword check.
                    pass
                return "Other"
            else:
                image.save("temp_img_pytesract.png")
                try:
                    import pytesseract
                    temp_text_content = pytesseract.image_to_string(Image.open("temp_img_pytesract.png")).lower()
                    if ("invoice" in temp_text_content) or ("กำกับภาษี" in temp_text_content) or ("quotation" in temp_text_content) or ("ใบเสร็จ" in temp_text_content) or ("ใบสั่งซื้อ" in temp_text_content) or ("receipt" in temp_text_content) or ("เสนอราคา" in temp_text_content):
                        return "Invoice"
                    elif "analysis" in temp_text_content:
                        return "Analysis Report"
                    else:
                        return "Other"
                except ImportError:
                    pass

                return "Other"

        except (json.JSONDecodeError, AttributeError, IndexError):
            # If parsing fails, return 'Other'
            return "Other"

    def pdf_to_image(self, pdf_path: str) -> Image.Image:
        """
        Converts the first page of a PDF file to a PIL Image object.

        Args:
            pdf_path: The file path to the PDF document.

        Returns:
            A PIL Image object of the first page, converted to RGB.
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)  # Load the first page
            pix = page.get_pixmap()
            doc.close()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return image
        except Exception as e:
            print(f"Error processing PDF file: {e}")
            # exit(1)
            return None


    def check_if_it_invoice(self, pdf_file_path: str) -> str:    
        # Convert PDF to image
        doc_image = self.pdf_to_image(pdf_file_path)

        if doc_image is not None:
            print("Classifying document...")
            # Classify the document
            category = self.classify_document(doc_image)

            # Print the final output
            print(category)

        else:
            category = "Invoice"


        return category


    def generate_payment_term_date(self, message : str):
        # now = datetime.now().strftime("%A %d %B %Y")
        # generator = Generator(self.ext_model, InvoicePaymentDate)

        # prompt = self.prompt_template(now=now, message=message) 

        result = self.ext_model(message, InvoicePaymentDate)
        print(result) 
        r = InvoicePaymentDate.model_validate_json(result)
        d = r.payment_day
        m = r.payment_month
        y = r.payment_year
        if (d is not None) and (m is not None) and (y is not None):
            temp_payment_date = f"{d}/{m}/{y}"
        else:
            temp_payment_date = None

        # date_to_return = generator(prompt)
        # temp_payment_date = None
        # print("--------------------------------")
        # print(date_to_return)
        # print(type(date_to_return))
        # print("--------------------------------")
        # Extract the date if available
        # if isinstance(date_to_return, dict) and "payment_term_date" in date_to_return:
            # temp_payment_date = date_to_return["payment_term_date"]
        # elif hasattr(date_to_return, 'payment_term_date'):
            # temp_payment_date = date_to_return.payment_term_date

        return temp_payment_date


    def generate_invoice_term_date(self, message : str):
        # now = datetime.now().strftime("%A %d %B %Y")
        # generator = Generator(self.ext_model, InvoiceIssueDate)

        result = self.ext_model(message, InvoiceIssueDate)
        print(result) 
        r = InvoiceIssueDate.model_validate_json(result)
        # prompt = self.invoice_prompt_template(now=now, message=message) 
        # date_to_return = generator(prompt)
        # temp_invoice_date = None
        # print("--------------------------------")
        # print(date_to_return)
        # print("--------------------------------")
        dd = r.invoice_day
        mm = r.invoice_month
        yy = r.invoice_year
        # temp_invoice_date = datetime(yy, mm, dd)

        # concat the dd, mm, yy together using '/' as the separator
        if (dd is not None) and (mm is not None) and (yy is not None):
            temp_invoice_date = f"{dd}/{mm}/{yy}"
        else:
            temp_invoice_date = None

        # Extract the date if available
        # if isinstance(date_to_return, dict) and "invoice_issue_date" in date_to_return:
        #     temp_invoice_date = date_to_return["invoice_issue_date"]
        # elif hasattr(date_to_return, 'invoice_issue_date'):
        #     temp_invoice_date = date_to_return.invoice_issue_date

        return temp_invoice_date
    

    
    # def structured_output(self, markdown:str, schema:BaseModel):
    #     prompt=f"""
    #     The Markdown text to translate into JSON is as follows:

    #     Markdown text: {markdown}
    #     """
    #     response = self.llm.chat(
    #         model=self.llm_model,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": f"You are a helpful assistant that understands and translates Markdown to JSON format according to the following schema. {schema.model_json_schema()}"
    #             },
    #             {
    #                 "role": "user", 
    #                 "content": prompt
    #             }
    #         ],
    #         format=schema.model_json_schema()

    #         # stream=True
    #     )

    #     # answer = schema.model_validate_json(response.message.content)
    #     answer = response.message.content
    #     print(answer)
    #     return answer
    
        # for chunk in stream:
        #     if 'message' in chunk and 'content' in chunk['message']:
        #         yield chunk['message']['content']
