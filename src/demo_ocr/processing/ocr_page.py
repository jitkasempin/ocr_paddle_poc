import random
from unittest import result
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
from typing import List, Dict, Any, Optional, Literal, Tuple
from decimal import Decimal
from datetime import datetime
from supabase import create_client, Client
import io
from pydantic import BaseModel, Field, model_validator, field_validator
from ultralytics import YOLO
from .ocr_no_flash import OCR
from .invoice_integration import flag_repetitive_output

from .schema_helper import parse_decimal_like, parse_thai_date, extract_code, extract_all_codes, extract_only_branch_code_number, extract_po_decimal
from .hybrid_search import HybridSearch
from pathlib import Path
# from paddleocr import PPStructureV3
from doc_classification.zero_shot import get_classifier,crop_top_percent
from bs4 import BeautifulSoup
import requests

import time
import json, re
import pandas as pd

import cv2
import numpy as np
from core.pdf2md.pdf2md import convert_to_markdown_stream
import uuid


def load_centroids_dict(path: str) -> Dict[str, np.ndarray]:
    """
    Load กลับมาเป็น dict[label -> vector] เหมือนตอน build
    """
    blob = np.load(path, allow_pickle=True)
    labels = blob["labels"].tolist()
    C = blob["centroids"]  # (K, D)
    centroids = {lbl: C[i] for i, lbl in enumerate(labels)}
    # (optional) meta = dict(blob["meta"].item()) if blob["meta"].size else {}
    print(f"📂 Loaded {len(labels)} centroids from {path}")
    return centroids



def preprocess_for_improve_ocr(image_path: str, output_path: str = None) -> np.ndarray: 
    # Load image 
    img = cv2.imread(image_path) 
 
    # Convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Apply Gaussian blur to reduce noise 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
 
    # Adaptive thresholding (Gaussian) 
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2) 

    # Deskewing: detect angle and rotate 
    coords = np.column_stack(np.where(binary > 0)) 
    angle = cv2.minAreaRect(coords)[-1] 
    if angle < -45: 
        angle = -(90 + angle) 
    else: 
        angle = -angle 

    (h, w) = gray.shape[:2] 
    center = (w // 2, h // 2) 
    M = cv2.getRotationMatrix2D(center, angle, 1.0) 
    deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE) 

    # Morphological operations to clean 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) 
    cleaned = cv2.morphologyEx(deskewed, cv2.MORPH_CLOSE, kernel) 

    # Add border padding 
    padded = cv2.copyMakeBorder(cleaned, 20, 20, 20, 20, 
                                cv2.BORDER_CONSTANT, value=255) 

    # Optional: resize for higher DPI simulation 
    processed = cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if output_path: 
        cv2.imwrite(output_path, processed)

    return processed



def load_image_as_rgb_array(path: str) -> np.ndarray:
    # เปิดรูปภาพด้วย PIL
    img = Image.open(path).convert("RGB")
    # แปลงเป็น numpy array
    arr = np.array(img)
    return arr

# def extract_code(text: str) -> Optional[str]:
#     """
#     Extract numeric code value from text containing CODE or MAT CODE keywords.

#     Args:
#         text: Input string that may contain code information

#     Returns:
#         Extracted numeric code as string, or None if not found

#     Examples:
#         >>> extract_code("Screw Cap SS/SUS304 (200/กล) CODE : 46924")
#         '46924'
#         >>> extract_code("สกรูหัวเหลี่ยม SS/SUS304 M14-2.0 (100/กล) MAT CODE : 1166")
#         '1166'
#         >>> extract_code("Product code: 12345")
#         '12345'
#         >>> extract_code("No code here")
#         None
#     """
#     if not text:
#         return None

#     # Pattern explanation:
#     # (?:mat\s+)?  - Optional "mat" followed by whitespace (non-capturing group)
#     # code         - The word "code" (case-insensitive due to re.IGNORECASE)
#     # \s*          - Optional whitespace
#     # [:：]?       - Optional colon (ASCII or full-width)
#     # \s*          - Optional whitespace
#     # (\d+)        - Capture group: one or more digits (0-9)
#     pattern = r'(?:mat\s+)?code\s*[:：]?\s*(\d+)'

#     match = re.search(pattern, text, re.IGNORECASE)

#     if match:
#         return match.group(1)

#     return None

class Properties(BaseModel):
    parameters: str = Field(None, description="Substance or Matter that used in the analysis")
    result: str = Field(None, description="Result from the parameters")
    unit: str = Field(None, description="The unit of the parameters")
    limit: str = Field(None, description="Limit of the parameters")
    method: str = Field(None, description="Test Method or Method")


class ItemInvoice(BaseModel):
    item_description: str = Field("", description="description ของสินค้า หรือ รายการสินค้า หรือ รายการ หรือ รายละเอียด")
    quantity: float = Field(0, description="จำนวน หรือ Quantity หรือ ปริมาณ")
    unit: str = Field("", description="ข้อความที่บ่งบอกถึงรูปแบบการขายสินค้า มักจะอยู่คู่กับปริมาณหรือ quantity หรือข้อความที่อยู่ใน column unit หรือ column หน่วย")
    unit_price: float = Field(0, description="ราคาต่อหน่วย หรือ หน่วยละ หรือ Unit Price")
    price: float = Field(0, description="จำนวนเงิน หรือ Amount หรือ ราคาของสินค้า")


class PackingListItem(BaseModel):
    item_description: str = Field("", description="description ของสินค้า หรือ รายการสินค้า หรือ รายการ หรือ รายละเอียด")
    quantity: float = Field(0, description="จำนวน หรือ Quantity หรือ ปริมาณ")
    unit: str = Field("", description="ข้อความที่บ่งบอกถึงรูปแบบการขายสินค้า มักจะอยู่คู่กับปริมาณหรือ quantity หรือข้อความที่อยู่ใน column unit หรือ column หน่วย")
    # unit_price: float = Field(0, description="ราคาต่อหน่วย หรือ หน่วยละ หรือ Unit Price")
    # price: float = Field(0, description="จำนวนเงิน หรือ Amount หรือ ราคาของสินค้า")


class PoItem(BaseModel):
    description: str = Field("", description="Description หรือ รายละเอียด หรือ รหัสสินค้า หรือ รายการ หรือ Description of goods")
    quantity: float = Field(0, description="จำนวน หรือ Quantity หรือ Qty หรือ Order Quantity")
    uom: str = Field("", description="UOM หรือ Unit of Measure")
    unit_price: float = Field(0, description="ราคาต่อหน่วย หรือ ราคา/หน่วย หรือ หน่วยละ หรือ Unit Price หรือ Unit")
    amount: float = Field(0, description="จำนวนเงิน หรือ Amount หรือ Total หรือ Total Amount หรือ Total Price")


class PassPortData(BaseModel):
    passport_number: str = Field(..., description="Unique passport identifier")
    # document_type: PassportType = Field(default=PassportType.PASSPORT)
    issuing_country: str = Field(..., description="ISO 3166-1 alpha-3 country code")
    issuing_authority: Optional[str] = Field(None, max_length=100)
    
    # Personal Information
    surname: str = Field(..., min_length=1, max_length=50, description="Last name/family name")
    given_names: str = Field(..., min_length=1, max_length=100, description="First and middle names")
    nationality: str = Field(..., description="Nationality as ISO 3166-1 alpha-3 code")
    
    # Birth Information
    date_of_birth: str = Field(..., description="Date of birth")
    place_of_birth: Optional[str] = Field(None, max_length=100)
    country_of_birth: Optional[str] = None
    
    # Physical Characteristics
    gender: str = Field(..., description="Gender marker")
    
    # Document Validity
    date_of_issue: str = Field(..., description="Passport issue date")
    date_of_expiry: str = Field(..., description="Passport expiry date")
    

class InvoiceOutput(BaseModel):
    company: str = Field(None, description="ชื่อบริษัทที่ออก Invoice หรือบริษัทที่ขายของ")
    invoice_number: str = Field(None, description="เลขที่ใบส่งของ หรือเลขที่ใบกำกับ หรือเลขที่ใบกำกับภาษี หรือ INVOICE NO. หรือ INV NO. หรือ เลขที่เฉยๆ ก็ได้")
    invoice_date: str = Field("0/0/0", description="วันที่ออกใบแจ้งหนี้ หรือวันที่ ที่ได้เขียนไว้บน Invoice")
    tax_id: str = Field(None, description="เป็นเลขประจำตัวผู้เสียภาษี ของบริษัทที่ออก Invoice เท่านั้น")
    items_list: list[ItemInvoice] = Field(description="List ของรายการสินค้าที่อยู่ใน Invoice นี้เท่านั้น ไม่เอาจำนวนรวมของสินค้า หรือส่วนลดใดๆ ทั้งสิ้น")
    purchase_order_number: str = Field(None, description="เลขที่ใบสั่งซื้อ หรือ PO NO. หรือ P/O NO. หรือ เลขที่ PO และต้องไม่ใช่เลขเดียวกับเลขที่ใบส่งของ(INVOICE NUMBER)")
    payment_date: str = Field("0/0/0", description="คือ วันที่ครบกำหนดชำระเงิน หรือ Payment Due Date หรือ วันครบกำหนด หรือ กำหนดชำระเงิน หรือ ครบกำหนด หรือ Due Date (แต่ต้องไม่ใช่ เครดิต หรือ เงื่อนไขการชำระเงิน)")
                              
class PackingList(BaseModel):
    table: list[PackingListItem] = Field(description="รายการสินค้ารวมถึงปริมาณที่อยู่ใน Packling List")


class PurchaseOrder(BaseModel):
    purchase_order_number: str = Field(default="", description="Purchase Order Number หรือ เลขที่ใบสั่งซื้อ หรือ PO NO หรือ P/O No หรือ ใบสั่งซื้อเลขที่")
    date: str = Field(default="", description="Purchase Order Date หรือ PO Date หรือ วันที่")
    credit_term: str = Field(default="", description="เครดิต หรือ credit term หรือ payment term หรือ การชำระเงิน หรือ เงื่อนไข")
    company_name: str = Field(default="", description="Company name at the top of the invoice. The company name must not be 'THAI KK INDUSTRY' or 'ไทย เคเค อุตสาหกรรม'")
    tax_id: str = Field(default="", description="เลขประจำตัวผู้เสียภาษี หรือ เลขประจำตัวผู้เสียภาษีอากร หรือ TAX ID หรือ VAT Registration No")
    items_list: list[PoItem] = Field(description="List of items in the Purchase Order. Each item should have description, quantity, unit price, and amount")
    # email_address: str = Field(default="", description="Email Address of Contact Person in the Purchase Order.")

class AnalysisReport(BaseModel):
    sampling_date: str = Field(None, description="วันที่เริ่มการ sampling หรือ Sampling Date")
    sampling_by: str = Field(None, description="ผู้ที่ทำการ sampling หรือ Sampling By")
    report_number: str = Field(None, description="เลขที่ของ Analysis Report หรือ Report No.")
    analysis_parameters: list[Properties] = Field(None, description="List of Substance or Matter that used in the analysis")
    remark: str = Field(None, description="Remark of the Analysis Report")


class Receipt(BaseModel):
    store_name: str = Field(None, description="ชื่อร้านค้า หรือ ชื่อบริษัท ที่ออกใบเสร็จ")
    receipt_number: str = Field(None, description="เลขที่ใบเสร็จ หรือ Receipt Number")
    date: str = Field(None, description="วันที่ที่แสดงบนใบเสร็จ หรือ Date of purchase")
    time: str = Field(None, description="เวลาที่แสดงบนใบเสร็จ หรือ Time of purchase")
    items: list[ItemInvoice] = Field(description="รายการสินค้าที่ซื้อ พร้อมราคาและจำนวน")
    total_amount: float = Field(None, description="จำนวนเงินรวมทั้งหมด หรือ Total amount")
    payment_method: str = Field(None, description="วิธีการชำระเงิน เช่น เงินสด, บัตรเครดิต, บัตรเดบิต หรือ Payment method")
    tax_id: str = Field(None, description="เลขประจำตัวผู้เสียภาษีของร้านค้า หรือ Tax ID")
    address: str = Field(None, description="ที่อยู่ของร้านค้า หรือ Store address")

# from pydantic import BaseModel, Field
# from typing import List, Optional, Literal
# from pydantic import BaseModel, Field, field_validator, model_validator



# from .schema_helper import parse_decimal_like, parse_thai_date, extract_code, extract_all_codes


class Document(BaseModel):
    invoice_number: str = Field(
        default="", description="เลขที่ หรือ เลขที่ใบกำกับ หรือ เลขที่ใบกำกับภาษี หรือ invoice number หรือ invoice no หรือ inv no")
    po_number: str = Field(
        default="", description="ใบสั่งซื้อ หรือ P/O NO หรือ เลขที่ใบสั่งซื้อ หรือ PO NO หรือ Purchase order number หรือ เลขที่ PO")
    date: str = Field(
        default="", description="วันที่ออกใบแจ้งหนี้ หรือวันที่ ที่ได้เขียนไว้บน Invoice")
    document_name: str = Field(
        default="", description="ชื่อเอกสาร สามารถมีค่าต่างๆ ได้เช่น ต้นฉบับใบกำกับภาษี TAX INVOICE หรือ ต้นฉบับใบแจ้งหนี้ หรือ ใบกำกับภาษี หรือ Original Invoice หรือ ต้นฉบับใบเสร็จรับเงิน หรือ ใบแจ้งหนี้")
    payment_due_date: str = Field(
        default="0/0/0", description="คือ วันที่ครบกำหนดชำระเงิน หรือ Payment Due Date หรือ วันครบกำหนด หรือ กำหนดชำระเงิน หรือ ครบกำหนด หรือ Due Date (แต่ต้องไม่ใช่ เครดิต หรือ เงื่อนไขการชำระเงิน)")
    payment_terms: str = Field(
        default="", description="Credit หรือ Payment terms หรือ เงื่อนไขการชำระเงิน หรือ เครดิต หรือ เงื่อนไขเครดิต หรือ เงื่อนไขการชำระเงิน เช่น เครดิต 30 วัน")

    @field_validator("date", "payment_due_date", mode="before")
    @classmethod
    def _heal_all_dates(cls, v):
        parsed_dated = parse_thai_date(v)
        if parsed_dated is None:
            parsed_dated = "00.00.0000"

        return parsed_dated

    @field_validator("po_number", mode="before")
    @classmethod
    def _heal_po_number(cls, v):
        parsed_for_po = extract_po_decimal(v)
        if parsed_for_po is None:
            parsed_for_po = ""

        return parsed_for_po


class SellerCompany(BaseModel):
    name: str = Field(
        default="", description="Company name at the top of the invoice")
    tax_id: str = Field(
        default="", description="เลขประจำตัวผู้เสียภาษี หรือ Tax ID")
    address: str = Field(
        default="", description="ที่อยู่ของบริษัท หรือที่ตั้งของสำนักงานใหญ่บริษัท")
    contact: Optional[str] = Field(
        default=None, description="ข้อมูลการติดต่อ เช่น เบอร์โทรศัพท์ หรือ อีเมล หรือ Email หรือ phone number")
    branch_name: Optional[str] = Field(
        default=None, description="ชื่อสาขา หรือ branch name")
    branch_code: Optional[str] = Field(
        default=None, description="สาขาที่ หรือ สาขา หรือ branch number")


class CustomerCompany(BaseModel):
    name: str = Field(
        default="", description="ชื่อลูกค้า หรือ นามลูกค้า หรือ ลูกค้า หรือ ขายให้ หรือ ผู้ซื้อ หรือ CUSTOMER หรือ SOLD TO หรือ ชื่อ หรือ BILL TO หรือ SHIP TO")
    tax_id: str = Field(
        default="", description="เลขประจำตัวผู้เสียภาษี หรือ เลขประจำตัวผู้เสียภาษีอากร หรือ เลขประจำตัวผู้เสียภาษีอากรของผู้ซื้อ หรือ Tax ID ของลูกค้า")
    address: str = Field(
        default="", description="ที่อยู่ลูกค้า หรือ ที่อยู่ หรือ Address")
    contact: Optional[str] = Field(
        default=None, description="ข้อมูลการติดต่อ เช่น เบอร์โทรศัพท์ หรือ อีเมล หรือ Email หรือ phone number")
    branch_name: Optional[str] = Field(
        default=None, description="ชื่อสาขา หรือ branch name")
    branch_code: Optional[str] = Field(
        default=None, description="สาขาที่ หรือ สาขา หรือ branch number")

    @field_validator("branch_code", mode="before")
    @classmethod
    def _heal_branch_code(cls, v):
        if v is not None:
            return extract_only_branch_code_number(v)
        return v


class Item(BaseModel):
    code: str = Field(default="", description="Product code")
    description: str = Field(
        default="", description="description ของสินค้า หรือ รายการสินค้า หรือ รายการ หรือ รายละเอียด หรือ รายละเอียดสินค้า")
    unit_price: Decimal = Field(default=Decimal(
        "0.00"), description="ราคาต่อหน่วย หรือ หน่วยละ หรือ Unit Price")
    uom: str = Field(
        default="", description="ข้อความที่บ่งบอกถึงรูปแบบการขายสินค้า มักจะอยู่คู่กับปริมาณหรือ quantity หรือข้อความที่อยู่ใน column unit หรือ column หน่วย")
    quantity: int = Field(
        default=0, description="จำนวน หรือ Quantity หรือ ปริมาณ")
    discount: Decimal = Field(default=Decimal(
        "0.00"), description="ส่วนลด หรือ Discount amount")
    amount: Decimal = Field(default=Decimal(
        "0.00"), description="ราคารวม หรือ amount หรือ Total amount for this item")
    note: str = Field(
        default="-", description="หมายเหตุ หรือ Additional notes")
    # --- self-healing for Decimal fields ---

    @field_validator("unit_price", "discount", "amount", mode="before")
    @classmethod
    def _heal_item_decimals(cls, v):
        return parse_decimal_like(v)

    @model_validator(mode="after")
    def extract_code_from_description(self):
        if self.description:
            # Add implementation code here
            tmp_return_code = extract_code(self.description)
            if tmp_return_code is not None:
                self.code = tmp_return_code
            else:
                self.code = ""

            # parts = self.description.split()
            # if parts:
            #     self.code = parts[0]

        return self


class Summary(BaseModel):
    subtotal: Decimal = Field(default=Decimal(
        "0.00"), description="Subtotal before tax")
    discount_total: Decimal = Field(
        default=Decimal("0.00"), description="Total discount")
    vat_rate: int = Field(default=7, description="VAT rate percentage")
    vat: Decimal = Field(
        default=Decimal("0.00"), description="VAT amount")
    total_amount: Decimal = Field(default=Decimal(
        "0.00"), description="Total amount including VAT")
    currency: Literal["BAHT", "THB", "บาท", "US DOLLAR", "USD", "SING.DOLLAR", "SGD", "POUND STERING", "EURO", "GBP", "RUPEE", "INR", "SWISS FRANC", "CHF", "CHINESE YUAN", "CNY", "YEN", "JPY", "CANADIAN DOLLAR", "CAD"] = Field(
        default="THB", description="สกุลเงินที่ใช้ใน Invoice เช่น BAHT หรือ THB หรือ บาท หรือ US DOLLAR หรือ USD หรือ SING.DOLLAR หรือ SGD หรือ POUND STERING หรือ EURO หรือ GBP หรือ RUPEE หรือ INR หรือ SWISS FRANC หรือ CHF หรือ CHINESE YUAN หรือ CNY หรือ YEN หรือ JPY หรือ CANADIAN DOLLAR หรือ CAD")
    # --- self-healing for Decimal fields ---

    @field_validator("subtotal", "discount_total", "vat", "total_amount", mode="before")
    @classmethod
    def _heal_summary_decimals(cls, v):
        return parse_decimal_like(v)


class Invoice(BaseModel):
    document: Document = Field(
        default_factory=Document, description="General invoice information consist of invoice number, purchase order number or reference number and invoice date")
    seller: SellerCompany = Field(default_factory=SellerCompany,
                                  description="The information about company that issue the invoice (company that sell the products to customer)")
    customer: CustomerCompany = Field(
        default_factory=CustomerCompany, description="The information about the company that buy the products from seller")
    items: List[Item] = Field(
        default_factory=list, description="List ของรายการสินค้าที่อยู่ใน Invoice นี้ และต้องเป็นรายการสินค้าที่มี Quantity มากกว่า 0 เท่านั้นด้วย")
    summary: Summary = Field(default_factory=Summary,
                             description="Invoice summary")



class DeltaItem(BaseModel):
    po_no: str = Field(
        default="", description="""The value of the column "P.O. NO." from the HTML table of this invoice.""")
    qty: int = Field(
        default=0, description="""The value of the column "QTY" from the HTML table of this invoice.""")
    unit_price: Decimal = Field(default=Decimal(
        "0.00"), description="""The value of the column "UNIT PRICE" from the HTML table of this invoice.""")
    amount: Decimal = Field(default=Decimal(
        "0.00"), description="""The value of the column "AMOUNT" from the HTML table of this invoice.""")


class DeltaInvoice(BaseModel):
    invoice_number: str = Field(default="", description="""The invoice number from this invoice text content. It is the decimal number after "Invoice No.:" text.""")
    invoice_date: str = Field(default="0/0/0", description="""The invoice date from this invoice text content. It is date string in the format of DD/MM/YYYY after "Invoice Date:" text.""")
    invoice_to: str = Field(default="", description="""The value of "Invoice to:" from this invoice text content. 
                                                    It must begin with 'D' character follow by 7 decimal numbers and the text that denote the company name and branch.
                                                    Grab all the content after "Invoice to:" text and stop when you found "Address:" text.""")
    address: str = Field(default="", description="""The address of the company after "Address:" text and before "Attn. to:" text.""")
    


class AllItemsInDelta(BaseModel):    
    items: List[DeltaItem] = Field(default_factory=list, description="List of all items in this DELTA invoice")


# class InvoiceIssueDate(BaseModel):
#     invoice_day: int = Field(
#         default=None, description="The day part of the invoice issued date. Must have the value between 1 and 31"
#     )
#     invoice_month: int = Field(
#         default=None, description="The month part of the invoice issued date. Must have the value between 1 and 12. Convert the month name in English or month name in three english letters abbreviation (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec) or month name in Thai (มกราคม, กุมภาพันธ์, มีนาคม, เมษายน, พฤษภาคม, มิถุนายน, กรกฎาคม, สิงหาคม, กันยายน, ตุลาคม, พฤศจิกายน, ธันวาคม) to the number 1 - 12"
#     )
#     invoice_year: int = Field(
#         default=None, description="The year part of the invoice issued date. The year can be Common Era (CE) or Buddhist Era (BE) and can be in 2 or 4 digits. The year must be in the range of 2010 - 2025 (or 10 - 25) for CE year and 2543 - 2568 (or 43 - 68) for BE year"
#     )

class StakeHolders(BaseModel):
    stakeholder_name: str = Field(default="", description="ชื่อผู้ถือหุ้น หรือ รายชื่อผู้ถือหุ้น (สามารถเป็นชื่อบุคคล หรือชื่อบริษัทก็ได้)")
    stock_amount: Decimal = Field(default=Decimal("0.00"), description="จำนวนหุ้นที่ถือ")
    stakeholder_nationality: str = Field(default="", description="สัญชาติ")



class CompanyStock(BaseModel):
    company_name: str = Field(default="", description="ชื่อบริษัทจำกัด")
    register_number: str = Field(default="", description="ทะเบียนเลขที่")
    company_stakeholders: List[StakeHolders] = Field(default_factory=list, description="รายชื่อผู้ถือหุ้นของบริษัท")
    thai_stakeholders_number: str = Field(default="-", description="ผู้ถือหุ้น ไทย")
    other_stakeholders_number: str = Field(default="-", description="หุ้น อื่นๆ")

class Certificate_DBD(BaseModel):
    company_name: str = Field(default="", description="ชื่อบริษัท ยกตัวอย่างเช่น 'บริษัท เฉิงหลัน เทคโนโลยี จำกัด'")
    certificate_issued_date: str = Field(default="", description="วันที่ออกใบอนุญาต ยกตัวอย่างเช่น 'ออกให้ ณ วันที่ 5 เดือน เมษายน พ.ศ. 2567'")
# class InvoicePaymentDate(BaseModel):
#     payment_day: int = Field(
#         default=None, description="The day part of the payment term date. Must have the value between 1 and 31"
#     )
#     payment_month: int = Field(
#         default=None, description="The month part of the payment term date. Must have the value between 1 and 12. Convert the month name in English or month name in three english letters abbreviation (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec) or month name in Thai (มกราคม, กุมภาพันธ์, มีนาคม, เมษายน, พฤษภาคม, มิถุนายน, กรกฎาคม, สิงหาคม, กันยายน, ตุลาคม, พฤศจิกายน, ธันวาคม) to the number 1 - 12"
#     )
#     payment_year: int = Field(
#         default=None, description="The year part of the payment term date. The year can be Common Era (CE) or Buddhist Era (BE) and can be in 2 or 4 digits. The year must be in the range of 2010 - 2025 (or 10 - 25) for CE year and 2543 - 2568 (or 43 - 68) for BE year"
#     )


class FixedLangExtractProcessor:
    """Enhanced metadata extraction with better prompts and normalization"""
    
    def __init__(self):
        try:
            import langextract as lx
            self.lx = lx
            self.setup_complete = True
            print("✅ LangExtract initialized")
        except ImportError:
            print("⚠️  LangExtract not installed - using enhanced regex extraction")
            self.setup_complete = False
    
        
    def extract_metadata(self, document: str) -> Dict:
        """Extract and normalize metadata"""
        
        if not self.setup_complete:
            return None # self._enhanced_regex_extraction(documents)

        # Improved extraction prompt
        prompt = """
        Extract these specific fields from certificate documentation:
        
        1. name: ชื่อของคนที่ปรากฎในหนังสือรับรองนี้
        2. certification_number: เลขทะเบียนใบอนุญาต
        3. allowance_date: ได้รับใบอนุญาตครั้งแรกตั้งแต่วันที่เท่าไร 
        4. work_type: ประเภทงาน
        5. work_description: งานที่รับผิดชอบ
        6. project_owner: เจ้าของ หรือ เจ้าของโครงการ
        
        Be very precise - extract the EXACT data contained in the certificate document."""

        
        # Better examples
        examples = [
            self.lx.data.ExampleData(
                text=f"""ที่ D-COE๐๖๙๗๒๖/๒๕๖๗

                ๑๖๑๖/๑ ถนนลาดพร้าว แขวงวังทองหลาง
                เขตวังทองหลาง กรุงเทพมหานคร ๑๐๓๑๐ สายด่วน ๑๓๐๓
                โทรสาร ๐-๒๙๓๕-๖๖๙๕, ๐-๒๙๓๕-๖๖๙๗
                www.coe.or.th

                หนังสือรับรอง

                หนังสือรับรองฉบับนี้ให้ไว้เพื่อรับรองว่า นายภณ สุวรรณไกรษร เลขทะเบียนใบอนุญาต ภย.๗๖๕๙๒ เป็นผู้ได้รับใบอนุญาตประกอบวิชาชีพวิศวกรรมควบคุม ระดับภาคีวิศวกร สาขาวิศวกรรมโยธา ได้รับใบอนุญาตครั้งแรกตั้งแต่วันที่ ๘ มีนาคม ๒๕๖๔ ใบอนุญาตประกอบวิชาชีพวิศวกรรมควบคุมฉบับปัจจุบันออกให้ตั้งแต่วันที่ ๘ มีนาคม ๒๕๖๔ ถึง ๗ มีนาคม ๒๕๖๙ ขณะนี้ไม่ได้ถูกพักใช้หรือเพิกถอนใบอนุญาตประกอบวิชาชีพวิศวกรรมควบคุม

                ให้ไว้ ณ วันที่ ๑๓ พฤษภาคม ๒๕๖๗

                สภาวิศวกร

                หมายเหตุ หนังสือฉบับนี้ให้ใช้ภายใน ๑๒๐ วัน นับแต่วันที่ออกหนังสือ

                ข้อมูลสรุปตามที่ระบุไว้ในคำขอหนังสือรับรองนี้ เพื่อใช้ในการยื่นคำขออนุญาตตามแบบ ข.1 - ข.7
                ประเภทงาน: งานควบคุมการสร้างหรือการผลิต
                งานที่รับผิดชอบ: ก่อสร้าง
                สิ่งปลูกสร้างชนิด: งานทาง โครงการซ่อมสร้างถนนแอสฟัลต์คอนกรีต ถนนสุขาภิบาล 1
                เจ้าของ: เทศบาลตำบลเขื่อนอุบลรัตน์

                รายละเอียดเพิ่มเติม โปรดตรวจสอบตาม QR CODE ท้ายหนังสือรับรองฉบับนี้

                คำเตือน : หนังสือรับรองฉบับนี้พิมพ์จากต้นฉบับที่เป็นไฟล์อิเล็กทรอนิกส์ ภายใต้การรับรอง Digital Certificate

                สภาวิศวกร
                COUNCIL OF ENGINEERS

                โดยสามารถตรวจสอบด้วยเลข Ref No. ผ่านเว็บไซต์
                www.coe.or.th หรือตรวจสอบผ่าน QR CODE

                ออกให้ ณ วันที่ 2024-05-13 14:16:49
                Ref : 674071455
                """,
                extractions=[
                    self.lx.data.Extraction(
                        extraction_class="name",
                        extraction_text="นายภณ สุวรรณไกรษร",
                        attributes={}
                    ),
                    self.lx.data.Extraction(
                        extraction_class="certification_number", 
                        extraction_text="ภย.๗๖๕๙๒",
                        attributes={}
                    ),
                    self.lx.data.Extraction(
                        extraction_class="allowance_date",
                        extraction_text="๘ มีนาคม ๒๕๖๔", 
                        attributes={}
                    ),
                    self.lx.data.Extraction(
                        extraction_class="work_type",
                        extraction_text="งานควบคุมการสร้างหรือการผลิต",
                        attributes={}
                    ),
                    self.lx.data.Extraction(
                        extraction_class="work_description",
                        extraction_text="ก่อสร้าง",
                        attributes={}
                    ),
                    self.lx.data.Extraction(
                        extraction_class="project_owner",
                        extraction_text="เทศบาลตำบลเขื่อนอุบลรัตน์",
                        attributes={}
                    )
                ]
            )
        ]

        
        extracted_docs = []
        
        # for doc in documents:
            # print(f"📄 Processing: {doc['title']}")
            
        try:
            result = self.lx.extract(
                text_or_documents=document,
                prompt_description=prompt,
                examples=examples,
                model_id="gemini-2.5-pro",
                api_key="AIzaSyAaJcvCSi4s9FvVi5JGSzkEQ8uP_45tttw",  
                extraction_passes=2
            )
                
            # Process and normalize extractions
            print(f"The result from extraction: {result.extractions}")

            metadata = self._process_and_normalize(result.extractions, document)
                
        except Exception as e:
            print(f"  ⚠️  LangExtract failed: {e}")
                # metadata = self._enhanced_regex_extraction([doc])[0]['metadata']
            
            # extracted_docs.append({
                # 'id': doc['id'],
                # 'title': doc['title'],
                # 'content': doc['content'],
                # 'metadata': metadata
            # })
        
        return metadata
   
    
    def _process_and_normalize(self, extractions, document: str) -> Dict:
        """Process LangExtract results and normalize them"""
        
        metadata = {
            'name': 'unknown',
            'certification_number': 'unknown', 
            'allowance_date': 'unknown',
            'work_type': 'unknown',
            'work_description': 'unknown',
            'project_owner': 'unknown'
        }
        
        for extraction in extractions:
            if extraction.extraction_class == "name":
                metadata['name'] = extraction.extraction_text
            elif extraction.extraction_class == "certification_number":
                metadata['certification_number'] = extraction.extraction_text
            elif extraction.extraction_class == "allowance_date":
                metadata['allowance_date'] = extraction.extraction_text
            elif extraction.extraction_class == "work_type":
                metadata['work_type'] = extraction.extraction_text
            elif extraction.extraction_class == "work_description":
                metadata['work_description'] = extraction.extraction_text
            elif extraction.extraction_class == "project_owner":
                metadata['project_owner'] = extraction.extraction_text
        
        # Fallback to regex if LangExtract missed key fields
        # if metadata['service'] == 'unknown' or metadata['version'] == 'unknown':
        #     regex_metadata = self._enhanced_regex_extraction([doc])[0]['metadata']
        #     if metadata['service'] == 'unknown':
        #         metadata['service'] = regex_metadata['service']
        #     if metadata['version'] == 'unknown':
        #         metadata['version'] = regex_metadata['version']
        #     if metadata['doc_type'] == 'reference':
        #         metadata['doc_type'] = regex_metadata['doc_type']
        
        return metadata
    
    def _enhanced_regex_extraction(self, documents: List[Dict]) -> List[Dict]:
        """Enhanced regex-based extraction with better patterns"""
        
        extracted_docs = []
        
        for doc in documents:
            metadata = {
                'service': 'unknown',
                'version': 'unknown',
                'doc_type': 'reference', 
                'rate_limits': [],
                'deprecated': False
            }
            
            title = doc.get('title', '')
            content = doc['content']
            
            # Extract service name from title
            service_match = re.search(r'([\w\s]+(?:API|Service))', title)
            if service_match:
                metadata['service'] = service_match.group(1).strip()
            
            # Extract version number
            version_match = re.search(r'v?([\d.]+)', title)
            if version_match:
                metadata['version'] = version_match.group(1)
            
            # Determine document type
            if 'troubleshooting' in title.lower():
                metadata['doc_type'] = 'troubleshooting'
            elif 'guide' in title.lower():
                metadata['doc_type'] = 'guide'
            else:
                metadata['doc_type'] = 'reference'
            
            # Extract rate limits
            rate_matches = re.findall(r'(\d+)\s*(?:requests?|req)[/\s]*(?:per\s*)?min', content.lower())
            metadata['rate_limits'] = [f"{r} req/min" for r in rate_matches]
            
            # Check for deprecation
            if 'deprecated' in content.lower():
                metadata['deprecated'] = True
            
            extracted_docs.append({
                'id': doc['id'],
                'title': doc['title'], 
                'content': doc['content'],
                'metadata': metadata
            })
        
        return extracted_docs



@st.cache_resource
def load_model()->tuple[OCR, YOLO, Dict[str, np.ndarray]]:
    # from .ocr import OCR
    x = OCR(ocr_model="FILM6912/typhoon-ocr-7b",llm_model="qwen3:14b", load_in_4bit=False)
    # signature=YOLO("C:\\Users\\User\\wb_project\\data\\yolo12l_27062568.pt")
    signature=YOLO("/data/yolo12l_27062568.pt")

    OCR_CENTROIDS_PATH =  "/data/centroids_bank.npz"

    centroids = load_centroids_dict(OCR_CENTROIDS_PATH)

    hb_search = HybridSearch()
    hb_search.build_collection_with_tenant_index("1011")

    return x,signature, centroids, hb_search


# Load model
model, signature, my_centroid, my_hybrid_search = load_model()
lang_extract_model = FixedLangExtractProcessor()

# pd_pipeline = PPStructureV3(lang="th", device="gpu")



def get_items_from_html_table(html_string):
    """
    Parse HTML table and extract tbody content as 2D list.
    Each row becomes a list, and each <td> element becomes one item.
    
    Args:
        html_string: HTML string containing the table
        
    Returns:
        List of lists containing the extracted table data
    """
    # Parse the HTML
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Find the tbody
    tbody = soup.find('tbody')
    
    if not tbody:
        return []
    
    # Extract all rows
    result = []
    for row in tbody.find_all('tr'):
        # Extract all cells in the row
        cells = row.find_all('td')
        
        # Get the inner HTML or text from each cell
        row_data = []
        for cell in cells:
            # Get inner HTML (preserves <br> tags, etc.)
            content = ''.join(str(item) for item in cell.contents)
            row_data.append(content.strip())
        
        result.append(row_data)
    
    return result





def criteria_check_for_report(parameters_value:list[float], check_method:str, threshold_value:float):
    """
    Check if the parameters meet the criteria for the report.
    """
    if check_method == "greater_than":
        # check if the sum of all parameters value is greater than the threshold value or not
        return sum(parameters_value) > threshold_value
    elif check_method == "less_than":
        return sum(parameters_value) < threshold_value
    elif check_method == "equal_to":
        return sum(parameters_value) == threshold_value
    elif check_method == "not_equal_to":
        return sum(parameters_value) != threshold_value
    
def is_match_centroids(img_rgb: np.ndarray) -> bool:
    img_rgb = crop_top_percent(img_rgb, 20)
    clip_model_name: str = "ViT-B-16"

    tau ={'list_delta': 0.88}

    OCR_CLIP_PRETRAINED="/data/modules_resouces/openclip_backend/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_model.safetensors"

    res =   get_classifier(my_centroid,input_tau=tau,
                           input_clip_model_name=clip_model_name,
                           input_clip_pretrained=OCR_CLIP_PRETRAINED).predict_from_rgb(img_rgb) 
    # print("res", res)
    if res["pred"] == "UNKNOWN":
        return False
    return True


def extract_html_table(input_text: str) -> str:
    """
    Extract the HTML table from the input text.
    The HTML table begin with <table> tag and end with </table> tag.
    Return the first substring begin with "<table>" and end with "</table>".
    """
    # Find the start position of <table>
    start_pos = input_text.find("<table>")
    
    # If <table> not found, return empty string
    if start_pos == -1:
        return ""
    
    # Find the end position of </table> after the start position
    end_pos = input_text.find("</table>", start_pos)
    
    # If </table> not found, return empty string
    if end_pos == -1:
        return ""
    
    # Extract and return the table including the closing tag
    # end_pos points to the start of "</table>", so we add len("</table>") to include it
    return input_text[start_pos:end_pos + len("</table>")]


def handle_check_for_oxides():
    oxide_values = []
    oxide_names = ["Si","Al","Fe","Ca"]
    g_cv_v = -1.0

    prop_list = st.session_state["json"]["analysis_parameters"]
    for prop in prop_list:
        for abbreviate_name in oxide_names:
            if abbreviate_name in prop["parameters"]:
                # Convert the result value to float before append to the list
                if ("O" in prop["parameters"]) or ("0" in prop["parameters"]):
                    r_text = prop["result"]
                    r_text = r_text.replace(" ","")
                    r_text = r_text.replace(">","")
                    r_text = r_text.replace("<","")
                    r_text = r_text.replace("≤","")
                    r_text = r_text.replace("≥","")
                    r_text = r_text.replace("=","")
                    oxide_values.append(float(r_text))
                    break
    
    for p in prop_list:
        lower_parameter_name = p["parameters"].lower()
        if "gross cv" in lower_parameter_name:
            gross_cv_value = p["result"]
            gross_cv_value = gross_cv_value.replace(" ","")
            gross_cv_value = gross_cv_value.replace(">","")
            gross_cv_value = gross_cv_value.replace("<","")
            gross_cv_value = gross_cv_value.replace("≥","")
            gross_cv_value = gross_cv_value.replace("≤","")
            gross_cv_value = gross_cv_value.replace("=","")
            g_cv_v = float(gross_cv_value)
            break

    if len(oxide_values) == len(oxide_names):
        criteria_result = criteria_check_for_report(oxide_values, "greater_than", 20)
        if criteria_result == True:
            result_dialog("The total sum of Oxide is more than 20. Passed")
        else:
            result_dialog("The total sum of Oxide is less than 20. Failed")
    else:
        # There is no oxides in the parameters so we need to check for the gross cv instead
        if g_cv_v >= 0:
            _result = criteria_check_for_report([g_cv_v], "greater_than", 2000)
            if _result == True:
                result_dialog("The gross cv is more than 2000. Passed")
            else:
                result_dialog("The gross cv is less than 2000. Failed")
        else:
            result_dialog("Cannot find the oxides and gross cv in this Analysis Report")




def delta_items_ui():
    with st.container(border=True):
        items_lst_of_dic = []
        for item in st.session_state["delta_item_array"]:
            items_lst_of_dic.append({
                "item_name": item[0],
                "quantity": item[1]
            })

        current_df = pd.DataFrame(items_lst_of_dic)
        newer_df = st.data_editor(current_df, num_rows="dynamic", use_container_width=True)

        st.session_state["delta_item_list_of_dict"] = newer_df.to_dict("records")



def ui_js(session="json_str"):
    with st.container(border=True):
        json_data = json.loads(st.session_state["json_str"]) if session == "json_str" else st.session_state["json"]
        print("The value is: ")
        print(session)
        print(json_data)
        js = {}
        for k, v in json_data.items():
            if isinstance(v, list):
                df = pd.DataFrame(v)
                st_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
                js[k] = st_df.to_dict("records")
            elif isinstance(v, dict):
                # Handle nested dict - use expander with nested inputs
                with st.expander(f"📁 {k}", expanded=True):
                    nested_dict = {}
                    for nested_k, nested_v in v.items():
                        if isinstance(nested_v, (str, int, float, type(None))):
                            nested_value = st.text_input(
                                nested_k, 
                                value=str(nested_v) if nested_v is not None else "", 
                                key=f"{k}_{nested_k}"
                            )
                            nested_dict[nested_k] = nested_value
                        elif isinstance(nested_v, list):
                            # Handle list inside dict
                            nested_df = pd.DataFrame(nested_v)
                            nested_st_df = st.data_editor(nested_df, num_rows="dynamic", use_container_width=True, key=f"{k}_{nested_k}_editor")
                            nested_dict[nested_k] = nested_st_df.to_dict("records")
                        else:
                            # For complex nested structures, display as JSON
                            st.json({nested_k: nested_v})
                            nested_dict[nested_k] = nested_v
                    js[k] = nested_dict
            else:
                text_value = st.text_input(k, value=v, key=k)
                js[k] = text_value
                


        
        # if "items" in js and isinstance(js["items"], list):
            # filtered_items = []
            # for item in js["items"]:
            #     if isinstance(item, dict):
            #         quantity = item.get("quantity", 0)
            #         try:
            #             if isinstance(quantity, str):
            #                 quantity = int(quantity.replace(",", ""))
            #             elif isinstance(quantity, (int, float)):
            #                 quantity = int(quantity)
            #             else:
            #                 quantity = 0
                        
            #             if quantity != 0:
            #                 filtered_items.append(item)
            #         except (ValueError, AttributeError):
                        # filtered_items.append(item)
            
            # js["items"] = filtered_items
        
        st.session_state["json"] = js

        # st.session_state["payment_term"] = js["payment_term"]

@st.dialog("Result")
def result_dialog(message):
    st.write(message)

@st.dialog("Approval Status")
def approval_success_dialog():
    st.success("Approval is success")
    if st.button("Close"):
        st.session_state["refresh_page"] = True
        st.rerun()

@st.dialog("Rejection Status")
def rejection_dialog():
    st.warning("Not approve the items")
    if st.button("Close"):
        st.session_state["refresh_page"] = True
        st.rerun()


def approve_items_into_db():
    """Approve items and store them into the database"""
    try:
        only_not_approved_items = st.session_state.get("only_not_approved_items", [])
        print(f"Only not approved items: {only_not_approved_items}")

        if len(only_not_approved_items) > 0:
            not_approved_metadata = [
                {
                    "doc_id": (k + random.randint(21, 1000)),
                    "category": "DELTA"
                }
                for k in range(len(only_not_approved_items))
            ]
            
            my_hybrid_search.process_dataset(texts=only_not_approved_items, tenant_id="1011", metadata=not_approved_metadata)

            # if approve_items_into_db():
            approval_success_dialog()
            # else:
        
            # return True
        # return True
    except Exception as e:
        print(f"Error in approve_items_into_db: {str(e)}")
        st.error("Failed to approve items")
        # return False


def handle_delta_items_from_the_invoice():
    if len(st.session_state["delta_item_list_of_dict"]) > 0:
        print("Has the items in the invoice")
        final_result_lst = []
        for itm in st.session_state["delta_item_list_of_dict"]:
            it_name = itm["item_name"]
            results = my_hybrid_search.advanced_search(query=it_name, tenant_id="1011")
            current_item_status = {}
            current_item_status["name"] = it_name
            if len(results) > 0:
                print(f"Found {len(results)} results for {it_name}")
                itm_found = results[0]
                current_item_status["status"] = f"Approved Item: Match item is {itm_found['name']}"
                current_item_status["doc_id"] = f"Doc ID: {str( itm_found['doc_id'] )}"
                # for result in results:
                    # print(f"Result: {result}")
            else:
                current_item_status["status"] = "Not Approved Item"
                current_item_status["doc_id"] = "NONE"
                # print(f"No results found for {it_name}")
            
            final_result_lst.append(current_item_status)

        # Render Streamlit UI to show the data in final_result_lst variable (it is the list of dict with key "name" and "status")
        # Please use Streamlit table to show the data in final_result_lst variable

        # Display the results in a table format
        if final_result_lst:
            st.subheader("Item Approval Status")

            # Convert the list of dictionaries to a DataFrame for better table display
            import pandas as pd
            df = pd.DataFrame(final_result_lst)
            # Apply color styling based on status column
            def color_rows(row):
                if row['status'] == 'Not Approved Item':
                    return ['background-color: #ffcccc'] * len(row)  # Red background
                else:
                    return ['background-color: #ccffcc'] * len(row)  # Green background
            
            styled_df = df.style.apply(color_rows, axis=1)

            # Display the table using Streamlit's dataframe method
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Item Name", width="medium"),
                    "status": st.column_config.TextColumn("Approval Status", width="large")
                }
            )

            # Count the number of items in final_result_lst variable that have the value of "status" key not equal to "Not Approved Item"
            number_of_approved_items = len([item for item in final_result_lst if item["status"] != "Not Approved Item"])
            only_not_approved_items = [item["name"] for item in final_result_lst if item["status"] == "Not Approved Item"]

            # Format it_name_that_not_match as HTML list (one item per row)
            if only_not_approved_items:
                it_name_that_not_match = "<br>".join([f"• {item}" for item in only_not_approved_items])
            else:
                it_name_that_not_match = ""
            
            # Get the doc_id from final_result_lst variable that have the value of "doc_id" key not equal to "NONE"
            doc_ids = [item["doc_id"] for item in final_result_lst if item["doc_id"] != "NONE"]
            # Format doc_ids_str as HTML list (one item per row)
            if doc_ids:
                doc_ids_str = "<br>".join([f"• {doc_id}" for doc_id in doc_ids])
            else:
                doc_ids_str = ""
            
            recommend_final = ""
            
            if number_of_approved_items == len(final_result_lst):
                recommend_final = f"""
                <div style="background-color: #9B7EBD; padding: 20px; border-radius: 5px; color: white;">
                    <strong>Ready for Fast-Track Approval (พร้อมสำหรับอนุมัติด่วน)</strong><br><br>
                    ทั้ง {number_of_approved_items} items ใน invoice นี้ ตรงกับ items จากเอกสารที่เคยอนุมัติแล้ว คำขอนี้มีสิทธิ์ได้รับการ pre-approval<br><br>
                    <strong>Reference Documents (เอกสารอ้างอิง):</strong><br>
                    {doc_ids_str}
                    <br><br>
                </div>
                """
                st.markdown(recommend_final, unsafe_allow_html=True)
            elif number_of_approved_items > 0 and number_of_approved_items < len(final_result_lst):

                recommend_final = f"""
                <div style="background-color: #9B7EBD; padding: 20px; border-radius: 5px; color: white;">
                    <strong>Partial Match Found - Review Required (พบบางส่วนตรงกัน - ต้องตรวจสอบ)</strong><br><br>
                    บาง items ใน invoice นี้ พบในเอกสารอนุมัติก่อนหน้า (ดูเอกสารอ้างอิง) อย่างไรก็ตาม items ต่อไปนี้เป็น items ใหม่ 
                    หรือยังไม่เคยตรวจสอบ (unverified):<br>
                    {it_name_that_not_match}<br><br>
                    กรุณาตรวจสอบ (review) items ใหม่เหล่านี้เทียบกับ approval guidelines (แนวทางการอนุมัติ) ด้วยตนเอง<br><br>
                    <strong>Reference Documents for Matched Items (เอกสารอ้างอิงสำหรับรายการที่ตรงกัน):</strong><br>
                    {doc_ids_str}
                    <br><br>
                </div>
                """
                st.markdown(recommend_final, unsafe_allow_html=True)

            else:
                recommend_final = """
                <div style="background-color: #9B7EBD; padding: 20px; border-radius: 5px; color: white;">
                    <strong>Full Manual Review Required (ต้องตรวจสอบด้วยตนเองทั้งหมด)</strong><br><br>
                    items ทั้งหมดใน invoice นี้เป็น items ใหม่ และไม่มีประวัติการอนุมัติ<br>
                    กรุณาตรวจสอบ line items ทั้งหมดและรายละเอียดเอกสารเทียบกับ guidelines (แนวทาง) ของบริษัทอย่างรอบคอบก่อนอนุมัติ <br><br>
                </div>
                """
                st.markdown(recommend_final, unsafe_allow_html=True)


            # st.info(f"Number of approved items: {number_of_approved_items}")
            # Get the not approved items from the final_result_lst variable as the list of name of the items
            if number_of_approved_items < len(final_result_lst):
                
                
                # Store only_not_approved_items in session state
                st.session_state["only_not_approved_items"] = only_not_approved_items

                # Add Approve and Reject buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.button("Reject", type="secondary", use_container_width=True, on_click=rejection_dialog)
                with col2:
                    st.button("Approve", type="primary", use_container_width=True, on_click=approve_items_into_db)
                        

        else:
            st.info("No items to display")

    # pass

def perform_chandra_ocr_online() -> list[tuple]:
    
    API_URL = "https://www.datalab.to/api/v1/marker"
    HEADERS = {"X-Api-Key": "hdv5UyGQj5hg2WUXa7B88NrysefSCKfuoVpnlRY13lE"}

    #
    # Schema and file configuration
    #
    schema_json = """{
    "type": "object",
    "title": "ExtractionSchema",
    "description": "Schema for structured data extraction",
    "properties": {
        "supplier_info": {
        "type": "object",
        "description": "Details about the company issuing the invoice.",
        "properties": {
            "name": {
            "type": "string",
            "description": "Full legal name of the supplier."
            },
            "address": {
            "type": "string",
            "description": "Full address of the supplier."
            },
            "tax_id": {
            "type": "string",
            "description": "Tax identification number of the supplier."
            },
            "phone": {
            "type": "string",
            "description": "Supplier's telephone number."
            },
            "fax": {
            "type": "string",
            "description": "Supplier's fax number."
            }
        }
        },
        "invoice_header": {
        "type": "object",
        "description": "Key identifying information for the invoice.",
        "properties": {
            "invoice_number": {
            "type": "string",
            "description": "The unique invoice number."
            },
            "invoice_date": {
            "type": "string",
            "description": "The date the invoice was issued (MM/DD/YYYY format)."
            },
            "aeo_number": {
            "type": "string",
            "description": "Authorized Economic Operator (AEO) number."
            },
            "country_of_origin": {
            "type": "string",
            "description": "The country where the goods originated."
            },
            "price_term": {
            "type": "string",
            "description": "The Incoterm or price term used (e.g., CIF, FOB)."
            }
        }
        },
        "recipient_info": {
        "type": "object",
        "description": "Details about the recipient (Invoice To/Ship To party).",
        "properties": {
            "customer_id": {
            "type": "string",
            "description": "The customer or recipient identification code."
            },
            "name": {
            "type": "string",
            "description": "Name of the recipient company."
            },
            "address": {
            "type": "string",
            "description": "Full address of the recipient."
            },
            "is_ship_to_same_as_invoice_to": {
            "type": "boolean",
            "description": "Indicates if the Ship To address is identical to the Invoice To address."
            }
        }
        },
        "shipping_details": {
        "type": "object",
        "description": "Information regarding the transportation and logistics of the shipment.",
        "properties": {
            "shipped_per": {
            "type": "string",
            "description": "Method of shipment (e.g., TRUCK, AIR)."
            },
            "ex_factory_date": {
            "type": "string",
            "description": "Date the goods left the factory (MM/DD/YYYY format)."
            },
            "sailing_date": {
            "type": "string",
            "description": "Estimated sailing or departure date (MM/DD/YYYY format)."
            },
            "gross_weight_kg": {
            "type": "number",
            "description": "Total gross weight of the shipment in kilograms."
            },
            "net_weight_kg": {
            "type": "number",
            "description": "Total net weight of the shipment in kilograms."
            }
        }
        },
        "line_items": {
        "type": "array",
        "description": "List of products or services included in the invoice.",
        "items": {
            "type": "object",
            "properties": {
            "po_number": {
                "type": "string",
                "description": "Purchase Order number associated with the line item."
            },
            "part_number": {
                "type": "string",
                "description": "Manufacturer or internal part number."
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the item."
            },
            "quantity": {
                "type": "string",
                "description": "Quantity of the item, including unit (e.g., '1 SET')."
            },
            "unit_price": {
                "type": "string",
                "description": "Price per unit, including currency."
            },
            "amount": {
                "type": "string",
                "description": "Total amount for the line item, including currency and any notes (e.g., F.O.C.)."
            }
            }
        }
        }
    },
    "required": [
        "supplier_info",
        "invoice_header",
        "recipient_info",
        "shipping_details",
        "line_items"
    ]
    }"""
    pdf_path = "document.pdf"

    #
    # Load file and submit request. You can also pass in file_url
    #
    with open(pdf_path, "rb") as f:
        files = {
            'file': ('document.pdf', f, 'application/pdf'),
            'page_schema': (None, schema_json)
        }

        # Submit request
        response = requests.post(API_URL, files=files, headers=HEADERS)

        if response.status_code != 200:
            print(f"Error submitting job: {response.status_code}")
            print(response.text)
            exit(1)

        data = response.json()
        check_url = data["request_check_url"]
        print(f"Job submitted successfully. Polling URL: {check_url}")

    #
    # Poll for completion using request_check_url
    #
    max_polls = 300
    for i in range(max_polls):
        time.sleep(2)
        poll_response = requests.get(check_url, headers=HEADERS)

        if poll_response.status_code != 200:
            print(f"Error polling status: {poll_response.status_code}")
            break

        poll_data = poll_response.json()
        status = poll_data.get("status")

        print(f"Poll {i+1}: Status = {status}")

        if status == "failed":
            print(f"\nExtraction failed: {poll_data.get('error', 'Unknown error')}")
            break

        if status == "complete":
            print("\nExtraction completed successfully!")
            #
            # Your extracted results are in a field called extraction_schema_json
            # The raw parsed document with detected blocks are in json
            # Your extracted_schema_json contains contains to block IDs in the json
            # response field
            #
            extraction_result = json.loads(poll_data.get('extraction_schema_json', '{}'))
            print("\nExtracted data:")
            print(json.dumps(extraction_result, indent=2))
            all_it_des = []
            all_it_qty = []

            l_itm_lst = extraction_result.get("line_items",[])
            for e_it in l_itm_lst:
                tmp_itm_description = e_it["description"]
                tmp_itm_quantity    = e_it["quantity"]

                all_it_des.append(tmp_itm_description.strip())
                all_it_qty.append(tmp_itm_quantity.strip())


            return  list(zip(all_it_des, all_it_qty))

            # break
    else:
        print("\nTimeout reached after 300 polls. Job may still be processing.")
    # pass


async def ocr_processing_page():
    """Main OCR processing page function"""
    # st.header("📄 OCR Processing")
    
    # Initialize session state variables
    if "new_upload" not in st.session_state:
        st.session_state["new_upload"] = True

    if "delta_item_array" not in st.session_state:
        st.session_state["delta_item_array"] = []

    if "delta_item_list_of_dict" not in st.session_state:
        st.session_state["delta_item_list_of_dict"] = []

    if "markdown" not in st.session_state:
        st.session_state["markdown"] = ""
        
    if "json_str" not in st.session_state:
        st.session_state["json_str"] = ""
        
    if "payment_term" not in st.session_state:
        st.session_state["payment_term"] = ""
        
    if "json" not in st.session_state:
        st.session_state["json"] = {}

    if "json_button_disabled" not in st.session_state:
        st.session_state["json_button_disabled"] = False

    if "invoice_check_button_disabled" not in st.session_state:
        st.session_state["invoice_check_button_disabled"] = True

    if "resultss" not in st.session_state:
        st.session_state["resultss"] = 0

    if "total_images" not in st.session_state:
        st.session_state["total_images"] = 0

    if "processing_time" not in st.session_state:
        st.session_state["processing_time"] = 0

    if "only_not_approved_items" not in st.session_state:
        st.session_state["only_not_approved_items"] = []

    if "refresh_page" not in st.session_state:
        st.session_state["refresh_page"] = False

    # Check if page needs to be refreshed
    if st.session_state.get("refresh_page", False):
        # Reset all session state variables to initial state
        st.session_state["new_upload"] = True
        st.session_state["delta_item_array"] = []
        st.session_state["delta_item_list_of_dict"] = []
        st.session_state["markdown"] = ""
        st.session_state["json_str"] = ""
        st.session_state["payment_term"] = ""
        st.session_state["json"] = {}
        st.session_state["json_button_disabled"] = False
        st.session_state["invoice_check_button_disabled"] = True
        st.session_state["resultss"] = 0
        st.session_state["total_images"] = 0
        st.session_state["processing_time"] = 0
        st.session_state["only_not_approved_items"] = []
        st.session_state["refresh_page"] = False
        st.rerun()


    # === Upload PDF ===
    with st.sidebar:
        document_type = st.radio(
            "Choose the document type",
            options=["Invoice", "Packing List", "Passport", "Certificate", "Stock Shareholder BOJ5", "DBD"],
            index=0  # Default to "Invoice"
        )

        selected_ocr_model = "high_performance_ocr"
        # st.radio(
        #     "Select OCR Model",
        #     options=["text_ocr", "high_performance_ocr", "legacy_ocr"],
        #     index=0,
        #     horizontal=True
        # )
        st.session_state["ocr_model"] = selected_ocr_model

        uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
        if uploaded_file:
            # doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            images = []
            texts = []
            is_invoice_or_quotation = False
            image_inp_path = []
            pdf_input_paths = []

            # Check if uploaded file is PDF or image
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == "pdf":
                # Handle PDF file
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                doc.save("document.pdf")

                for i, page in enumerate(doc):
                    # Page as image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(f"{uploaded_file.name.replace('.pdf', '')}_{i}.png")

                    image_inp_path.append(f"{uploaded_file.name.replace('.pdf', '')}_{i}.png")
                    images.append(img)

                    # Save each page as a separate PDF and append to pdf_input_paths
                    base_no_ext = uploaded_file.name.rsplit('.', 1)[0]
                    page_pdf_path = f"{base_no_ext}_{i}.pdf"
                    single_doc = fitz.open()  # create an empty PDF
                    single_doc.insert_pdf(doc, from_page=i, to_page=i)
                    single_doc.save(page_pdf_path)
                    single_doc.close()
                    pdf_input_paths.append(page_pdf_path)
                    
            elif file_extension in ["png", "jpg", "jpeg"]:
                # Handle image file
                img = Image.open(uploaded_file)
                
                # Convert to PNG format and save
                base_name = uploaded_file.name.rsplit('.', 1)[0]
                png_filename = f"{base_name}.png"
                
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode != 'RGB' and img.mode != 'RGBA':
                    img = img.convert('RGB')
                
                img.save(png_filename, 'PNG')
                image_inp_path.append(png_filename)
                images.append(img)

                # enhancer = ImageEnhance.Sharpness(img)
                # img_sharp = enhancer.enhance(2)
                # contrast = ImageEnhance.Contrast(img_sharp)
                # img_final = contrast.enhance(2)
                
            st.markdown("### 📷 Preview")

            if image_inp_path:
                st.text_input(label="Filename :",value=image_inp_path[0])
                        
            for img in images:
                # results = signature.predict(np.array(img), conf=0.5)
                # print(results)
                # st.image(results[0].plot())

                # resultss=[]
                # for result in results:
                #     for box in result.boxes:
                #         cls_id = int(box.cls[0]) 
                #         class_name = signature.names[cls_id]
                #         conf = float(box.conf[0])
                #         resultss.append({"class_name":class_name,"score":conf,"class_id":cls_id})

                # st.session_state["resultss"]=len(resultss)
                st.image(img)

    if uploaded_file and st.session_state["new_upload"]:
        # OCR model selection (default model_a)
        
        with st.status("Checking if this document is invoice or not", expanded=True) as status_check_invoice:
            doc_category = model.check_if_it_invoice(image_inp_path[0])
            # doc_category_md = st.empty()

            if doc_category == "Invoice" or doc_category == "Quotation":
                is_invoice_or_quotation = True


            if doc_category == "Invoice":
                status_check_invoice.update(
                    label="This document is an invoice", state="complete", expanded=False
                )
            else:
                status_check_invoice.update(
                    label="This document is not an invoice", state="complete", expanded=False
                )

                # doc_category_md.markdown("This document is an invoice")
                # st.session_state["invoice_check_button_disabled"] = False
            # else:
                # doc_category_md.markdown(f"This document is not an invoice. It is {doc_category}")
                # st.session_state["invoice_check_button_disabled"] = True


            

        
        with st.status("Checking if this invoice is from DELTA or not", expanded=True) as status_check_for_delta:

            img_rgb = load_image_as_rgb_array(image_inp_path[0])
            is_inv_delta = is_match_centroids(img_rgb)
            if is_inv_delta:
                status_check_for_delta.update(
                    label="✅ This invoice is from DELTA", state="complete", expanded=False
                )
            else:
                status_check_for_delta.update(
                    label="❌ This invoice is not from DELTA", state="error", expanded=False
                )
            
            # status_check_for_delta.update(
                # label="Successfully check DELTA invoice", state="complete", expanded=False
            # )

        # Check if is_inv_delta is True or False.
        # If it's True, then show the GREEN text ("This invoice is DELTA. Continue to extract the text from invoice") on the screen below the "Checking if this invoice is from DELTA or not" status and continue to extract the text.
        # If it's False, then show the RED text ("Please upload DELTA invoice only") below "Checking if this invoice is from DELTA or not" status and do not continue to extract the text.
        
        if is_inv_delta:
            st.success("✅ This invoice is DELTA. Continue to extract the text from invoice")
        else:
            st.info("Process invoice parsing without pre-approval process")
            # st.session_state["new_upload"] = False
            # return

        with st.status("Extracting text", expanded=True) as status:
            center_md = st.empty()
            center_stream = ""
            index = 0
            sig_stream = ""

            # total_sig = count_signatures(sig_stream)
            # st.session_state["resultss"] = total_sig

            # all_images_count = count_imgs(sig_stream)
            # st.session_state["total_images"] = all_images_count

            # for img in images:
                # index += 1
                # center_stream += "##### Page " + str(index) + "\n"
                # for i in model.predict(img):
                    # center_stream += i
                    # center_md.markdown(center_stream[1:-1].replace('"', ''))
                # center_stream += "\n***\n"
            if len(image_inp_path) > 0:
                start_time = time.time()
                if True:
                    if selected_ocr_model == "text_ocr":
                        # output_path = Path("./output_markdown")

                        # for img_nn in image_inp_path:
                        #     print(f"Processing image name: {img_nn}")

                        #     output = pd_pipeline.predict(input=img_nn)

                        #     markdown_list = []
                        #     # markdown_images = []

                        #     for res in output:
                        #         md_info = res.markdown
                        #         markdown_list.append(md_info)
                        #         # markdown_images.append(md_info.get("markdown_images", {}))

                        #     markdown_texts = pd_pipeline.concatenate_markdown_pages(markdown_list)

                            center_stream += ""
                            center_stream += "\n"

                    elif selected_ocr_model == "high_performance_ocr":
                        for img_nn in image_inp_path:
                            if (document_type == "Invoice") or (document_type == "Packing List"):
                                tmp_center_stream = "Error" # await model.typhoon_runpod_predict(img_nn, "structure", 1)
                                # if tmp_center_stream contain the word "Error"
                                if "Error" in tmp_center_stream:
                                    if is_inv_delta:
                                        tmp_center_stream = await model.typhoon_runpod_predict(img_nn, "structure", 1)
                                    else:
                                        tmp_center_stream = await model.typhoon_runpod_predict(img_nn, "structure", 1) 
                                        # tmp_center_stream = await model.run_hunyuan_predict(img_nn)
                                # else:
                                    # tmp_center_stream = await model.typhoon_runpod_predict(img_nn, "structure", 1)

                                # flag_repetition_result = flag_repetitive_output(tmp_center_stream)

                                # if flag_repetition_result["flagged"]:
                                    # print("Repetitive result detected. Switching to alternative OCR model.")
                                    # np_array_image_date = preprocess_for_improve_ocr(image_path=img_nn, output_path="improved_ocr_quality.png")

                                    # retry doing the typhoon OCR with the improved image
                                    # tmp_center_stream = await model.typhoon_runpod_predict("improved_ocr_quality.png", "structure", 1)


                                center_stream += tmp_center_stream

                            elif document_type == "Passport":
                                center_stream = await model.parsing_mrz_passport(img_nn)
                                # center_stream += tmp_center_stream

                            elif document_type == "Stock Shareholder BOJ5":
                                # tmp_center_stream = await model.docling_with_surya("document.pdf")
                                tmp_center_stream = await model.run_hunyuan_predict(img_nn)

                                center_stream += tmp_center_stream

                            elif document_type == "DBD":
                                tmp_center_stream = await model.typhoon_runpod_predict(img_nn, "structure", 1)
                                # if tmp_center_stream contain the word "Error"

                                center_stream += tmp_center_stream

                                if "ออกให้" in tmp_center_stream:
                                    break
                                    # tmp_center_stream = await model.dotsocr_runpod_predict(img_nn)


                            else:
                                tmp_center_stream = await model.typhoon_runpod_predict(img_nn, "structure", 1)
                                # if tmp_center_stream contain the word "Error"
                                if "Error" in tmp_center_stream:
                                    tmp_center_stream = await model.dotsocr_runpod_predict(img_nn)

                                center_stream += tmp_center_stream
                            
                            # center_stream += "\n"
                    
                    elif selected_ocr_model == "legacy_ocr":
                        for img_nn in image_inp_path:
                            center_stream += await model.typhoon_runpod_predict(img_nn, "structure", 1)
                            center_stream += "\n"


                    # if selected_ocr_model == "model_a":
                    #     for img_nn in image_inp_path:
                    #         center_stream += await model.typhoon_runpod_predict(img_nn, "structure", 1)
                    #         center_stream += "\n\n"
                    # elif selected_ocr_model == "model_b":
                    #     for img_nn in image_inp_path:
                    #         center_stream += await model.dotsocr_runpod_predict(img_nn)
                    #         center_stream += "\n\n"
                    # elif selected_ocr_model == "model_c":
                    #     for img_nn in image_inp_path:
                    #         center_stream += await model.dotsocr_runpod_predict(img_nn)
                    #         center_stream += "\n\n"
                    # elif selected_ocr_model == "TheBest":
                    #     for img_nn in image_inp_path:
                    #         center_stream += await model.dotsocr_runpod_predict(img_nn)
                    #         center_stream += "\n\n"
                            
                else:
                    current_page = 1
                    request_id = str(uuid.uuid4())[:8]
                    num_pages = len(image_inp_path)

                    try:
                        for markdown_content in convert_to_markdown_stream(
                            image_inp_path,
                            model_name="nanonets/Nanonets-OCR-s",
                            max_img_size=2048,
                            concurrency_limit=1,
                            max_gen_tokens=10000,
                        ):
                            # Add progress indicator at the top for multi-page documents
                            if num_pages > 1:
                                progress_header = f"(Page {min(current_page, num_pages)} of {num_pages})\n\n"
                                center_stream = progress_header + process_tags(markdown_content)
                            else:
                                center_stream = process_tags(markdown_content)

                            # Estimate current page based on content length (rough approximation)
                            if "---" in markdown_content:
                                current_page = markdown_content.count("---") + 1

                            # Reduced delay for better concurrent performance
                            # center_md.markdown(center_stream)

                            time.sleep(0.01)

                    except Exception as e:
                        error_message = f"❌ **Error processing request {request_id}**: {str(e)}\n\nPlease try again or contact support if the issue persists."
                        # yield error_message
                        print(error_message)

                    # pass
                    # center_stream = model.typhoon_runpod_predict(image_inp_path[0], "structure", 1)
                end_time = time.time()
                processing_time = end_time - start_time
                st.session_state["processing_time"] = processing_time
            
            if document_type == "Stock Shareholder BOJ5":
                st.text_area("Center Stream", center_stream, height=500)
            elif document_type == "Passport":
                st.json(center_stream)
            else:
                center_md.markdown(center_stream)
                
            st.session_state["markdown"] = center_stream

            if is_inv_delta:
                # Extract chunk based on P.O. NO. markers
                items_is_inside_html = False
                po_start = center_stream.find('| P.O. NO.')
                if po_start != -1:
                    # Found '| P.O. NO.'
                    vvvv_end = center_stream.find('VVVV |\\n', po_start)
                    if vvvv_end != -1:
                        # Found 'VVVV |\n' - extract from po_start to end of 'VVVV |\n'
                        po_no_chunk = center_stream[po_start:vvvv_end + len('VVVV |')]
                    else:
                        # 'VVVV |\n' not found - extract from po_start until '"' is found
                        quote_pos = center_stream.find('"', po_start)
                        if quote_pos != -1:
                            po_no_chunk = center_stream[po_start:quote_pos]
                        else:
                            # No quote found either, just take from po_start to end
                            po_no_chunk = center_stream[po_start:]
                else:
                    # '| P.O. NO.' not found - use whole content
                    # po_no_chunk = center_stream
                    # Check if po_no_chunk contain the HTML table tag or not (The table tag is <table>...</table>)
                    if "<table>" in center_stream and "</table>" in center_stream:
                        # Extract the HTML table from po_no_chunk
                        po_no_chunk = extract_html_table(center_stream)
                        print("HTML table:", po_no_chunk)
                        if len(po_no_chunk) > 5:
                            items_is_inside_html = True
                    else:
                        # No HTML table found - use whole content
                        po_no_chunk = center_stream

            
            
                if items_is_inside_html == False:
                    # po_no_chunk = po_no_chunk.replace('\n', '_')

                    # replace the new line character in po_no_chunk with the underscore character
                    po_no_chunk = po_no_chunk.replace('\\n', '_')

                    print("The input string is:", po_no_chunk)

                    # Step 1: Split by newline
                    lines = po_no_chunk.split('_')
                    print("Lines:", lines)
                    print("Length of lines:", len(lines))
                    # Step 2: Filter out lines containing "QTY", "---", or "VVVVV"
                    filtered_lines = [
                        line for line in lines 
                        if "QTY" not in line and "---" not in line and "VVVVV" not in line
                    ]
                    

                    print("Filtered lines:", filtered_lines)
                    # Lists to store extracted data
                    item_name_lst = []
                    qty_item_lst = []
                    
                    # Step 3: Process each remaining line
                    for line in filtered_lines:
                        # Step 3.1: Split by pipe
                        words = line.split('|')
                        print("Words:", words)
                        # Ensure we have at least 3 elements (indices 0, 1, 2)
                        if len(words) < 4:
                            # check if the words index 0 or 1 is not empty (contain the non-blank string)
                            if len(words) == 1:
                                if len(words[0].strip()) > 1 and not words[0].strip().startswith('('):
                                    if len(qty_item_lst) > 0:
                                        item_name_lst.append(words[0].strip())

                            elif len(words) == 2:
                                if len(words[1].strip()) > 1 and not words[1].strip().startswith('('):
                                    if len(qty_item_lst) > 0:
                                        item_name_lst.append(words[1].strip())
                                elif len(words[0].strip()) > 1 and not words[0].strip().startswith('('):
                                    if len(qty_item_lst) > 0:
                                        item_name_lst.append(words[0].strip())

                            continue
                            
                        # Step 3.2: Check third element (index 2)
                        third_element = words[3].strip()
                        print(third_element)
                        
                        if third_element == '' or third_element == ' ':
                            # Third element is blank or empty
                            first_element = words[1].strip()
                            
                            # Check if first element doesn't begin with '('
                            if first_element and not first_element.startswith('('):
                                if len(qty_item_lst) > 0:
                                    item_name_lst.append(first_element)
                        else:
                            # Third element has content - it's a quantity
                            qty_item_lst.append(third_element)
                            # If we also found that the first element is exists, then we need to append the first element to the item_name_lst
                            tmp_first_ele = words[1].strip()
                            if tmp_first_ele and len(tmp_first_ele) > 1 and not tmp_first_ele.startswith('('):
                                item_name_lst.append(tmp_first_ele)
                    
                    # Step 4: Zip the lists together
                    # Ensure both lists have the same length by padding qty_item_lst if needed
                    while len(qty_item_lst) < len(item_name_lst):
                        qty_item_lst.append("")

                    print("Qty item lst:", qty_item_lst)
                    print("Item name lst:", item_name_lst)
                    # Truncate if qty_item_lst is longer
                    qty_item_lst = qty_item_lst[:len(item_name_lst)]
                    
                    result = list(zip(item_name_lst, qty_item_lst))
                
                else:
                    tmp_result_lst = get_items_from_html_table(po_no_chunk)

                    item_n_lst = []
                    qty_lst = []

                    for each_row in tmp_result_lst:
                        if len(each_row) < 2:
                            first_ele = each_row[0].strip()
                            if first_ele.startswith('$') or first_ele.startswith("VVVV"):
                                break
                            else:
                                if len(qty_lst) > 0:
                                    if len(first_ele) > 0 and not first_ele.startswith('('):
                                        item_n_lst.append(first_ele)

                            continue

                        second_ele = each_row[1].strip()
                        f_ele = each_row[0].strip()
                        if f_ele.startswith('$') or f_ele.startswith("VVVV"):
                            break

                        if second_ele.startswith("VVVV"):
                            break


                        if len(second_ele) > 0:
                            qty_lst.append(second_ele)

                            # We still need to check for the first element to see if it is empty string or has the value
                            
                            if len(f_ele) > 0 and not f_ele.startswith('$') and not f_ele.startswith('('):
                                item_n_lst.append(f_ele)



                    while len(qty_lst) < len(item_n_lst):
                        qty_lst.append("")

                    qty_lst = qty_lst[:len(item_n_lst)]

                    result = list(zip(item_n_lst, qty_lst))

                # Add code to filter the items in result (list of tuple) variable.
                # The example of data in result list => [('AC Motor','1SET'), ('DC Motor','2SET'), ('Wire','5SET')]
                # Filter out the items in result list which have the second element of tuple equal to empty string (the string which length = 0)
                if len(result) == 0:
                    result = perform_chandra_ocr_online()


                result_filter_out = [item for item in result if (len(item[1]) > 0 and bool(re.search(r'\d', item[1])) == True)]

                st.session_state["delta_item_array"] = result_filter_out
                
                # pre_process_invoice_string(po_no_chunk)
                

                # Example usage with the robust function
                # result_count = extract_observation_and_check_robust(center_stream)
                # if result_count["contains_target_words"] == True:
                    # st.session_state["total_images"] = 1
                # else:
                    # st.session_state["total_images"] = 0 


                #[1:-1].replace('"', '')
            
            status.update(
                label="Successfully extracted text", state="complete", expanded=False
            )

        
                

        er={"error_bool":False,"error":""}

        with st.status("Extracting Invoice Data from OCR and Checking for LOGO and Signature", expanded=True) as status_json:
            right_md = st.empty()
            right_stream = ""

            try:

                start_time_2 = time.time()

                meta_certificate = {}

                # if is_invoice_or_quotation == True:
                if document_type == "Invoice":
                    if is_inv_delta:
                        right_stream = await model.structured_output(center_stream, DeltaInvoice)
                    else:
                        right_stream = await model.structured_output(center_stream, Invoice)

                    right_md.markdown(right_stream)
                elif document_type == "Passport":
                    right_stream = await model.structured_output(center_stream, PassPortData)
                    right_md.markdown(right_stream)
                    
                elif document_type == "Stock Shareholder BOJ5":
                    right_stream = await model.structured_output(center_stream, CompanyStock)
                    right_md.markdown(right_stream)
                
                elif document_type == "DBD":
                    right_stream = await model.structured_output(center_stream, Certificate_DBD)
                    right_md.markdown(right_stream)
                    
                elif document_type == "Certificate":
                    print("Processing certificate")
                    meta_certificate = lang_extract_model.extract_metadata(center_stream)
                    right_md.markdown("###See the in result below tab")
                else:
                    right_stream = await model.structured_output(center_stream, PackingList)
                    right_md.markdown(right_stream)


                end_time_2 = time.time()
                processing_time_2 = end_time_2 - start_time_2
                st.session_state["json_extract_time"] = processing_time_2
                # right_stream = model.structured_output(center_stream, InvoiceOutput)
                # right_md.markdown(right_stream)
                st.session_state["over_all_processing_time"] = st.session_state["processing_time"] + processing_time_2

                if document_type != "Certificate":
                    st.session_state["json_str"] = re.search(r'```json\s*(\{.*?\})\s*```', right_stream, re.DOTALL).group(1)
                else:
                    # Convert meta_certificate from Dict into Json String
                    st.session_state["json_str"] = json.dumps(meta_certificate)
                    print(st.session_state["json_str"])


                # st.session_state["json_str"] = right_stream

                er["error_bool"] = False

            except Exception as e:
                er["error_bool"] = True
                er["error"] = e
                right_md.error(f"{er['error']}", icon="❌")

            if not er["error_bool"]:
                status_json.update(
                    label="Successfully extracted Json", state="complete", expanded=False
                )
            else:
                status_json.update(
                    label="Error occurred", state="error", expanded=True
                )
            
        # st.text_input(label="signature",value=st.session_state["resultss"])

        if "processing_time" in st.session_state and st.session_state["processing_time"] > 0:
            st.metric(label="OCR Processing Time", value=f"{st.session_state['processing_time']:.2f} seconds")
            st.metric(label="JSON Extract Time", value=f"{st.session_state['json_extract_time']:.2f} seconds")
            st.metric(label="Over All Processing Time", value=f"{st.session_state['over_all_processing_time']:.2f} seconds")


        # if is_invoice_or_quotation == True:
        if not er["error_bool"]:
            ui_js("json_str")

            # Write OCR processing time values to CSV file
            try:
                csv_output_dir = Path("/data/result_ocr")
                csv_output_dir.mkdir(parents=True, exist_ok=True)

                csv_filename = f"{st.session_state.get('selected_pdf_name', 'output')}.csv"
                csv_filepath = csv_output_dir / csv_filename

                csv_data = {
                    'processing_time': [st.session_state.get('processing_time', 0)],
                    'json_extract_time': [st.session_state.get('json_extract_time', 0)],
                    'over_all_processing_time': [st.session_state.get('over_all_processing_time', 0)],
                    'json_str': [st.session_state.get('json_str', '')]
                }

                df = pd.DataFrame(csv_data)
                df.to_csv(csv_filepath, index=False, encoding='utf-8')

                st.toast("SUCCEED", icon="✅")
            except Exception as e:
                st.toast("ERROR", icon="❌")

            if is_inv_delta:
                if len(st.session_state["delta_item_array"]) > 0:
                    delta_items_ui()
                else:
                    st.warning("No items found in the delta invoice")

                # handle_json_button_click()
                # st.button("Send OCR data to database", use_container_width=True, on_click=handle_json_button_click)
                if len(st.session_state["delta_item_array"]) > 0:
                    st.button("Invoice Check", use_container_width=True, on_click=handle_delta_items_from_the_invoice, disabled=False)
        else:
            st.error("❌ Error occurred while extracting JSON")
            # st.button("Invoice Check", use_container_width=True, on_click=handle_delta_items_from_the_invoice, disabled=True)    
        # else:
            # if not er["error_bool"]:
                # ui_js("json_str")
                # st.button("Check for Oxide", use_container_width=True, on_click=handle_check_for_oxides)

            
        st.session_state["new_upload"] = False
            
    elif st.session_state.get("ocr_submitted", False) and not st.session_state["new_upload"]:
        
        # with st.status("Extracting text", expanded=True) as status:
            # st.markdown(st.session_state["markdown"])
            # status.update(
                # label="Successfully extracted text", state="complete", expanded=False
            # )

        # if "processing_time" in st.session_state and st.session_state["processing_time"] > 0:
            # st.metric(label="OCR Processing Time", value=f"{st.session_state['processing_time']:.2f} seconds")

        # with st.status("Extracting Json", expanded=True) as status_json:
            # st.markdown(st.session_state["json_str"])
            # status_json.update(
                # label="Successfully extracted Json", state="complete", expanded=False
            # )
            

        # st.text_input(label="signature",value=st.session_state["resultss"])
        ui_js("json")
        
        # st.button("Send OCR data to database", use_container_width=True, on_click=handle_json_button_click)
        # st.button("Invoice Check", use_container_width=True, on_click=handle_invoice_check_click, disabled=st.session_state["invoice_check_button_disabled"])

    else:
        st.session_state["new_upload"] = True
        st.session_state["markdown"] = ""
        st.session_state["delta_item_array"] = []

        st.session_state["delta_item_list_of_dict"] = []
        st.session_state["json_str"] = ""
        st.session_state["json"] = {}
        st.session_state["payment_term"] = ""
        st.session_state["processing_time"] = 0
