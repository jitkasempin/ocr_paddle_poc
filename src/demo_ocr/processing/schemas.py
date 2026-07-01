"""Pydantic extraction schemas for the OCR CLI.

Extracted verbatim from ``ocr_page.py`` (which cannot be imported outside the
GPU/Streamlit container because it imports streamlit, ultralytics, supabase,
etc. at module top). These depend only on the light-weight ``schema_helper``.
"""
from __future__ import annotations

from decimal import Decimal
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .schema_helper import (
    extract_code,
    extract_only_branch_code_number,
    extract_po_decimal,
    parse_decimal_like,
    parse_thai_date,
)


# --- Passport -------------------------------------------------------------
class PassPortData(BaseModel):
    passport_number: str = Field(..., description="Unique passport identifier")
    issuing_country: str = Field(..., description="ISO 3166-1 alpha-3 country code")
    surname: str = Field(..., min_length=1, max_length=100, description="Last name/family name")
    given_names: str = Field(..., min_length=1, max_length=100, description="First and middle names")
    date_of_expiry: str = Field(..., description="Passport expiry date")


# --- Packing list ---------------------------------------------------------
class PackingListItem(BaseModel):
    item_description: str = Field("", description="description ของสินค้า หรือ รายการสินค้า หรือ รายการ หรือ รายละเอียด")
    quantity: float = Field(0, description="จำนวน หรือ Quantity หรือ ปริมาณ")
    unit: str = Field("", description="ข้อความที่บ่งบอกถึงรูปแบบการขายสินค้า มักจะอยู่คู่กับปริมาณหรือ quantity หรือข้อความที่อยู่ใน column unit หรือ column หน่วย")


class PackingList(BaseModel):
    table: list[PackingListItem] = Field(description="รายการสินค้ารวมถึงปริมาณที่อยู่ใน Packling List")


# --- Invoice --------------------------------------------------------------
class Document(BaseModel):
    invoice_number: str = Field(
        default="", description="เลขที่ หรือ เลขที่ใบกำกับ หรือ เลขที่ใบกำกับภาษี หรือ invoice number หรือ invoice no หรือ inv no")
    po_number: str = Field(
        default="", description=(
            "'ใบสั่งซื้อ', 'P/O NO', 'P.O. No', 'เลขที่ใบสั่งซื้อ', 'ใบสั่งซื้อเลขที่', 'PO NO', 'Purchase order number', 'เลขที่ PO'"
            "ต้องไม่ใช่ค่าของ 'เลขที่ใบสั่งขาย' หรือ 'S.O. No' หรือ 'เลขที่ใบส่งของ'"
        )
    )
    date: str = Field(
        default="", description="date หรือ date: หรือ วันที่ออกใบแจ้งหนี้ หรือวันที่ ที่ได้เขียนไว้บน Invoice")
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
    name: str = Field(default="", description="Company name at the top of the invoice")
    tax_id: str = Field(default="", description="เลขประจำตัวผู้เสียภาษี หรือ Tax ID")
    address: str = Field(default="", description="ที่อยู่ของบริษัท หรือที่ตั้งของสำนักงานใหญ่บริษัท")
    contact: Optional[str] = Field(default=None, description="ข้อมูลการติดต่อ เช่น เบอร์โทรศัพท์ หรือ อีเมล หรือ Email หรือ phone number")
    branch_name: Optional[str] = Field(default=None, description="ชื่อสาขา หรือ branch name")
    branch_code: Optional[str] = Field(default=None, description="สาขาที่ หรือ สาขา หรือ branch number")


class CustomerCompany(BaseModel):
    name: str = Field(default="", description="ชื่อลูกค้า หรือ นามลูกค้า หรือ ลูกค้า หรือ ขายให้ หรือ ผู้ซื้อ หรือ CUSTOMER หรือ SOLD TO หรือ ชื่อ หรือ BILL TO หรือ SHIP TO")
    tax_id: str = Field(default="", description="เลขประจำตัวผู้เสียภาษี หรือ Tax ID ของลูกค้า")
    address: str = Field(default="", description="ที่อยู่ลูกค้า หรือ ที่อยู่ หรือ Address")
    contact: Optional[str] = Field(default=None, description="ข้อมูลการติดต่อ เช่น เบอร์โทรศัพท์ หรือ อีเมล หรือ Email หรือ phone number")
    branch_name: Optional[str] = Field(default=None, description="ชื่อสาขา / branch name")
    branch_code: Optional[str] = Field(default=None, description="'สาขาที่', 'สาขา', 'branch no', 'branch code', 'branch number', 'branch #', 'branch :'")

    @field_validator("branch_code", mode="before")
    @classmethod
    def _heal_branch_code(cls, v):
        if v is not None:
            return extract_only_branch_code_number(v)
        return v


class Item(BaseModel):
    code: str = Field(default="", description="'ลำดับ', 'item', 'item code', 'item no', 'item number', 'item #', 'code'")
    description: str = Field(default="", description="description ของสินค้า หรือ รายการสินค้า หรือ รายการ หรือ รายละเอียด")
    unit_price: Decimal = Field(default=Decimal("0.00"), description="ราคาต่อหน่วย หรือ หน่วยละ หรือ Unit Price")
    uom: str = Field(default="", description="unit of material หรือ column unit หรือ column หน่วย")
    quantity: int = Field(default=0, description="จำนวน หรือ Quantity หรือ ปริมาณ")
    discount: Decimal = Field(default=Decimal("0.00"), description="ส่วนลด หรือ Discount amount")
    amount: Decimal = Field(default=Decimal("0.00"), description="ราคารวม หรือ amount หรือ Total amount for this item")
    note: str = Field(default="-", description="หมายเหตุ หรือ Additional notes")

    @field_validator("unit_price", "discount", "amount", mode="before")
    @classmethod
    def _heal_item_decimals(cls, v):
        return parse_decimal_like(v)

    @model_validator(mode="after")
    def extract_code_from_description(self):
        if self.description:
            tmp_return_code = extract_code(self.description)
            self.code = tmp_return_code if tmp_return_code is not None else ""
        return self


class Summary(BaseModel):
    subtotal: Decimal = Field(default=Decimal("0.00"), description="Subtotal before tax")
    discount_total: Decimal = Field(default=Decimal("0.00"), description="Total discount")
    vat_rate: int = Field(default=7, description="VAT rate percentage")
    vat: Decimal = Field(default=Decimal("0.00"), description="VAT amount")
    total_amount: Decimal = Field(default=Decimal("0.00"), description="Total amount including VAT")
    currency: Literal["BAHT", "THB", "บาท", "US DOLLAR", "USD", "SING.DOLLAR", "SGD", "POUND STERING", "EURO", "GBP", "RUPEE", "INR", "SWISS FRANC", "CHF", "CHINESE YUAN", "CNY", "YEN", "JPY", "CANADIAN DOLLAR", "CAD"] = Field(
        default="THB", description="สกุลเงินที่ใช้ใน Invoice")

    @field_validator("subtotal", "discount_total", "vat", "total_amount", mode="before")
    @classmethod
    def _heal_summary_decimals(cls, v):
        return parse_decimal_like(v)


class Invoice(BaseModel):
    document: Document = Field(default_factory=Document, description="General invoice information")
    seller: SellerCompany = Field(default_factory=SellerCompany, description="The company that issues the invoice")
    customer: CustomerCompany = Field(default_factory=CustomerCompany, description="The company that buys the products")
    items: List[Item] = Field(default_factory=list, description="List ของรายการสินค้าที่มี Quantity มากกว่า 0")
    summary: Summary = Field(default_factory=Summary, description="Invoice summary")


# --- Stock shareholder (BOJ5) --------------------------------------------
class StakeHolders(BaseModel):
    stakeholder_name: str = Field(default="", description="ชื่อผู้ถือหุ้น หรือ รายชื่อผู้ถือหุ้น")
    stock_amount: Decimal = Field(default=Decimal("0.00"), description="จำนวนหุ้นที่ถือ")
    stakeholder_nationality: str = Field(default="", description="สัญชาติ")


class CompanyStock(BaseModel):
    company_name: str = Field(default="", description="ชื่อบริษัทจำกัด")
    company_stakeholders: List[StakeHolders] = Field(default_factory=list, description="รายชื่อผู้ถือหุ้นของบริษัท")
    thai_stakeholders_number: str = Field(default="-", description="ผู้ถือหุ้น ไทย (เฉพาะจำนวนหุ้นที่เป็นตัวเลข)")
    other_stakeholders_number: str = Field(default="-", description="หุ้น อื่นๆ (เฉพาะจำนวนหุ้นที่เป็นตัวเลข)")


# doc-type -> schema, mirroring the dispatch in ocr_processing_page()
DOC_TYPE_SCHEMAS = {
    "invoice": Invoice,
    "markdown": Invoice,
    "passport": PassPortData,
    "packing_list": PackingList,
    "stock_boj5": CompanyStock,
}
