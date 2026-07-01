from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from .schema_helper import parse_decimal_like, parse_float_like
from .schema_helper import extract_po_decimal


from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, model_validator


class Document(BaseModel):
    po_number: str = Field(
        default="", description="ใบสั่งซื้อ หรือ P/O NO หรือ เลขที่ใบสั่งซื้อ หรือ PO NO หรือ Purchase order number หรือ เลขที่ PO")
    date: str = Field(
        default="", description="วันที่ออกใบสั่งซื้อ หรือ DATE หรือ Purchase Order Date หรือ วันที่ ที่ได้เขียนไว้บน Purchase Order")

    @field_validator("po_number", mode="before")
    @classmethod
    def _heal_po_number(cls, v):
        parsed_for_po = extract_po_decimal(v)
        if parsed_for_po is None:
            parsed_for_po = ""

        return parsed_for_po


class VendorCompany(BaseModel):
    name: str = Field(
        default="", description="ชื่อของบริษัทที่เป็น Vendor ซึ่งอยู่ใต้ supplier code")
    supplier_code: str = Field(
        default="", description="เลขที่ของ Supplier code")


class AdityaCompany(BaseModel):
    name: str = Field(
        default="", description="Full company name of Aditya. Must begin with 'ADITYA BIRLA ...'")
    tax_id: str = Field(
        default="", description="TAX ID of Aditya Birla Company")
    address: str = Field(
        default="", description="Address of Aditya Birla Company")
    contact: Optional[str] = Field(
        default=None, description="Contact information of Aditya Birla Company such as phone number or email")
    branch_name: Optional[str] = Field(
        default=None,
        description="ชื่อเรียกของสาขาหรือชื่อของ Division"
    )
    branch_code: Optional[str] = Field(
        default=None,
        description="รหัสสาขาที่เป็นตัวเลขเท่านั้น มักปรากฏหลังคำว่า 'Branch' หรือ 'สาขา' (เช่น Branch: 00008 ให้ดึง '00008') ห้ามตัดเลข 0 ข้างหน้าทิ้ง"
    )

    @field_validator("tax_id", mode="before")
    @classmethod
    def _heal_tax_id(cls, v):
        if v is not None:
            # if v contain the space character or tab character then replace it with empty string
            v = v.replace(" ", "").replace("\t", "")
            # return extract_tax_id(v)
        return v


class PoItem(BaseModel):
    description: str = Field(
        "", description="Description หรือ รายละเอียด หรือ รหัสสินค้า หรือ รายการ หรือ Description of goods")
    quantity: float = Field(
        0, description="จำนวน หรือ Quantity หรือ Qty หรือ Order Quantity")
    unit_price: float = Field(
        0, description="ราคาต่อหน่วย หรือ ราคา/หน่วย หรือ หน่วยละ หรือ Unit Price หรือ Unit")
    amount: float = Field(
        0, description="จำนวนเงิน หรือ Amount หรือ Total หรือ Total Amount หรือ Total Price")
    code: str = Field("", description="CODE หรือ Mat.Code")
    uom: str = Field("", description="UOM หรือ Unit of Measure")

    # --- self-healing numeric fields (float) ---
    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _heal_item_numbers(cls, v):
        return parse_float_like(v)


class PoSummary(BaseModel):
    subtotal: Decimal = Field(default=Decimal(
        "0.00"), description="The value of 'TOTAL AMOUNT'")
    vat: Decimal = Field(
        default=Decimal("0.00"), description="The value from VAT, V.A.T, ภาษี or VAT amount")

    vat_rate: Optional[int] = Field(
        default=0, description="VAT rate in percent only (The number must follow by '%'). Output only the integer number.")
    total_amount: Decimal = Field(default=Decimal(
        "0.00"), description="The value of 'GRAND TOTAL'")
    currency: Literal["BAHT", "THB", "บาท", "US DOLLAR", "USD", "SING.DOLLAR", "SGD", "POUND STERING", "EURO", "GBP", "RUPEE", "INR", "SWISS FRANC", "CHF", "CHINESE YUAN", "CNY", "YEN", "JPY", "CANADIAN DOLLAR", "CAD"] = Field(
        default="THB", description="CURRENCY in Purchase Order เช่น BAHT หรือ THB หรือ บาท หรือ US DOLLAR หรือ USD หรือ SING.DOLLAR หรือ SGD หรือ POUND STERING หรือ EURO หรือ GBP หรือ RUPEE หรือ INR หรือ SWISS FRANC หรือ CHF หรือ CHINESE YUAN หรือ CNY หรือ YEN หรือ JPY หรือ CANADIAN DOLLAR หรือ CAD")
    # --- self-healing decimal field ---

    @field_validator("subtotal", "vat", "total_amount", mode="before")
    @classmethod
    def _heal_summary_total(cls, v):
        return parse_decimal_like(v)

    # @model_validator(mode="after")
    # def check_and_adjust_total_amount(self):
    #     # Recalculate total_amount if it's 0 or less than subtotal
    #     if self.total_amount == Decimal("0.00") or self.total_amount < self.subtotal:
    #         self.total_amount = self.subtotal + self.vat
    #     return self


class PurchaseOrder(BaseModel):
    document: Document = Field(
        default_factory=Document, description="General purchase order information consist of purchase order number (PO NO) and date")
    # seller: SellerCompany = Field(default_factory=SellerCompany,
    #   description="The information about company that issue the invoice (company that sell the products to customer)")

    seller: VendorCompany = Field(
        default_factory=VendorCompany, description="The information about the vendor company that sell the products to Aditya Birla")

    customer: AdityaCompany = Field(
        default_factory=AdityaCompany, description="The information about the company that buy the products from seller")

    items: List[PoItem] = Field(
        default_factory=list, description="List ของรายการสินค้าที่อยู่ใน Purchase Order นี้เท่านั้น")

    summary: PoSummary = Field(default_factory=PoSummary,
                               description="Purchase Order summary")
