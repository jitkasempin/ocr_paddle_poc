from pydantic import BaseModel

class InvoiceInfo(BaseModel):
    invoice_number: str = Field(default="", description="เลขที่ หรือ Tax Invoice No")
    invoice_date: str = Field(default="", description="วันที่ หรือ Tax Invoice Date")
    # document_copy_status: str


class Seller(BaseModel):
    company_name: str = Field(default="", description="Company name at the top of invoice markdown text")
    address: str = Field(
        default="", description="ที่อยู่ของบริษัท หรือที่ตั้งของสำนักงานใหญ่บริษัท")
    tax_id: str = Field(
        default="", description="เลขประจำตัวผู้เสียภาษี หรือ Tax ID")
    branch_code: str = Field(default="", description="ออกโดยสาขาที่")
    telephone: str = Field(default="", description="TEL")
    fax: str = Field(default="", description="FAX")


class Buyer(BaseModel):
    company_name: str = Field(default="", description="ได้รับเงินจาก")
    address: str = Field(default="", description="Received from หรือ address ของผู้จ่ายเงิน")
    tax_id: str = Field(default="", description="เลขประจำตัวผู้เสียภาษี ของผู้ซื้อ/ผู้รับบริการ")
    # branch_type: str = Field(default="", description="ประเภทสาขา")


class Detail(BaseModel):
    pay_for: str = Field(default="", description="ชำระค่า")
    transporter_name: str = Field(default="", description="ผู้ขนส่ง")
    license_plate: str = Field(default="", description="ทะเบียนรถ")
    payment_no: str = Field(default="", description="In payment of")


class Costs(BaseModel):
    total_amount: float = Field(default=0.0, description="จำนวนเงิน(TOTAL)")
    # vat_percentage: str
    vat_amount: float = Field(default=0.0, description="ค่าของ V.A.T. 7%")
    net_total: float = Field(default=0.0, description="จำนวนเงินรวม(NET TOTAL)")
    is_bank_cheuqe: str = Field(default="", description="เป็นเช็คธนาคาร(cheuqe)")
    bank_cheuqe_no: str = Field(default="", description="เลขที่(NO) ของเช็คธนาคาร(cheuqe)")
    net_total_text: str = Field(default="", description="ค่าของ 'บาท(BAHT)' (ค่าต้องเป็นตัวอักษร เท่านั้น)")

# class PaymentDetails(BaseModel):
    # branch_type: str

# class LineItem(BaseModel):
#     description: str
#     transporter: str
#     license_plate: str
#     container_or_reference_no: str

# class Totals(BaseModel):
#     sub_total_amount: float
#     vat_percentage: str
#     vat_amount: float
#     net_total_amount: float
#     net_total_text: str

class PaymentReceiver(BaseModel):
    # payment_method: str
    collector_name: str = Field(default="", description="ผู้รับเงิน หรือ Collector")
    manager_name: str = Field(default="", description="ผู้จัดการ หรือ Manager")

class TaxInvoice(BaseModel):
    invoice_info: InvoiceInfo  = Field(
        default_factory=InvoiceInfo, description="General invoice information")
    seller: Seller = Field(default_factory=Seller, description="The information about company that issue this tax invoice")
    buyer: Buyer = Field(default_factory=Buyer, description="The information about the company that pay for the service")
    item_detail: Detail = Field(default_factory=Detail, description="The detail of service that buyer pay for")
    totals: Costs = Field(default_factory=Costs, description="The total cost and tax in this invoice")
    payment_receiver: PaymentReceiver = Field(default_factory=PaymentReceiver, description="The information about the person who receives the payment")
