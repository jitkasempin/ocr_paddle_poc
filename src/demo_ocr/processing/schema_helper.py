from typing import Dict, List, Tuple, Optional
import json
import re
from datetime import datetime

from decimal import Decimal, InvalidOperation


# Thai month mappings
thai_months = {
    'มกราคม': '01', 'ม.ค.': '01', 'มค': '01',
    'กุมภาพันธ์': '02', 'ก.พ.': '02', 'กพ': '02',
    'มีนาคม': '03', 'มี.ค.': '03', 'มีค': '03',
    'เมษายน': '04', 'เม.ย.': '04', 'เมย': '04',
    'พฤษภาคม': '05', 'พ.ค.': '05', 'พค': '05',
    'มิถุนายน': '06', 'มิ.ย.': '06', 'มิย': '06',
    'กรกฎาคม': '07', 'ก.ค.': '07', 'กค': '07',
    'สิงหาคม': '08', 'ส.ค.': '08', 'สค': '08',
    'กันยายน': '09', 'ก.ย.': '09', 'กย': '09',
    'ตุลาคม': '10', 'ต.ค.': '10', 'ตค': '10',
    'พฤศจิกายน': '11', 'พ.ย.': '11', 'พย': '11',
    'ธันวาคม': '12', 'ธ.ค.': '12', 'ธค': '12'
}

# Thai digit mappings
thai_digits = {
    '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
    '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
}

# English month mappings
english_months = {
    'jan': '01', 'january': '01',
    'feb': '02', 'february': '02',
    'mar': '03', 'march': '03',
    'apr': '04', 'april': '04',
    'may': '05',
    'jun': '06', 'june': '06',
    'jul': '07', 'july': '07',
    'aug': '08', 'august': '08',
    'sep': '09', 'september': '09',
    'oct': '10', 'october': '10',
    'nov': '11', 'november': '11',
    'dec': '12', 'december': '12'
}


def format_date(date_obj: datetime) -> str:
    return date_obj.strftime("%d.%m.%Y")

# ---------- helper ----------


def parse_decimal_like(value) -> Decimal:
    """
    Self-healing converter:
    - None -> Decimal('0.00')
    - int/float/Decimal -> Decimal
    - str -> strip, remove commas (',' and '，'), support parentheses negatives "(1,234)" -> -1234
    - raise ValueError ถ้าแปลงไม่ได้
    """
    if value is None:
        return Decimal("0.00")

    if isinstance(value, Decimal):
        return value

    if isinstance(value, (int, float)):
        # หลีกเลี่ยง float binary ไว้ก่อนด้วยการแปลงผ่าน str
        return Decimal(str(value))

    if isinstance(value, str):
        s = value.strip()

        # วงเล็บหมายถึงลบ เช่น (1,234.50)
        negative = s.startswith("(") and s.endswith(")")
        if negative:
            s = s[1:-1].strip()

        # ลบคอมมาแบบ half/full width
        s = s.replace(",", "").replace("，", "")

        # กรณีมีช่องว่างคั่นหลักพัน เช่น "1 234.50" -> "1234.50"
        s = s.replace(" ", "")

        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError) as e:
            d = Decimal(0)
            print(f"Cannot parse decimal from: {value!r} set to 0")
            # raise ValueError(f"Cannot parse decimal from: {value!r}") from e

        return -d if negative else d

    # ชนิดอื่น ๆ ไม่รองรับ
    raise ValueError(
        f"Unsupported type for decimal field: {type(value).__name__}")


def parse_float_like(value) -> float:
    """
    Self-healing for float fields using Decimal as an accurate intermediary.
    """
    d = parse_decimal_like(value)
    return float(d)


# Convert year to CE
def convert_year_to_ce(year: int) -> int:

    if year > 2400:  # BE (full year)
        return year - 543
    elif year > 50 and year < 100:  # 2-digit BE (51-99)
        return (year + 2500) - 543  # 68 -> 2568 -> 2025
    elif year <= 50:  # 2-digit CE (00-50)
        return year + 2000  # 25 -> 2025
    else:  # Full CE year
        return year


def convert_thai_digits(text: str) -> str:

    for thai_digit, arabic_digit in thai_digits.items():
        text = text.replace(thai_digit, arabic_digit)
    return text


def parse_thai_date(date_str: str) -> Optional[str]:
    """Parse Thai date string to datetime object with enhanced format support"""
    if not date_str:
        return None

    date_str = date_str.strip()

    # Skip invalid dates
    if date_str in ['0/0/0', '00/00/00', '']:
        return None

    # Convert Thai digits to Arabic digits
    date_str = convert_thai_digits(date_str)

    # Pattern 1: DD/MM/YY , DD/MM/YYYY
    pattern1 = r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})'
    match1 = re.search(pattern1, date_str)
    if match1:
        day, month, year = match1.groups()
        year = convert_year_to_ce(int(year))

        try:
            return format_date(datetime(year, int(month), int(day)))
        except:
            pass

    # Pattern 2: DD-MM-YY , DD-MM-YYYY
    pattern2 = r'(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{2,4})'
    match2 = re.search(pattern2, date_str)
    if match2:
        day, month, year = match2.groups()
        year = convert_year_to_ce(int(year))

        try:
            return format_date(datetime(year, int(month), int(day)))
        except:
            pass

    # Pattern 3: DD.MM.YY or DD.MM.YYYY
    pattern3 = r'(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{2,4})'
    match3 = re.search(pattern3, date_str)
    if match3:
        day, month, year = match3.groups()
        year = convert_year_to_ce(int(year))

        try:
            return format_date(datetime(year, int(month), int(day)))
        except:
            pass

    # Pattern 4: YYYY-MM-DD
    pattern4 = r'(\d{4})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})'
    match4 = re.search(pattern4, date_str)
    if match4:
        year, month, day = match4.groups()
        year = convert_year_to_ce(int(year))

        try:
            return format_date(datetime(year, int(month), int(day)))
        except:
            pass

    # Pattern 5: DD-MMM-YY or DD-MMM-YYYY
    pattern5 = r'(\d{1,2})\s*-\s*([a-zA-Z]{3,9})\s*-\s*(\d{2,4})'
    match5 = re.search(pattern5, date_str)
    if match5:
        day, month_str, year = match5.groups()
        month_num = english_months.get(month_str.lower())
        if month_num:
            year = convert_year_to_ce(int(year))

            try:
                return format_date(datetime(year, int(month_num), int(day)))
            except:
                pass

    # Pattern 6: DD/MMM/YY or DD MMM YY
    pattern6 = r'(\d{1,2})\s*[/\s]\s*([a-zA-Z]{3,9})\s*[/\s]\s*(\d{2,4})'
    match6 = re.search(pattern6, date_str)
    if match6:
        day, month_str, year = match6.groups()
        month_num = english_months.get(month_str.lower())
        if month_num:
            year = convert_year_to_ce(int(year))

            try:
                return format_date(datetime(year, int(month_num), int(day)))
            except:
                pass

    # Pattern 7: MMM DD, YYYY
    pattern7 = r'([a-zA-Z]{3,9})\s+(\d{1,2}),?\s*(\d{2,4})'
    match7 = re.search(pattern7, date_str)
    if match7:
        month_str, day, year = match7.groups()
        month_num = english_months.get(month_str.lower())
        if month_num:
            year = convert_year_to_ce(int(year))

            try:
                return format_date(datetime(year, int(month_num), int(day)))
            except:
                pass

    # Pattern 8: DD month_name YY/YYYY (Thai months)
    for thai_month, month_num in thai_months.items():
        if thai_month in date_str:
            pattern8 = rf'(\d{{1,2}})\s*{re.escape(thai_month)}\s*(\d{{2,4}})'
            match8 = re.search(pattern8, date_str)
            if match8:
                day, year = match8.groups()
                year = convert_year_to_ce(int(year))

                try:
                    return format_date(datetime(year, int(month_num), int(day)))
                except:
                    pass
            break

    return None

def extract_po_decimal(ocr_text: str) -> Optional[str]:
    """
    Extract a decimal number string from OCR text.
    
    This function handles various OCR text formats including:
    - Different prefixes (PO, po, purchase number, etc.)
    - Various punctuation (periods, colons, spaces)
    - Trailing/leading whitespace
    - Numbers with or without separators
    
    Args:
        ocr_text: The OCR text containing a decimal number
        
    Returns:
        The extracted decimal number as a string, or None if no number found
        
    Examples:
        >>> extract_decimal("PO. 45000341")
        '45000341'
        >>> extract_decimal("po. 66000178")
        '66000178'
        >>> extract_decimal("PO 67000258")
        '67000258'
        >>> extract_decimal("PO NUMBER 45000456")
        '45000456'
        >>> extract_decimal("44000867.")
        '44000867'
        >>> extract_decimal("43000546 ")
        '43000546'
        >>> extract_decimal("Po:86000455")
        '86000455'
        >>> extract_decimal("purchase number: 99000333")
        '99000333'
    """
    if not ocr_text:
        return None
    
    # Find all sequences of consecutive digits
    # This pattern matches one or more digits
    matches = re.findall(r'\d+', ocr_text)
    
    if not matches:
        return None
    
    # Return the longest sequence of digits found
    # This handles cases where there might be multiple digit sequences
    longest_match = max(matches, key=len)
    return longest_match




def extract_code(text: str) -> Optional[str]:
    """
    Extract numeric code value from text containing CODE or MAT CODE keywords.

    Args:
        text: Input string that may contain code information

    Returns:
        Extracted numeric code as string, or None if not found

    Examples:
        >>> extract_code("Screw Cap SS/SUS304 (200/กล) CODE : 46924")
        '46924'
        >>> extract_code("สกรูหัวเหลี่ยม SS/SUS304 M14-2.0 (100/กล) MAT CODE : 1166")
        '1166'
        >>> extract_code("Product code: 12345")
        '12345'
        >>> extract_code("No code here")
        None
    """
    if not text:
        return None

    # Pattern explanation:
    # (?:mat\s+)?  - Optional "mat" followed by whitespace (non-capturing group)
    # code         - The word "code" (case-insensitive due to re.IGNORECASE)
    # \s*          - Optional whitespace
    # [:：]?       - Optional colon (ASCII or full-width)
    # \s*          - Optional whitespace
    # (\d+)        - Capture group: one or more digits (0-9)
    pattern = r'(?:mat\s+)?code\s*[:：]?\s*(\d+)'

    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def extract_only_branch_code_number(text: str) -> Optional[str]:
    """
    Implement the robust python script in this function to extract branch code as the numeric string with one lenght long from text.
    The branch code must be in this set: {"8", "3", "7"} only, otherwise return None.
    For example:
    text: "สาขาที่ 00008" -> return "8"
    text: "สาขาที่ 00003" -> return "3"
    text: "สาขาที่ 00007" -> return "7"
    text: "สาขาที่ 00005" -> return None
    text: "สาขาที่ 0008" -> return "8"
    text: "สาขาที่ 0003" -> return "3"
    text: "สาขาที่ 8" -> return "8"
    text: "สาขาที่ 3" -> return "3"
    text: "00008" -> return "8"
    text: "00003" -> return "3"
    text: "00007" -> return "7"
    text: "00005" -> return None
    text: "0008" -> return "8"
    text: "0003" -> return "3"
    text: "8" -> return "8"
    text: "3" -> return "3"
    text: "0009" -> return None
    text: "000018" -> return None
    text" "00018" -> return None
    text: "5" -> return None
    """
    if not text:
        return None

    valid_codes = {"8", "3", "7"}

    pattern = r'\d+'
    match = re.search(pattern, text)

    if match:
        number_str = match.group(0).lstrip('0') or '0'

        if len(number_str) == 1 and number_str in valid_codes:
            return number_str

    return None


def extract_code_strict(text: str) -> Optional[str]:
    """
    Extract numeric code with stricter pattern matching.
    Only matches when "CODE" or "MAT CODE" is clearly separated.

    Args:
        text: Input string that may contain code information

    Returns:
        Extracted numeric code as string, or None if not found
    """
    if not text:
        return None

    # Stricter pattern that requires word boundaries
    pattern = r'\b(?:mat\s+)?code\s*[:：]?\s*(\d+)\b'

    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def extract_all_codes(text: str) -> list[str]:
    """
    Extract all CODE or MAT CODE values from text.
    Useful when multiple codes might be present.

    Args:
        text: Input string that may contain multiple codes

    Returns:
        List of all extracted numeric codes
    """
    if not text:
        return []

    pattern = r'(?:mat\s+)?code\s*[:：]?\s*(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    return matches
