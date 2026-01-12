"""
Invoice Data Extractor Integration Module

This module provides easy integration of repetition detection with
invoice data extractor projects using OCR models from Huggingface.

Usage:
    from invoice_integration import OCROutputValidator, validate_ocr_output
    
    # Quick validation
    is_valid, result = validate_ocr_output(ocr_markdown_text)
    
    # Or use the validator class for more control
    validator = OCROutputValidator()
    validation_result = validator.validate(ocr_markdown_text)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum

from .repetition_detector import (
    detect_repetition,
    is_repetitive,
    DetectionResult,
    RepetitionMatch,
    RepetitionType
)


class ValidationStatus(Enum):
    """Status of OCR output validation."""
    VALID = "valid"
    WARNING = "warning"  # Some repetition detected but below threshold
    INVALID = "invalid"  # Significant repetition detected


@dataclass
class ValidationResult:
    """Result of OCR output validation."""
    status: ValidationStatus
    is_repetitive: bool
    detection_result: DetectionResult
    message: str
    confidence_score: float  # 0.0 to 1.0, higher means more confident the output is valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "is_repetitive": self.is_repetitive,
            "message": self.message,
            "confidence_score": self.confidence_score,
            "repetitive_content_ratio": self.detection_result.repetitive_content_ratio,
            "total_repetitions": len(self.detection_result.repetitions),
            "original_length": self.detection_result.original_length
        }


class OCROutputValidator:
    """
    Validator for OCR output from invoice processing.
    
    This class wraps the repetition detection functionality and provides
    a simple interface for validating OCR output before further processing.
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.1,
        invalid_threshold: float = 0.3,
        min_phrase_length: int = 20,
        min_repetitions: int = 2
    ):
        """
        Initialize the OCR output validator.
        
        Args:
            warning_threshold: Repetition ratio to trigger warning (0.0 to 1.0)
            invalid_threshold: Repetition ratio to mark as invalid (0.0 to 1.0)
            min_phrase_length: Minimum phrase length to consider as repetition
            min_repetitions: Minimum occurrences to count as repetition
        """
        self.warning_threshold = warning_threshold
        self.invalid_threshold = invalid_threshold
        self.min_phrase_length = min_phrase_length
        self.min_repetitions = min_repetitions
    
    def validate(self, ocr_output: str) -> ValidationResult:
        """
        Validate OCR output for repetitive content.
        
        Args:
            ocr_output: The markdown/text output from OCR model
            
        Returns:
            ValidationResult with status, detection details, and confidence score
        """
        if not ocr_output or len(ocr_output.strip()) == 0:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                is_repetitive=False,
                detection_result=DetectionResult(
                    has_repetition=False,
                    repetitions=[],
                    total_repetition_count=0,
                    repetitive_content_ratio=0.0,
                    original_length=0,
                    detection_method="none"
                ),
                message="Empty OCR output",
                confidence_score=0.0
            )
        
        # Run detection
        detection_result = detect_repetition(
            ocr_output,
            min_phrase_length=self.min_phrase_length,
            min_repetitions=self.min_repetitions
        )
        
        # Determine status based on thresholds
        ratio = detection_result.repetitive_content_ratio
        
        if ratio >= self.invalid_threshold:
            status = ValidationStatus.INVALID
            message = f"High repetition detected ({ratio:.1%}). OCR output may be corrupted."
            confidence = max(0.0, 1.0 - ratio)
        elif ratio >= self.warning_threshold:
            status = ValidationStatus.WARNING
            message = f"Some repetition detected ({ratio:.1%}). Review recommended."
            confidence = max(0.2, 1.0 - ratio)
        else:
            status = ValidationStatus.VALID
            message = "OCR output appears valid."
            confidence = min(1.0, 1.0 - ratio + 0.1)
        
        return ValidationResult(
            status=status,
            is_repetitive=detection_result.has_repetition,
            detection_result=detection_result,
            message=message,
            confidence_score=confidence
        )
    
    def is_valid(self, ocr_output: str) -> bool:
        """
        Quick check if OCR output is valid (not significantly repetitive).
        
        Args:
            ocr_output: The markdown/text output from OCR model
            
        Returns:
            True if output is valid, False if repetitive
        """
        result = self.validate(ocr_output)
        return result.status != ValidationStatus.INVALID


def validate_ocr_output(
    ocr_output: str,
    invalid_threshold: float = 0.3
) -> Tuple[bool, ValidationResult]:
    """
    Convenience function to validate OCR output.
    
    Args:
        ocr_output: The markdown/text output from OCR model
        invalid_threshold: Threshold for marking as invalid
        
    Returns:
        Tuple of (is_valid, validation_result)
    """
    validator = OCROutputValidator(invalid_threshold=invalid_threshold)
    result = validator.validate(ocr_output)
    return result.status != ValidationStatus.INVALID, result


def flag_repetitive_output(
    ocr_output: str,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Flag repetitive OCR output and return detailed information.
    
    This function is designed for easy integration into existing pipelines.
    
    Args:
        ocr_output: The markdown/text output from OCR model
        threshold: Repetition ratio threshold for flagging
        
    Returns:
        Dictionary with flagging information:
        {
            "flagged": bool,
            "repetitive_ratio": float,
            "repetition_count": int,
            "top_repetitions": List[Dict],
            "message": str
        }
    """
    detection_result = detect_repetition(ocr_output)
    
    flagged = detection_result.repetitive_content_ratio >= threshold
    
    # Get top repetitions for reporting
    top_repetitions = []
    sorted_reps = sorted(
        detection_result.repetitions,
        key=lambda x: x.count,
        reverse=True
    )[:5]
    
    for rep in sorted_reps:
        preview = rep.phrase[:100] + "..." if len(rep.phrase) > 100 else rep.phrase
        top_repetitions.append({
            "phrase_preview": preview,
            "count": rep.count,
            "phrase_length": rep.phrase_length,
            "type": rep.repetition_type.value
        })
    
    message = ""
    if flagged:
        message = f"⚠️ OCR output flagged: {detection_result.repetitive_content_ratio:.1%} repetitive content detected"
    else:
        message = "✓ OCR output passed repetition check"
    
    return {
        "flagged": flagged,
        "repetitive_ratio": detection_result.repetitive_content_ratio,
        "repetition_count": len(detection_result.repetitions),
        "top_repetitions": top_repetitions,
        "message": message,
        "original_length": detection_result.original_length
    }


class InvoiceOCRPipeline:
    """
    Example pipeline integration class showing how to use repetition
    detection in an invoice processing workflow.
    """
    
    def __init__(
        self,
        ocr_model=None,
        llm_model=None,
        repetition_threshold: float = 0.3
    ):
        """
        Initialize the pipeline.
        
        Args:
            ocr_model: The OCR model instance (e.g., from Huggingface)
            llm_model: The LLM model for JSON extraction
            repetition_threshold: Threshold for flagging repetitive output
        """
        self.ocr_model = ocr_model
        self.llm_model = llm_model
        self.validator = OCROutputValidator(invalid_threshold=repetition_threshold)
    
    def process_invoice(self, image_path: str) -> Dict[str, Any]:
        """
        Process an invoice image through the pipeline.
        
        Args:
            image_path: Path to the invoice image
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "status": "success",
            "image_path": image_path,
            "ocr_output": None,
            "validation": None,
            "extracted_data": None,
            "error": None
        }
        
        try:
            # Step 1: OCR processing (placeholder - integrate your OCR model)
            if self.ocr_model:
                ocr_output = self.ocr_model.process(image_path)
            else:
                # Placeholder for demonstration
                ocr_output = self._mock_ocr_output()
            
            result["ocr_output"] = ocr_output
            
            # Step 2: Validate OCR output for repetitions
            validation = self.validator.validate(ocr_output)
            result["validation"] = validation.to_dict()
            
            # Step 3: Check if we should proceed
            if validation.status == ValidationStatus.INVALID:
                result["status"] = "flagged"
                result["error"] = validation.message
                return result
            
            # Step 4: Extract structured data (placeholder - integrate your LLM)
            if self.llm_model and validation.status != ValidationStatus.INVALID:
                extracted_data = self.llm_model.extract(ocr_output)
                result["extracted_data"] = extracted_data
            
            # Add warning if validation status is WARNING
            if validation.status == ValidationStatus.WARNING:
                result["status"] = "warning"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _mock_ocr_output(self) -> str:
        """Mock OCR output for demonstration."""
        return """
        | Invoice Number | INV-2024-001 |
        | Date | 2024-01-15 |
        | Customer | ABC Company |
        | Total | $1,500.00 |
        """


# Example usage
if __name__ == "__main__":
    # Test with sample repetitive OCR output
    sample_ocr_output = """
    <table>
    <tr><td>บริษัท ช้อยส์ จำกัด</td></tr><tr><td>ที่อยู่</td><td>18/1 หมู่ 1 ถนนศรีนครินทร์ ตำบลบางพลัด อำเภอเมือง สมุทรสาคร จ.สมุทรสาคร 74000</td></tr><tr><td>เลขที่ใบกำกับภาษี</td><td>000071131150</td></tr><tr><td>วันที่ใบกำกับภาษี</td><td>20 ธ.ค. 60</td></tr><tr><td>สถานะใบกำกับภาษี</td><td>รอตรวจสอบ</td></tr><tr><td>รหัสลูกค้า</td><td>000071131150</td></tr><tr><td>ชื่อลูกค้า</td>
    <tr><td>บริษัท ช้อยส์ จำกัด</td></tr><tr><td>ที่อยู่</td><td>18/1 หมู่ 1 ถนนศรีนครินทร์ ตำบลบางพลัด อำเภอเมือง สมุทรสาคร จ.สมุทรสาคร 74000</td></tr><tr><td>เลขที่ใบกำกับภาษี</td><td>000071131150</td></tr><tr><td>วันที่ใบกำกับภาษี</td><td>20 ธ.ค. 60</td></tr><tr><td>สถานะใบกำกับภาษี</td><td>รอตรวจสอบ</td></tr><tr><td>รหัสลูกค้า</td><td>000071131150</td></tr><tr><td>ชื่อลูกค้า</td>
    <tr><td>บริษัท ช้อยส์ จำกัด</td></tr><tr><td>ที่อยู่</td><td>18/1 หมู่ 1 ถนนศรีนครินทร์ ตำบลบางพลัด อำเภอเมือง สมุทรสาคร จ.สมุทรสาคร 74000</td></tr><tr><td>เลขที่ใบกำกับภาษี</td><td>000071131150</td></tr><tr><td>วันที่ใบกำกับภาษี</td><td>20 ธ.ค. 60</td></tr><tr><td>สถานะใบกำกับภาษี</td><td>รอตรวจสอบ</td></tr><tr><td>รหัสลูกค้า</td><td>000071131150</td></tr><tr><td>ชื่อลูกค้า</td>
    </table>
    """
    
    print("=" * 60)
    print("Invoice Integration Test")
    print("=" * 60)
    
    # Test 1: Quick validation
    is_valid, result = validate_ocr_output(sample_ocr_output)
    print(f"\n1. Quick Validation:")
    print(f"   Is Valid: {is_valid}")
    print(f"   Status: {result.status.value}")
    print(f"   Message: {result.message}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    
    # Test 2: Flag repetitive output
    print(f"\n2. Flag Repetitive Output:")
    flag_result = flag_repetitive_output(sample_ocr_output)
    print(f"   Flagged: {flag_result['flagged']}")
    print(f"   Repetitive Ratio: {flag_result['repetitive_ratio']:.1%}")
    print(f"   Message: {flag_result['message']}")
    
    # Test 3: Validator class
    print(f"\n3. Validator Class:")
    validator = OCROutputValidator()
    validation = validator.validate(sample_ocr_output)
    print(f"   Status: {validation.status.value}")
    print(f"   Detection Method: {validation.detection_result.detection_method}")
    print(f"   Repetitions Found: {len(validation.detection_result.repetitions)}")
    
    # Test with clean output
    clean_output = """
    | Invoice Number | INV-2024-001 |
    | Date | 2024-01-15 |
    | Customer | ABC Company |
    | Item 1 | Widget A | $500.00 |
    | Item 2 | Widget B | $1,000.00 |
    | Total | $1,500.00 |
    """
    
    print(f"\n4. Clean Output Test:")
    is_valid, result = validate_ocr_output(clean_output)
    print(f"   Is Valid: {is_valid}")
    print(f"   Status: {result.status.value}")
    print(f"   Message: {result.message}")
