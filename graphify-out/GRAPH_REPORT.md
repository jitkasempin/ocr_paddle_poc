# Graph Report - /home/jitkasem/ocr_paddle_poc  (2026-05-05)

## Corpus Check
- Large corpus: 119 files · ~541,192 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 527 nodes · 853 edges · 40 communities (35 shown, 5 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 161 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Hybrid Search Engine|Hybrid Search Engine]]
- [[_COMMUNITY_DotsOCR Model Inference|DotsOCR Model Inference]]
- [[_COMMUNITY_Typhoon OCR Pipeline|Typhoon OCR Pipeline]]
- [[_COMMUNITY_OCR Page Processing|OCR Page Processing]]
- [[_COMMUNITY_Invoice OCR No-Flash|Invoice OCR No-Flash]]
- [[_COMMUNITY_Output Cleaner Utils|Output Cleaner Utils]]
- [[_COMMUNITY_Core Extraction & Client|Core Extraction & Client]]
- [[_COMMUNITY_Repetition Detection Logic|Repetition Detection Logic]]
- [[_COMMUNITY_OCR Page Extraction|OCR Page Extraction]]
- [[_COMMUNITY_Document Classification|Document Classification]]
- [[_COMMUNITY_Invoice Integration Core|Invoice Integration Core]]
- [[_COMMUNITY_Format Transformer Utils|Format Transformer Utils]]
- [[_COMMUNITY_Invoice OCR Pipeline|Invoice OCR Pipeline]]
- [[_COMMUNITY_Invoice Viewer UI|Invoice Viewer UI]]
- [[_COMMUNITY_Consecutive Repetition Detector|Consecutive Repetition Detector]]
- [[_COMMUNITY_VLLM Backend|VLLM Backend]]
- [[_COMMUNITY_OCR Output Validator|OCR Output Validator]]
- [[_COMMUNITY_Invoice Streamlit App|Invoice Streamlit App]]
- [[_COMMUNITY_File Converters|File Converters]]
- [[_COMMUNITY_HTML Table Repetition Detector|HTML Table Repetition Detector]]
- [[_COMMUNITY_OCR No-Flash Chat Messages|OCR No-Flash Chat Messages]]
- [[_COMMUNITY_Demo Display Utils|Demo Display Utils]]
- [[_COMMUNITY_Detection Result Model|Detection Result Model]]
- [[_COMMUNITY_OCR Demo Main|OCR Demo Main]]
- [[_COMMUNITY_Test Imports|Test Imports]]
- [[_COMMUNITY_Hybrid Search Rationale|Hybrid Search Rationale]]
- [[_COMMUNITY_Hybrid Search Rationale 2|Hybrid Search Rationale 2]]
- [[_COMMUNITY_OpenCLIP Rationale|OpenCLIP Rationale]]

## God Nodes (most connected - your core abstractions)
1. `OCR` - 47 edges
2. `HybridSearch` - 31 edges
3. `StampDetection` - 29 edges
4. `PurchaseOrder` - 23 edges
5. `OutputCleaner` - 18 edges
6. `DetectionResult` - 15 edges
7. `RepetitionDetector` - 15 edges
8. `ocr_processing_page()` - 14 edges
9. `Qwen3VLLMClient` - 14 edges
10. `RepetitionMatch` - 13 edges

## Surprising Connections (you probably didn't know these)
- `ocr_processing_page()` --calls--> `convert_to_markdown_stream()`  [INFERRED]
  processing/ocr_page.py → core/pdf2md/pdf2md.py
- `is_match_centroids()` --calls--> `crop_top_percent()`  [INFERRED]
  processing/ocr_page.py → doc_classification/zero_shot.py
- `is_match_centroids()` --calls--> `get_classifier()`  [INFERRED]
  processing/ocr_page.py → doc_classification/zero_shot.py
- `_heal_summary_total()` --calls--> `parse_decimal_like()`  [INFERRED]
  processing/po_schema.py → processing/schema_helper.py
- `ValidationStatus` --uses--> `DetectionResult`  [INFERRED]
  processing/invoice_integration.py → processing/repetition_detector.py

## Communities (40 total, 5 thin omitted)

### Community 0 - "Hybrid Search Engine"
Cohesion: 0.05
Nodes (49): BaseModel, generate_embeddings(), HybridSearch, Process complete dataset: embed and store in Qdrant.                  Args:, Search for similar vectors., Normalize an array of vectors to unit length., Format texts according to GritLM requirements., # TODO: implement this function (+41 more)

### Community 1 - "DotsOCR Model Inference"
Cohesion: 0.07
Nodes (30): DotsOCRParser, main(), parse image or pdf file, inference_with_vllm(), fitz_doc_to_image(), load_images_from_pdf(), PageInfo, The width and height of page (+22 more)

### Community 2 - "Typhoon OCR Pipeline"
Cohesion: 0.08
Nodes (33): Run OCR prediction using Typhoon model deployed on RunPod via vLLM.         Logs, Typhoon OCR is a model for extracting structured markdown from images or PDFs., BoundingBox, _cap_split_string(), _cleanup_element_text(), Element, ensure_image_in_path(), from_rectangle() (+25 more)

### Community 3 - "OCR Page Processing"
Cohesion: 0.06
Nodes (34): _check_required_fields_not_empty(), extract_code_from_description(), _heal_all_dates(), _heal_branch_code(), _heal_item_decimals(), _heal_po_number(), _heal_summary_decimals(), AdityaCompany (+26 more)

### Community 4 - "Invoice OCR No-Flash"
Cohesion: 0.08
Nodes (14): InvoiceIssueDate, InvoicePaymentDate, InvoiceIssueDate, InvoicePaymentDate, OCR, Classifies a document image into 'Invoice', 'Quotation', or 'Other'.          Ar, Converts the first page of a PDF file to a PIL Image object.          Args:, Qwen3VLLMClient (+6 more)

### Community 5 - "Output Cleaner Utils"
Cohesion: 0.1
Nodes (17): CleanedData, main(), OutputCleaner, Cleans string-type data, Fixes missing delimiters, Truncates the last incomplete element, Removes duplicate complete dict objects, preserving original order, Data structure for cleaned data (+9 more)

### Community 6 - "Core Extraction & Client"
Cohesion: 0.1
Nodes (23): sync_request(), get_fields_confidence_score_messages_binary(), extract_fields_from_documents(), extract_information(), extract_tables_from_documents(), get_fields_messages(), _get_fields_output_format(), _get_name_desc_prompt() (+15 more)

### Community 7 - "Repetition Detection Logic"
Cohesion: 0.13
Nodes (14): Initialize the repetition detector.                  Args:             min_phras, Detect repetitions in the given text using the best available method., Use multiple detection methods and combine results for better accuracy., Detect repetitions using suffix array and longest repeated substring approach., Detect repetitions using N-gram frequency analysis.                  This method, Detect repetitions using rolling hash (Rabin-Karp style) approach., Calculate the longest common prefix length of two strings., Try to extend an n-gram to find the longest repeated phrase. (+6 more)

### Community 8 - "OCR Page Extraction"
Cohesion: 0.09
Nodes (20): delta_items_ui(), extract_and_render_content(), extract_html_table(), get_items_from_html_table(), html_table_to_dataframe(), load_image_as_rgb_array(), ocr_processing_page(), parse_markdown_pages() (+12 more)

### Community 9 - "Document Classification"
Cohesion: 0.16
Nodes (11): classify_open_set(), crop_top_percent(), get_classifier(), Crop the top portion of an image by a given percentage of its height (full width, ZeroShotCentroidClassifier, embed_rgb_np(), get_openclip_backend(), _l2() (+3 more)

### Community 10 - "Invoice Integration Core"
Cohesion: 0.17
Nodes (15): Enum, flag_repetitive_output(), Invoice Data Extractor Integration Module  This module provides easy integration, Flag repetitive OCR output and return detailed information.          This functi, Status of OCR output validation., ValidationStatus, detect_repetition(), _filter_substrings() (+7 more)

### Community 11 - "Format Transformer Utils"
Cohesion: 0.21
Nodes (12): clean_latex_preamble(), clean_text(), fix_streamlit_formulas(), get_formula_in_markdown(), has_latex_markdown(), layoutjson2md(), Checks if a string contains LaTeX markdown patterns.          Args:         text, Cleans text by removing extra whitespace.          Args:         text: The origi (+4 more)

### Community 12 - "Invoice OCR Pipeline"
Cohesion: 0.18
Nodes (8): InvoiceOCRPipeline, Example pipeline integration class showing how to use repetition     detection i, Initialize the pipeline.                  Args:             ocr_model: The OCR m, Process an invoice image through the pipeline.                  Args:, Mock OCR output for demonstration., Result of OCR output validation., Convert to dictionary for JSON serialization., ValidationResult

### Community 13 - "Invoice Viewer UI"
Cohesion: 0.23
Nodes (11): display_point_details(), fetch_qdrant_data(), init_qdrant(), invoice_viewer_page(), load_css(), process_qdrant_data(), Process Qdrant points into a pandas DataFrame., Display detailed view of a single point. (+3 more)

### Community 14 - "Consecutive Repetition Detector"
Cohesion: 0.18
Nodes (7): ConsecutiveRepetitionDetector, Represents a detected repetition in the text., Specialized detector for consecutive exact phrase repetitions.          This det, Check if all occurrences are consecutive., Initialize the consecutive repetition detector.                  Args:, Detect consecutive repetitions using regex-based approach.                  This, RepetitionMatch

### Community 15 - "VLLM Backend"
Cohesion: 0.22
Nodes (5): Run the server in a background thread and wait for readiness., Start the vLLM server in a background thread., Wait until the vLLM server is ready., Stop the vLLM server gracefully., VLLMServer

### Community 16 - "OCR Output Validator"
Cohesion: 0.24
Nodes (7): OCROutputValidator, Quick check if OCR output is valid (not significantly repetitive)., Convenience function to validate OCR output.          Args:         ocr_output:, Validator for OCR output from invoice processing.          This class wraps the, Initialize the OCR output validator.                  Args:             warning_, Validate OCR output for repetitive content.                  Args:             o, validate_ocr_output()

### Community 17 - "Invoice Streamlit App"
Cohesion: 0.29
Nodes (9): display_invoice_details(), fetch_invoice_data(), init_supabase(), main(), process_invoice_data(), Display detailed view of a single invoice., Initialize Supabase client with credentials from secrets., Fetch invoice data from Supabase or return empty list. (+1 more)

### Community 18 - "File Converters"
Cohesion: 0.28
Nodes (4): ABC, FileConverter, PDFConverter, FileConverter

### Community 19 - "HTML Table Repetition Detector"
Cohesion: 0.29
Nodes (5): HTMLTableRepetitionDetector, Specialized detector for HTML table row repetitions.          This detector is s, Initialize the HTML table repetition detector.                  Args:, Detect repeated HTML table rows in the text., Determine repetition type based on positions.

### Community 20 - "OCR No-Flash Chat Messages"
Cohesion: 0.33
Nodes (3): Encode image file to base64 string.                  Args:             image_pat, Create chat messages with image and prompt.                  Args:             i, Process a single data item through the VLLM API.                  Args:

### Community 21 - "Demo Display Utils"
Cohesion: 0.5
Nodes (4): is_valid_image_path(), Reads an image and resizes it while maintaining aspect ratio.      Args:, Checks if the image path is valid.      Args:         image_path: The path to th, read_image()

### Community 22 - "Detection Result Model"
Cohesion: 0.5
Nodes (3): DetectionResult, Result of repetition detection., Generate a human-readable summary of the detection results.

## Knowledge Gaps
- **151 isolated node(s):** `Loads a local CSS file and injects it into the Streamlit app.`, `Test all imports to ensure they work correctly`, `Classifies a document image into 'Invoice', 'Quotation', or 'Other'.          Ar`, `Converts the first page of a PDF file to a PIL Image object.          Args:`, `Invoice Data Extractor Integration Module  This module provides easy integration` (+146 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OCR` connect `Hybrid Search Engine` to `DotsOCR Model Inference`, `Typhoon OCR Pipeline`, `Invoice OCR No-Flash`, `OCR No-Flash Chat Messages`?**
  _High betweenness centrality (0.367) - this node is a cross-community bridge._
- **Why does `PILimage_to_base64()` connect `DotsOCR Model Inference` to `Format Transformer Utils`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._
- **Why does `ocr_processing_page()` connect `OCR Page Extraction` to `Hybrid Search Engine`, `Document Classification`, `Core Extraction & Client`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `OCR` (e.g. with `Qwen3VLLMClient` and `SchematronClient`) actually correct?**
  _`OCR` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `HybridSearch` (e.g. with `Properties` and `ItemInvoice`) actually correct?**
  _`HybridSearch` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `StampDetection` (e.g. with `Properties` and `ItemInvoice`) actually correct?**
  _`StampDetection` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `PurchaseOrder` (e.g. with `Properties` and `ItemInvoice`) actually correct?**
  _`PurchaseOrder` has 21 INFERRED edges - model-reasoned connections that need verification._