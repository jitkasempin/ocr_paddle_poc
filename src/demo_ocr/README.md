# Invoice OCR System - Streamlit Multipage App

This is a Streamlit multipage application for OCR processing and invoice data viewing.

## Structure

```
src/demo-ocr/
├── main.py                           # Main application entry point
├── processing/
│   ├── __init__.py
│   ├── ocr_page.py                   # OCR processing page
│   ├── app.py                        # Original standalone OCR app
│   ├── fast_vision.py                # Vision model utilities
│   └── ocr.py                        # OCR model implementation
├── invoice_viewer/
│   ├── __init__.py
│   ├── invoice_viewer_page.py        # Invoice viewer page
│   └── invoice-streamlit-app.py      # Original standalone viewer app
└── README.md                         # This file
```

## Features

### OCR Processing Page
- Upload PDF files for OCR processing
- Extract text and structured data from invoices
- Save extracted data to Supabase database
- Interactive data editing before saving

### Invoice Viewer Page
- View all processed invoices from the database
- Detailed invoice information display
- Search and filter capabilities
- Export functionality

## How to Run

1. Navigate to the demo-ocr directory:
   ```bash
   cd src/demo-ocr
   ```

2. Run the Streamlit multipage app:
   ```bash
   streamlit run main.py
   ```

3. The app will open in your browser with two pages:
   - **OCR PROCESSING**: Upload and process PDF invoices
   - **INVOICE VIEWER**: View and manage processed invoices

## Navigation

Use the sidebar to navigate between the two main pages:
- Select "OCR PROCESSING" to upload and process new invoices
- Select "INVOICE VIEWER" to view existing invoice data

## Dependencies

Make sure you have all required dependencies installed:
- streamlit
- PyMuPDF (fitz)
- PIL (Pillow)
- pandas
- supabase
- pydantic
- plotly
- unsloth
- transformers
- langchain-ollama

## Configuration

The app uses Supabase for data storage. The connection details are configured in the page modules.

## Notes

- The original standalone apps (`app.py` and `invoice-streamlit-app.py`) are preserved for reference
- The multipage structure allows for better organization and navigation
- Each page maintains its own session state and functionality
- The OCR processing requires the typhoon-ocr-7b model and qwen3:4b model to be available
