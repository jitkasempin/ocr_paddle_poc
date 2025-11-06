import streamlit as st
import asyncio

# Page configuration - this should only be in the main file
st.set_page_config(
    page_title="Invoice OCR System",
    page_icon="📄",
    layout="wide"
)

# Import page functions
import sys
import os
sys.path.append(os.path.dirname(__file__))

from processing.ocr_page import ocr_processing_page
from invoice_viewer.invoice_viewer_page import invoice_viewer_page

# Main title
st.title("📄OCR System")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["OCR PROCESSING", "ITEM VIEWER"]
)

# Page routing
if page == "OCR PROCESSING":
    # Run the async function using asyncio
    asyncio.run(ocr_processing_page())
elif page == "ITEM VIEWER":
    invoice_viewer_page()
