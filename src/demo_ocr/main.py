import streamlit as st
import asyncio

# Page configuration - this should only be in the main file
st.set_page_config(
    page_title="Invoice OCR System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for light purple theme
st.markdown("""
    <style>
    /* Button styling - light purple border and text with white background */
    .stButton > button {
        color: #9B7EBD !important;
        background-color: #FFFFFF !important;
        border: 2px solid #9B7EBD !important;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #F5F3F7 !important;
        border-color: #7F5FA0 !important;
        color: #7F5FA0 !important;
    }

    .stButton > button:active {
        background-color: #E8E3F0 !important;
        transform: scale(0.98);
    }

    /* Text input styling - light purple background with black text */
    .stTextInput > div > div > input {
        background-color: #F5F3F7 !important;
        color: #000000 !important;
        border: 1px solid #D4C5E0 !important;
        border-radius: 4px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #9B7EBD !important;
        box-shadow: 0 0 0 1px #9B7EBD !important;
    }

    /* Text area styling - light purple background with black text */
    .stTextArea > div > div > textarea {
        background-color: #F5F3F7 !important;
        color: #000000 !important;
        border: 1px solid #D4C5E0 !important;
        border-radius: 4px;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #9B7EBD !important;
        box-shadow: 0 0 0 1px #9B7EBD !important;
    }

    /* Number input styling - light purple background with black text */
    .stNumberInput > div > div > input {
        background-color: #F5F3F7 !important;
        color: #000000 !important;
        border: 1px solid #D4C5E0 !important;
        border-radius: 4px;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #9B7EBD !important;
        box-shadow: 0 0 0 1px #9B7EBD !important;
    }

    /* Download button styling */
    .stDownloadButton > button {
        color: #9B7EBD !important;
        background-color: #FFFFFF !important;
        border: 2px solid #9B7EBD !important;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stDownloadButton > button:hover {
        background-color: #F5F3F7 !important;
        border-color: #7F5FA0 !important;
        color: #7F5FA0 !important;
    }

    /* Selectbox styling - light purple background with black text */
    .stSelectbox > div > div > div {
        background-color: #F5F3F7 !important;
        color: #000000 !important;
        border: 1px solid #D4C5E0 !important;
    }

    /* Multiselect styling */
    .stMultiSelect > div > div > div {
        background-color: #F5F3F7 !important;
        color: #000000 !important;
        border: 1px solid #D4C5E0 !important;
    }

    /* File uploader button styling */
    .stFileUploader > div > button {
        color: #9B7EBD !important;
        background-color: #FFFFFF !important;
        border: 2px solid #9B7EBD !important;
        border-radius: 4px;
    }

    .stFileUploader > div > button:hover {
        background-color: #F5F3F7 !important;
        border-color: #7F5FA0 !important;
    }

    /* Sidebar styling for consistency */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
    }

    /* Header and title color accent */
    h1, h2, h3 {
        color: #1F1F1F !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Import page functions
import sys
import os
sys.path.append(os.path.dirname(__file__))

from processing.ocr_page import ocr_processing_page
from invoice_viewer.invoice_viewer_page import invoice_viewer_page

# Main title
st.title("📄IEAT Pre-approve System")

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
