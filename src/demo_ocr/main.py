import streamlit as st
import asyncio

# Page configuration - this should only be in the main file
st.set_page_config(
    page_title="Invoice OCR System",
    page_icon="📄",
    layout="wide"
)

def load_css(file_name):
    """Loads a local CSS file and injects it into the Streamlit app."""
    try:
        with open(file_name) as f:
            # Use st.markdown with unsafe_allow_html=True to inject the CSS
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file '{file_name}' not found.")



# Load the custom CSS file
load_css('src/demo_ocr/style.css')



# Custom CSS for light purple theme
# st.markdown("""
#     <style>
#     /* Button styling - light purple border and text with white background */
#     .stButton > button {
#         color: #9B7EBD !important;
#         background-color: #FFFFFF !important;
#         border: 2px solid #9B7EBD !important;
#         border-radius: 4px;
#         padding: 0.5rem 1rem;
#         font-weight: 500;
#         transition: all 0.3s ease;
#     }

#     .stButton > button:hover {
#         background-color: #F5F3F7 !important;
#         border-color: #7F5FA0 !important;
#         color: #7F5FA0 !important;
#     }

#     .stButton > button:active {
#         background-color: #E8E3F0 !important;
#         transform: scale(0.98);
#     }

#     /* Text input styling - light purple background with black text */
#     .stTextInput > div > div > input {
#         background-color: #F5F3F7 !important;
#         color: #000000 !important;
#         border: 1px solid #D4C5E0 !important;
#         border-radius: 4px;
#     }

#     .stTextInput > div > div > input:focus {
#         border-color: #9B7EBD !important;
#         box-shadow: 0 0 0 1px #9B7EBD !important;
#     }

#     /* Text area styling - light purple background with black text */
#     .stTextArea > div > div > textarea {
#         background-color: #F5F3F7 !important;
#         color: #000000 !important;
#         border: 1px solid #D4C5E0 !important;
#         border-radius: 4px;
#     }

#     .stTextArea > div > div > textarea:focus {
#         border-color: #9B7EBD !important;
#         box-shadow: 0 0 0 1px #9B7EBD !important;
#     }

#     /* Number input styling - light purple background with black text */
#     .stNumberInput > div > div > input {
#         background-color: #F5F3F7 !important;
#         color: #000000 !important;
#         border: 1px solid #D4C5E0 !important;
#         border-radius: 4px;
#     }

#     .stNumberInput > div > div > input:focus {
#         border-color: #9B7EBD !important;
#         box-shadow: 0 0 0 1px #9B7EBD !important;
#     }

#     /* Download button styling */
#     .stDownloadButton > button {
#         color: #9B7EBD !important;
#         background-color: #FFFFFF !important;
#         border: 2px solid #9B7EBD !important;
#         border-radius: 4px;
#         padding: 0.5rem 1rem;
#         font-weight: 500;
#     }

#     .stDownloadButton > button:hover {
#         background-color: #F5F3F7 !important;
#         border-color: #7F5FA0 !important;
#         color: #7F5FA0 !important;
#     }

#     /* Selectbox styling - light purple background with black text */
#     .stSelectbox > div > div > div {
#         background-color: #F5F3F7 !important;
#         color: #000000 !important;
#         border: 1px solid #D4C5E0 !important;
#     }

#     /* Multiselect styling */
#     .stMultiSelect > div > div > div {
#         background-color: #F5F3F7 !important;
#         color: #000000 !important;
#         border: 1px solid #D4C5E0 !important;
#     }

#     /* File uploader button styling */
#     .stFileUploader > div > button {
#         color: #9B7EBD !important;
#         background-color: #FFFFFF !important;
#         border: 2px solid #9B7EBD !important;
#         border-radius: 4px;
#     }

#     .stFileUploader > div > button:hover {
#         background-color: #F5F3F7 !important;
#         border-color: #7F5FA0 !important;
#     }

#     /* Sidebar styling for consistency */
#     section[data-testid="stSidebar"] {
#         background-color: #FAFAFA;
#     }

#     /* Header and title color accent */
#     h1, h2, h3 {
#         color: #1F1F1F !important;
#     }

#     /* Radio button styling - purple selection */
#     /* Use CSS accent-color for native radio buttons (most reliable method) */
#     input[type="radio"] {
#         accent-color: #9B7EBD !important;
#     }

#     /* Streamlit radio button - unchecked state */
#     [data-testid="stRadio"] label div[data-baseweb="radio"] > div {
#         border-color: #D4C5E0 !important;
#     }

#     /* Streamlit radio button - checked state (outer circle) */
#     [data-testid="stRadio"] label div[data-baseweb="radio"] > div[data-checked="true"] {
#         background-color: #9B7EBD !important;
#         border-color: #9B7EBD !important;
#     }

#     /* Streamlit radio button - inner circle when checked */
#     [data-testid="stRadio"] label div[data-baseweb="radio"] > div[data-checked="true"]::after {
#         background-color: #FFFFFF !important;
#     }

#     /* Alternative selector for checked state */
#     [data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] div[data-baseweb="radio"] > div {
#         background-color: #9B7EBD !important;
#         border-color: #9B7EBD !important;
#     }

#     /* Hover effect for radio buttons */
#     [data-testid="stRadio"] label:hover div[data-baseweb="radio"] > div {
#         border-color: #9B7EBD !important;
#     }

#     /* Focus state */
#     [data-testid="stRadio"] label div[data-baseweb="radio"]:focus-within > div {
#         box-shadow: 0 0 0 2px rgba(155, 126, 189, 0.3) !important;
#     }

#     /* SVG elements in radio buttons */
#     [data-testid="stRadio"] svg {
#         color: #9B7EBD !important;
#     }
#     </style>
    
#     <script>
#     // JavaScript to dynamically style radio buttons after page load
#     function styleRadioButtons() {
#         // Target all radio button elements
#         const radioButtons = document.querySelectorAll('[data-testid="stRadio"] input[type="radio"]');
#         radioButtons.forEach(radio => {
#             radio.style.accentColor = '#9B7EBD';
#         });
        
#         // Target radio button containers
#         const radioContainers = document.querySelectorAll('[data-testid="stRadio"] [role="radiogroup"] label');
#         radioContainers.forEach(container => {
#             const radioDiv = container.querySelector('[data-baseweb="radio"] > div');
#             if (radioDiv) {
#                 // Check if the radio is checked
#                 if (radioDiv.getAttribute('data-checked') === 'true') {
#                     radioDiv.style.backgroundColor = '#9B7EBD';
#                     radioDiv.style.borderColor = '#9B7EBD';
#                 }
                
#                 // Add event listener for changes
#                 const input = container.querySelector('input[type="radio"]');
#                 if (input) {
#                     input.addEventListener('change', function() {
#                         // Update all radio buttons in the group
#                         setTimeout(() => {
#                             const allRadios = document.querySelectorAll('[data-testid="stRadio"] [data-baseweb="radio"] > div');
#                             allRadios.forEach(rd => {
#                                 if (rd.getAttribute('data-checked') === 'true') {
#                                     rd.style.backgroundColor = '#9B7EBD';
#                                     rd.style.borderColor = '#9B7EBD';
#                                 } else {
#                                     rd.style.backgroundColor = '#FFFFFF';
#                                     rd.style.borderColor = '#D4C5E0';
#                                 }
#                             });
#                         }, 50);
#                     });
#                 }
#             }
#         });
#     }
    
#     // Run the styling function when the page loads
#     if (document.readyState === 'loading') {
#         document.addEventListener('DOMContentLoaded', styleRadioButtons);
#     } else {
#         styleRadioButtons();
#     }
    
#     // Re-run styling when Streamlit reruns
#     window.addEventListener('load', styleRadioButtons);
    
#     // Use MutationObserver to watch for dynamically added radio buttons
#     const observer = new MutationObserver(function(mutations) {
#         mutations.forEach(function(mutation) {
#             if (mutation.addedNodes.length) {
#                 styleRadioButtons();
#             }
#         });
#     });
    
#     // Start observing the document
#     if (document.body) {
#         observer.observe(document.body, { childList: true, subtree: true });
#     }
#     </script>
#     """, unsafe_allow_html=True)

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
