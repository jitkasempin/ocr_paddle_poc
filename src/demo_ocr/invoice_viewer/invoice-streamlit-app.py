# Streamlit Invoice Data Viewer App
# This app connects to Supabase and displays invoice data from the public.invoice_data table

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Check if running on Streamlit Cloud or locally
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.warning("Supabase client not installed. Using sample data instead.")

# Page configuration
st.set_page_config(
    page_title="Invoice Data Viewer",
    page_icon="��",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .invoice-details {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .expandable-row {
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Supabase connection
@st.cache_resource
def init_supabase():
    """Initialize Supabase client with credentials from secrets."""
    if not SUPABASE_AVAILABLE:
        return None
    
    try:
        
        url = "https://ufffvetqzjuyfsxzlufj.supabase.co"
        key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVmZmZ2ZXRxemp1eWZzeHpsdWZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NDIwMTgsImV4cCI6MjA2MzExODAxOH0.amaDdf6kIoZgO-eiKxmB6qs4fEYIIcV-1U9gvyMDQuc"

        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None



# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_invoice_data():
    """Fetch invoice data from Supabase or return empty list."""
    supabase = init_supabase()
    
    if supabase:
        try:
            response = supabase.table('invoice_data').select('*').execute()
            return response.data
        except Exception as e:
            st.warning(f"Failed to fetch data from Supabase: {str(e)}.")
            return []
    else:
        return []

def process_invoice_data(invoices):
    """Process raw invoice data into a pandas DataFrame with calculated fields."""
    processed_data = []
    
    for invoice in invoices:
        # Calculate total amount from items_list
        if isinstance(invoice.get('items_list'), list):
            total_amount = sum(item.get("total", 0) for item in invoice["items_list"])
            items_count = len(invoice["items_list"])
        else:
            total_amount = 0
            items_count = 0
        
        processed_row = {
            "id": invoice.get("id"),
            "company": invoice.get("company", ""),
            "invoice_number": invoice.get("invoice_number", ""),
            "invoice_date": invoice.get("invoice_date", ""),
            "tax_id": invoice.get("tax_id", ""),
            "purchase_order_number": invoice.get("purchase_order_number", ""),
            "total_amount": total_amount,
            "items_count": items_count,
            "created_at": invoice.get("created_at", ""),
            "updated_at": invoice.get("updated_at", "")
        }
        processed_data.append(processed_row)
    
    df = pd.DataFrame(processed_data)
    
    # Convert date columns
    if not df.empty:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
    
    return df

def display_invoice_details(invoice):
    """Display detailed view of a single invoice."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Invoice Number:** {invoice.get('invoice_number', 'N/A')}")
        st.write(f"**Company:** {invoice.get('company', 'N/A')}")
        st.write(f"**Invoice Date:** {invoice.get('invoice_date', 'N/A')}")
        st.write(f"**Tax ID:** {invoice.get('tax_id', 'N/A')}")
    
    with col2:
        st.write(f"**Purchase Order:** {invoice.get('purchase_order_number', 'N/A')}")
        st.write(f"**Created:** {invoice.get('created_at', 'N/A')}")
        st.write(f"**Updated:** {invoice.get('updated_at', 'N/A')}")
    
    # Display items list
    if invoice.get('items_list') and isinstance(invoice['items_list'], list):
        st.subheader("Invoice Items")
        items_df = pd.DataFrame(invoice['items_list'])
        
        # Format the dataframe for better display
        if not items_df.empty:
            if 'unit_price' in items_df.columns:
                items_df['unit_price'] = items_df['unit_price'].apply(lambda x: f"${x:.2f}")
            if 'total' in items_df.columns:
                items_df['total'] = items_df['total'].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(items_df, use_container_width=True)
            
            # Calculate and display total
            total_amount = sum(item.get("total", 0) for item in invoice['items_list'])
            st.write(f"**Total Amount: ${total_amount:.2f}**")
        else:
            st.write("No items found for this invoice.")
    else:
        st.write("No items data available for this invoice.")

# Main application
def main():
    st.markdown('<h1 class="main-header">📋 Invoice Data Viewer</h1>', unsafe_allow_html=True)
    
    # Fetch and process data
    with st.spinner("Loading invoice data..."):
        raw_invoices = fetch_invoice_data()
        df = process_invoice_data(raw_invoices)
    
    if df.empty:
        st.error("No invoice data available.")
        return
    
    # Invoice table with expandable details
    st.subheader("Invoice Table")
    
    if not df.empty:
        # Display table
        display_df = df[['invoice_number', 'company', 'invoice_date', 'total_amount', 'items_count']].copy()
        display_df['total_amount'] = display_df['total_amount'].apply(lambda x: f"${x:,.2f}")
        display_df['invoice_date'] = display_df['invoice_date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "invoice_number": "Invoice #",
                "company": "Company",
                "invoice_date": "Date",
                "total_amount": "Total Amount",
                "items_count": "Items Count"
            }
        )
        
        # Detailed view for selected invoice
        st.subheader("Invoice Details")
        
        # Create a selectbox for choosing an invoice to view details
        invoice_options = {f"{row['invoice_number']} - {row['company']}": idx 
                         for idx, row in df.iterrows()}
        
        if invoice_options:
            selected_invoice_key = st.selectbox(
                "Select an invoice to view details:",
                options=list(invoice_options.keys())
            )
            
            if selected_invoice_key:
                selected_idx = invoice_options[selected_invoice_key]
                selected_invoice_data = raw_invoices[selected_idx]
                
                with st.expander("View Invoice Details", expanded=True):
                    display_invoice_details(selected_invoice_data)
    else:
        st.warning("No invoices available.")

# The sidebar and analytics tab have been removed.

if __name__ == "__main__":
    main()