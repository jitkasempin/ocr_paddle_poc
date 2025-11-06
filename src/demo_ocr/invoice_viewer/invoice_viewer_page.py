import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta

# Check if Qdrant client is available
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    st.warning("Qdrant client not installed. Please install with: pip install qdrant-client")

# Custom CSS for better styling
def load_css():
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

# Initialize Qdrant connection
@st.cache_resource
def init_qdrant():
    """Initialize Qdrant client with credentials."""
    if not QDRANT_AVAILABLE:
        return None
    
    try:
        QDRANT_URL = "https://qdrant.ml.weaverbase.com:443"
        QDRANT_API_KEY = "db090fa8926fdd3260fb5cf364b9e0b517ea5ab8fdd45f778bb1473603fdfde9"
        
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        return None

# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_qdrant_data():
    """Fetch all points from Qdrant collection."""
    client = init_qdrant()
    
    if not client:
        return []
    
    try:
        COLLECTION_NAME = "ieat_production_embeddings"
        all_points = []
        offset = None
        limit = 100  # Fetch 100 points per request for efficiency
        
        # Use scroll to retrieve all points with pagination
        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors for display
            )
            
            points, next_offset = result
            
            if not points:
                break
            
            all_points.extend(points)
            
            # If next_offset is None, we've reached the end
            if next_offset is None:
                break
            
            offset = next_offset
        
        return all_points
        
    except Exception as e:
        st.error(f"Failed to fetch data from Qdrant: {str(e)}")
        return []

def process_qdrant_data(points):
    """Process Qdrant points into a pandas DataFrame."""
    processed_data = []
    
    for point in points:
        # Extract payload data
        payload = point.payload if hasattr(point, 'payload') else {}
        
        processed_row = {
            "id": point.id,
            "text": payload.get("text", ""),
            # Extract any additional metadata from payload
            **{k: v for k, v in payload.items() if k != "text"}
        }
        processed_data.append(processed_row)
    
    df = pd.DataFrame(processed_data)
    
    return df

def display_point_details(point_data):
    """Display detailed view of a single point."""
    st.subheader("Point Details")
    
    # Display ID
    st.write(f"**Point ID:** {point_data.get('id', 'N/A')}")
    
    # Display text
    st.write("**Page Content:**")
    text = point_data.get('text', 'No content available')
    st.text_area("Content", text, height=200, disabled=True)
    
    # Display other metadata if available
    metadata_keys = [k for k in point_data.keys() if k not in ['id', 'text']]
    if metadata_keys:
        st.write("**Additional Metadata:**")
        metadata_dict = {k: point_data[k] for k in metadata_keys}
        st.json(metadata_dict)

def invoice_viewer_page():
    """Main Qdrant data viewer page function"""
    # Load CSS
    load_css()
    
    st.header("📊 DELTA Approved Items")
    
    # Add collection info
    # st.info("**Collection:** similarity_search_poc")
    
    # Fetch and process data
    with st.spinner("Loading data from Qdrant..."):
        raw_points = fetch_qdrant_data()
        df = process_qdrant_data(raw_points)
    
    if df.empty:
        st.error("No data available in the Qdrant collection.")
        return
    
    # Display statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Points", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    # Main data table
    st.subheader("Items Table")
    
    if not df.empty:
        # Display configuration for better readability
        column_config = {
            "id": st.column_config.TextColumn("Point ID", width="small"),
            "text": st.column_config.TextColumn("Text", width="large")
        }
        
        # Add any additional columns to config
        # for col in df.columns:
            # if col not in ['id', 'text']:
                # column_config[col] = st.column_config.TextColumn(col.replace('_', ' ').title())
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_config,
            height=400
        )
        
        # Detailed view for selected point
        # st.subheader("Point Details")
        
        # Create a selectbox for choosing a point to view details
        # point_options = {f"Point ID: {row['id']}": idx for idx, row in df.iterrows()}
        
        # if point_options:
        #     selected_point_key = st.selectbox(
        #         "Select a point to view details:",
        #         options=list(point_options.keys())
        #     )
            
        #     if selected_point_key:
        #         selected_idx = point_options[selected_point_key]
        #         selected_point_data = df.iloc[selected_idx].to_dict()
                
        #         with st.expander("View Point Details", expanded=True):
        #             display_point_details(selected_point_data)
        
        # Download option
        st.subheader("Export Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="qdrant_points_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No points available in the collection.")
