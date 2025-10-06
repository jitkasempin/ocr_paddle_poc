#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for the multipage app
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test all imports to ensure they work correctly"""
    print("Testing imports...")
    
    try:
        print("✓ Testing streamlit import...")
        import streamlit as st
        
        print("✓ Testing processing module import...")
        from processing.ocr_page import ocr_processing_page
        
        print("✓ Testing invoice_viewer module import...")
        from invoice_viewer.invoice_viewer_page import invoice_viewer_page
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 The multipage app should work correctly!")
        print("Run: streamlit run main.py")
    else:
        print("\n⚠️  There are import issues that need to be resolved.")
    
    sys.exit(0 if success else 1)
