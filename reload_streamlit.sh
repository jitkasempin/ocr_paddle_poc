#!/bin/bash
# Script to reload the Streamlit app with purple radio button styling

echo "🔄 Reloading Streamlit app with updated radio button styling..."
echo ""

echo "Stopping docker compose..."
docker compose down

echo "Waiting 3 seconds..."
sleep 3

echo "Rebuilding gradio3 service with updated styling..."
docker compose build gradio3

echo "Waiting 3 seconds..."
sleep 3

echo "Starting gradio3 service in detached mode..."
docker compose up -d gradio3

echo ""
echo "✅ Done! Streamlit app has been reloaded."
echo ""
echo "📋 Next steps:"
echo "1. Open your browser and navigate to your Streamlit app"
echo "2. Press Ctrl+Shift+R (or Cmd+Shift+R on Mac) to hard refresh"
echo "3. Go to the OCR Processing page"
echo "4. Check the radio buttons in the sidebar - they should now be PURPLE! 💜"
echo ""
echo "💡 Radio buttons are located at:"
echo "   - Document Type Selection (Invoice, Packing List, etc.)"
echo "   - OCR Model Selection (text_ocr, high_performance_ocr, etc.)"

