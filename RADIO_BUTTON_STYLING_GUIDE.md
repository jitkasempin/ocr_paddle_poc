# Radio Button Styling Fix - Purple Theme

## What Was Fixed

The radio button styling in your Streamlit app has been updated to use **purple (#9B7EBD)** instead of the default red color.

## Changes Made

### 1. **Updated CSS Selectors** (lines 128-169 in `src/demo_ocr/main.py`)
   - Simplified and corrected CSS selectors to target Streamlit radio buttons
   - Used modern CSS `accent-color` property for native radio inputs
   - Added specific selectors for checked, unchecked, hover, and focus states

### 2. **Added JavaScript Dynamic Styling** (lines 172-238 in `src/demo_ocr/main.py`)
   - JavaScript function to directly manipulate radio button styles after page load
   - Event listeners to update styles when radio buttons are clicked
   - MutationObserver to watch for dynamically added radio buttons (important for Streamlit's dynamic content)

### 3. **Theme Configuration** (already present in `.streamlit/config.toml`)
   - Your `config.toml` already has `primaryColor = "#9B7EBD"` which should work with the new styling

## How to Test

### Method 1: Restart the Streamlit App

```bash
cd /home/jitkasem/ocr_paddle_poc
# If running in Docker
docker-compose down
docker-compose up -d

# If running directly
streamlit run src/demo_ocr/main.py
```

### Method 2: Force Browser Refresh
1. Open your Streamlit app in the browser
2. Press `Ctrl + Shift + R` (or `Cmd + Shift + R` on Mac) to hard refresh
3. This will clear the cache and reload all CSS/JavaScript

## Where Radio Buttons Are Used

Radio buttons are located in the **sidebar** on the OCR Processing page:
1. **Document Type Selection**: "Invoice", "Packing List", "Passport", "Certificate"
2. **OCR Model Selection**: "text_ocr", "high_performance_ocr", "legacy_ocr" (horizontal)

## Color Scheme

- **Purple (Selected)**: #9B7EBD
- **Light Purple (Hover/Background)**: #F5F3F7
- **Border (Unchecked)**: #D4C5E0
- **White (Inner Circle)**: #FFFFFF

## Troubleshooting

If the purple styling still doesn't appear:

1. **Clear Browser Cache**: 
   - Chrome: Settings → Privacy → Clear browsing data → Cached images and files
   - Firefox: Settings → Privacy → Clear Data → Cached Web Content

2. **Check Browser Console**:
   - Open Developer Tools (F12)
   - Look for any JavaScript errors in the Console tab
   - Check if the CSS is being applied in the Elements/Inspector tab

3. **Verify Streamlit Version**:
   ```bash
   pip show streamlit
   ```
   - The styling should work with Streamlit 1.x versions
   - If you have a very old or very new version, the DOM structure might be different

4. **Inspect the Radio Button Element**:
   - Right-click on a radio button → Inspect
   - Check if the `data-testid="stRadio"` attribute is present
   - Verify the nested structure matches the CSS selectors

## Why the Previous CSS Didn't Work

The original CSS selectors (lines 129-151) used paths like:
```css
.stRadio > div > label > div[data-baseweb="radio"] input + div
```

This was too specific and didn't match Streamlit's actual DOM structure. The new selectors are:
- More direct: `[data-testid="stRadio"] label div[data-baseweb="radio"] > div`
- Combined with JavaScript for dynamic updates
- Using modern `accent-color` CSS property as a fallback

## Additional Notes

- The JavaScript code uses a MutationObserver to detect when Streamlit adds new radio buttons
- Event listeners are added to handle state changes when users click radio buttons
- The styling will persist even when Streamlit reruns the page

