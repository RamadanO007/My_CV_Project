# CV Desktop Automation Project

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Python automation system using **computer vision** and **OCR** to locate and interact with desktop icons on Windows. Reliably finds target icons (e.g., Notepad) regardless of desktop position or similar icons nearby.

## 🎯 Key Features

- **🔍 Smart Detection Strategy**
  - Primary: Multi-scale template matching (0.7x - 3.0x) with OpenCV
  - Multiple instance detection (finds all matching icons simultaneously)
  - **Intelligent OCR fallback** triggers when:
    - Template matching fails completely
    - Multiple similar icons detected (e.g., Notepad vs Notepad++)
    - High confidence (≥0.91) with multiple candidates present
    - Ambiguous matches within 0.1 confidence of each other
  - OCR verifies by matching text labels to ensure correct icon selection

- **🖼️ Image Processing Pipeline**
  - Bilateral filtering + CLAHE + Otsu's thresholding
  - Word boundary checking for exact text matches
  - 90% similarity weight for text matching vs 10% OCR confidence

- **🤖 Robust Desktop Automation**
  - Intelligent mouse positioning (moves away after click to prevent obstruction)
  - Adaptive cursor placement based on screen quadrant
  - Click-to-clear hover effects for consistent detection
  - Window management and file dialog navigation

- **📊 Production-Ready**
  - Comprehensive logging with detection method tracking
  - Visual debugging with annotated screenshots
  - Retry mechanisms and error handling

## 📁 Project Structure

```
My_CV_Project/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main orchestration workflow
│   ├── vision_core.py        # Icon detection (template + OCR)
│   ├── bot_controller.py     # Desktop automation functions
│   ├── data_provider.py      # API integration
│   └── utils.py              # Utility functions
├── assets/
│   └── notepad_icon.png      # Template image (you create this)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Python packaging config
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- **Windows OS** (tested on Windows 10/11)
- **Python 3.8+**
- **Tesseract OCR** installed and in PATH

### 1. Install Tesseract OCR

Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

**Important:** Add Tesseract to your system PATH or set the path in your code:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 2. Clone or Download Project

```bash
cd f:\JobbApps\NOTEPAD_DETECTOR\My_CV_Project
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Template Image

**CRITICAL STEP:** You must create a template image of the Notepad icon:

1. Take a screenshot of your desktop with the Notepad icon visible
2. Open the screenshot in an image editor (Paint, Photoshop, etc.)
3. Crop **only** the Notepad icon (approximately 64x64 pixels)
4. Save as `assets/notepad_icon.png`

**Tips for best results:**
- Include only the icon, not the label text
- Use a clean crop with minimal background
- Save in PNG format
- Icon should be at normal size (not zoomed in/out)

### 5. Run the Application

```bash
python -m src.main
```

Or:

```bash
cd src
python main.py
```

## 📖 How It Works

### Detection Pipeline

1. **Screenshot Capture**: Full desktop screenshot
2. **Template Matching** (Primary):
   - Multi-scale search (0.7x to 3.0x) using normalized cross-correlation
   - Detects **all instances** above confidence threshold (0.85)
   - Non-maximum suppression to avoid duplicates
   - Cross-scale deduplication (merges same icon at different scales)

3. **OCR Verification** (Triggered When):
   - **Scenario 1**: Template matching completely fails → Full OCR fallback
   - **Scenario 2**: Multiple icons detected with high confidence (≥0.91) → OCR disambiguates
   - **Scenario 3**: Ambiguous matches (within 0.1 confidence) → OCR verifies text labels
   - Uses Tesseract PSM 11 (sparse text mode) with preprocessing
   - Matches text labels with word boundary checking (90% text similarity + 10% OCR confidence)
   - Returns icon with exact text match (e.g., "Notepad" vs "Notepad++")

4. **Smart Click Handling**:
   - Double-clicks icon
   - **Moves mouse away** based on screen quadrant (50-150px offset)
   - Single click at new position to clear hover effects
   - Prevents cursor from obstructing subsequent detections

### Automation Workflow

For each API post:
1. Detect icon → 2. Launch app → 3. Wait for window → 4. Type content → 5. Save file → 6. Close app

## 🎨 Output

The application creates the following outputs:

```
Desktop/cv-project/tjm/
├── post_1.txt                    # API post content files
├── post_2.txt
├── ...
└── detection_screenshots/
    ├── detection_template_xxx.png  # Successful detections
    ├── detection_ocr_xxx.png
    └── candidates/
        ├── candidate_1_notepad.png # OCR candidates (debug)
        └── ...
```

### Sample Post File Content

```
Title:sunt cere repellat ident caecati cepturi  rehenderit
a uscipit
uscipit usandae uuntur rehenderit rum m t rem veniet chitecto
n
```

## 🔧 Configuration

Key parameters in `main.py`:

```python
template_confidence = 0.85      # Template matching threshold
ocr_confidence = 0.6            # OCR detection threshold
move_duration = 0.5             # Mouse movement speed
max_retries = 3                 # Detection retry attempts
NUM_POSTS = 10                  # Posts to process
```

## 📊 Logging

Comprehensive logging with detection method tracking:

```
INFO - High confidence (0.98) with 2 candidates detected, performing OCR verification...
INFO - ✓ OCR verification successful: Selected 'Notepad' - DetectionResult(center=(1456, 234), confidence=0.92, method=template+ocr)
INFO - ✓ Notepad launched successfully
```

- **Console**: INFO level with progress
- **File**: `automation.log` with DEBUG details
- **Screenshots**: Visual debugging in `detection_screenshots/`

## 🧪 Testing

Test with different icon positions:
1. **Multiple similar icons**: Place both Notepad and Notepad++ on desktop → System uses OCR to select correct one
2. **Various positions**: Top-left, center, bottom-right → Detects in all locations
3. **High confidence scenarios**: Single icon → Fast template matching without OCR overhead

## ⚠️ Troubleshooting

### Template Matching Fails
- **Cause**: Template image doesn't match desktop icon
- **Solution**: Recreate template image from your actual desktop icon
- **Tip**: Check `detection_screenshots/` for visual debugging

### OCR Fallback Not Working
- **Cause**: Tesseract not installed or not in PATH
- **Solution**: Install Tesseract and verify with `tesseract --version`
- **Alternative**: Set explicit path in code

### PyAutoGUI Failsafe Triggered
- **Cause**: Mouse moved to screen corner (safety feature)
- **Solution**: Keep mouse away from corners during execution
- **Disable**: Set `pyautogui.FAILSAFE = False` (not recommended)

### Window Detection Timeout
- **Cause**: Notepad takes too long to launch
- **Solution**: Increase `window_timeout` in configuration
- **Check**: Verify icon double-click works manually

### File Save Fails
- **Cause**: Insufficient permissions or path too long
- **Solution**: Use shorter output path or different location
- **Check**: Verify directory exists and is writable

## 🔬 Technical Details

### Template Matching
- **Algorithm**: Normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`)
- **Multi-scale**: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
- **Multiple instance detection**: Finds all matches above threshold using `np.where()`
- **Deduplication**: Groups nearby detections (20% template size) and across scales (50px)

### OCR Pipeline
```
Grayscale → Bilateral filter (9,75,75) → CLAHE (2.0, 8x8) → Otsu's threshold → PIL Image → Tesseract PSM 11
```

### Text Matching
- **Exact match**: 1.0
- **Starts with + word boundary**: 0.6-0.8
- **Starts with + no boundary** (e.g., "Notepad++"): 0.3-0.4
- **Contains (word boundary)**: 0.6
- **Score**: 90% text similarity + 10% OCR confidence

## 🚀 Advanced Usage

### Detect Different Icons
```python
detector = IconDetector(
    template_path=Path("assets/chrome_icon.png"),
    target_name="chrome",
    template_confidence=0.85
)
```

### Adjust Detection Sensitivity
```python
# More lenient
detector = IconDetector(template_confidence=0.75, max_retries=5)

# Stricter
detector = IconDetector(template_confidence=0.90, ocr_confidence=0.7)
```

## 📚 Dependencies

Core libraries:
- **opencv-python** (4.8+): Computer vision and image processing
- **pytesseract** (0.3.10+): OCR text recognition
- **pyautogui** (0.9.54+): Mouse and keyboard automation
- **pygetwindow** (0.0.9+): Window management
- **pillow** (9.0+): Image handling
- **requests** (2.31+): HTTP API calls
- **numpy** (1.24+): Numerical operations

Optional:
- **tqdm**: Progress bars
- **pyyaml**: Configuration files

## 🤝 Contributing

Improvement suggestions:
- Multi-resolution support for different DPI settings
- Machine learning detection (YOLO/CNN)
- Cross-platform support (macOS/Linux)
- GUI configuration interface

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **Tesseract** - OCR engine by Google
- **PyAutoGUI** - GUI automation
- **JSONPlaceholder** - Free API for testing

## 📞 Support

For issues:
1. Check troubleshooting section above
2. Review `automation.log`
3. Inspect debug screenshots in `detection_screenshots/`
4. Verify template matches your desktop icon

## 🎓 Key Concepts

This project demonstrates:
- **Multi-instance detection** with OpenCV template matching
- **OCR ambiguity resolution** for similar icons
- **Adaptive cursor management** to prevent detection obstruction
- **Word boundary text matching** for precise label identification
- **Production-grade error handling** and logging
- **Modular architecture** with separation of concerns

---

**Built for robust desktop automation with intelligent icon detection**