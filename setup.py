"""
Setup and validation script for CV Desktop Automation.
Run this before first use to verify environment and dependencies.
"""

import sys
import os
from pathlib import Path
import subprocess


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step_num, text):
    """Print step number and description."""
    print(f"\n[{step_num}] {text}")


def print_success(text):
    """Print success message."""
    print(f"  ✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"  ✗ {text}")


def print_warning(text):
    """Print warning message."""
    print(f"  ⚠ {text}")


def check_python_version():
    """Check Python version is 3.8+."""
    print_step(1, "Checking Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version_str} detected")
        return True
    else:
        print_error(f"Python {version_str} detected (requires 3.8+)")
        return False


def check_pip():
    """Check pip is available."""
    print_step(2, "Checking pip installation...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print_success(f"pip is available: {result.stdout.strip()}")
        return True
    except Exception as e:
        print_error(f"pip not found: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print_step(3, "Checking Python dependencies...")
    
    required_packages = [
        ("cv2", "opencv-python"),
        ("pytesseract", "pytesseract"),
        ("PIL", "pillow"),
        ("pyautogui", "pyautogui"),
        ("pygetwindow", "pygetwindow"),
        ("requests", "requests"),
        ("numpy", "numpy"),
    ]
    
    missing = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print_success(f"{package_name} installed")
        except ImportError:
            print_warning(f"{package_name} NOT installed")
            missing.append(package_name)
    
    if missing:
        print_warning(f"\nMissing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    return True


def check_tesseract():
    """Check if Tesseract OCR is installed."""
    print_step(4, "Checking Tesseract OCR installation...")
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print_success("Tesseract OCR is installed and accessible")
        return True
    except Exception as e:
        print_error("Tesseract OCR not found or not in PATH")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  After installation, add to system PATH or configure in code")
        return False


def check_project_structure():
    """Check project directory structure."""
    print_step(5, "Checking project structure...")
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        "src",
        "assets",
    ]
    
    required_files = [
        "src/__init__.py",
        "src/main.py",
        "src/vision_core.py",
        "src/bot_controller.py",
        "src/data_provider.py",
        "src/utils.py",
        "requirements.txt",
        "README.md",
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print_success(f"Directory exists: {dir_name}/")
        else:
            print_error(f"Directory missing: {dir_name}/")
            all_good = False
    
    # Check files
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print_success(f"File exists: {file_name}")
        else:
            print_error(f"File missing: {file_name}")
            all_good = False
    
    return all_good


def check_template_image():
    """Check if template image exists."""
    print_step(6, "Checking template image...")
    
    project_root = Path(__file__).parent
    template_path = project_root / "assets" / "notepad_icon.png"
    
    if template_path.exists():
        print_success(f"Template image found: {template_path}")
        
        # Check file size
        size_kb = template_path.stat().st_size / 1024
        if 1 <= size_kb <= 50:
            print_success(f"Template size looks good: {size_kb:.1f} KB")
        else:
            print_warning(f"Template size unusual: {size_kb:.1f} KB")
            print("  Expected: 1-50 KB")
        
        return True
    else:
        print_warning("Template image NOT found")
        print("  Expected location: assets/notepad_icon.png")
        print("  See assets/TEMPLATE_GUIDE.md for instructions")
        print("  Note: System will use OCR-only mode without template")
        return False


def check_api_connectivity():
    """Check API connectivity."""
    print_step(7, "Testing API connectivity...")
    
    try:
        import requests
        response = requests.get(
            "https://jsonplaceholder.typicode.com/posts/1",
            timeout=5
        )
        
        if response.status_code == 200:
            print_success("API is accessible")
            return True
        else:
            print_error(f"API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"API connection failed: {e}")
        print("  Check internet connection")
        return False


def check_screen_resolution():
    """Check screen resolution."""
    print_step(8, "Checking screen resolution...")
    
    try:
        import pyautogui
        width, height = pyautogui.size()
        print_success(f"Screen resolution: {width}x{height}")
        
        if width != 1920 or height != 1080:
            print_warning("Screen resolution is not 1920x1080")
            print("  System will adapt, but best tested at 1920x1080")
        
        return True
    except Exception as e:
        print_error(f"Failed to detect screen resolution: {e}")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print_step("*", "Installing dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print_success("Dependencies installed successfully")
        return True
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def main():
    """Run setup and validation."""
    print_header("CV DESKTOP AUTOMATION - SETUP & VALIDATION")
    
    print("\nThis script will verify your environment is ready to run the application.")
    print("It will check dependencies, Tesseract OCR, project structure, and more.")
    
    # Run checks
    results = {
        "Python version": check_python_version(),
        "pip": check_pip(),
        "Dependencies": check_dependencies(),
        "Tesseract OCR": check_tesseract(),
        "Project structure": check_project_structure(),
        "Template image": check_template_image(),
        "API connectivity": check_api_connectivity(),
        "Screen resolution": check_screen_resolution(),
    }
    
    # Print summary
    print_header("SETUP SUMMARY")
    
    critical_checks = ["Python version", "pip", "Tesseract OCR", "Project structure"]
    optional_checks = ["Template image"]
    
    critical_passed = all(results[check] for check in critical_checks)
    all_passed = all(results.values())
    
    print("\nCritical checks:")
    for check in critical_checks:
        status = "✓ PASS" if results[check] else "✗ FAIL"
        print(f"  {status:8} {check}")
    
    print("\nOptional checks:")
    for check in optional_checks:
        status = "✓ PASS" if results[check] else "⚠ WARN" 
        print(f"  {status:8} {check}")
    
    print("\nOther checks:")
    for check, passed in results.items():
        if check not in critical_checks and check not in optional_checks:
            status = "✓ PASS" if passed else "⚠ WARN"
            print(f"  {status:8} {check}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if critical_passed:
        print("✓ Setup complete! You can run the application.")
        print("\nTo run:")
        print("  python -m src.main")
        print("\nor:")
        print("  cd src")
        print("  python main.py")
        
        if not results["Template image"]:
            print("\nNote: Template image not found - system will use OCR-only mode")
            print("For best results, create template image (see assets/TEMPLATE_GUIDE.md)")
    else:
        print("✗ Setup incomplete - please fix the issues above")
        
        if not results["Dependencies"]:
            print("\nTo install dependencies:")
            print("  pip install -r requirements.txt")
        
        if not results["Tesseract OCR"]:
            print("\nTo install Tesseract OCR:")
            print("  1. Download: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  2. Install and add to system PATH")
            print("  3. Verify with: tesseract --version")
    
    print("=" * 60)
    
    return 0 if critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
