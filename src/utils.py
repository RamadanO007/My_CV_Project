import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import cv2
import numpy as np
import pyautogui
from PIL import Image

# Configure module logger
logger = logging.getLogger(__name__)


def capture_screenshot() -> np.ndarray:
    """
    Capture full screen screenshot and convert to OpenCV format.
    
    Returns:
        np.ndarray: Screenshot in BGR format for OpenCV processing
        
    Raises:
        Exception: If screenshot capture fails
    """
    try:
        logger.debug("Capturing screenshot...")
        screenshot = pyautogui.screenshot()
        
        # Convert PIL Image to OpenCV BGR format
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        logger.debug(f"Screenshot captured: {screenshot_bgr.shape}")
        return screenshot_bgr
        
    except Exception as e:
        logger.error(f"Failed to capture screenshot: {e}")
        raise


def annotate_detection(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding box and label on image to visualize detection.
    
    Args:
        image: OpenCV image to annotate
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Bounding box width
        height: Bounding box height
        label: Text label to display
        confidence: Detection confidence score
        color: BGR color tuple for drawing
        
    Returns:
        np.ndarray: Annotated image copy
    """
    try:
        # Create copy to avoid modifying original
        annotated = image.copy()
        
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3)
        
        # Prepare label text
        text = f"{label} ({confidence:.2f})"
        
        # Calculate text position (above rectangle)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x
        text_y = y - 10 if y > 30 else y + height + 25
        
        # Draw background rectangle for text
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Draw center crosshair
        center_x = x + width // 2
        center_y = y + height // 2
        cv2.drawMarker(
            annotated,
            (center_x, center_y),
            color,
            cv2.MARKER_CROSS,
            20,
            2
        )
        
        return annotated
        
    except Exception as e:
        logger.error(f"Failed to annotate image: {e}")
        return image


def save_annotated_image(
    image: np.ndarray,
    output_dir: Path,
    filename: str
) -> Optional[Path]:
    """
    Save annotated image to specified directory.
    
    Args:
        image: OpenCV image to save
        output_dir: Directory to save image
        filename: Name for the output file
        
    Returns:
        Path: Path to saved image, or None if failed
    """
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct full path
        filepath = output_dir / filename
        
        # Save image
        success = cv2.imwrite(str(filepath), image)
        
        if success:
            logger.info(f"Saved annotated image: {filepath}")
            return filepath
        else:
            logger.error(f"Failed to save image: {filepath}")
            return None
            
    except Exception as e:
        logger.error(f"Error saving image {filename}: {e}")
        return None


def ensure_directory(path: Path) -> bool:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        bool: True if directory exists/created, False otherwise
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def validate_coordinates(
    x: int,
    y: int,
    screen_width: int = 1920,
    screen_height: int = 1080
) -> bool:
    """
    Validate that coordinates are within screen bounds.
    
    Args:
        x: X coordinate
        y: Y coordinate
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        
    Returns:
        bool: True if coordinates are valid
    """
    valid = (0 <= x <= screen_width and 0 <= y <= screen_height)
    
    if not valid:
        logger.warning(
            f"Invalid coordinates: ({x}, {y}) outside "
            f"bounds (0-{screen_width}, 0-{screen_height})"
        )
    
    return valid


def calculate_text_similarity(detected_text: str, target: str = "Notepad") -> float:
    """
    Calculate similarity score between detected text and target.
    
    Implements fuzzy matching algorithm with word boundary enforcement:
    - Exact match: 1.0
    - Starts with target + word boundary: 0.6-0.8 (length-adjusted)
    - Starts with target + no boundary (e.g., "Notepad++"): 0.3-0.4
    - Contains target (word boundary): 0.6
    - Contains target (substring): 0.4
    - No match: 0.0
    
    Args:
        detected_text: Text detected by OCR
        target: Target text to match against
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not detected_text:
        return 0.0
    
    text_lower = detected_text.strip().lower()
    target_lower = target.lower()
    
    # Exact match
    if text_lower == target_lower:
        return 1.0
    
    # Starts with target - check word boundary
    if text_lower.startswith(target_lower):
        # Check if there are extra characters after target
        next_char_idx = len(target_lower)
        if next_char_idx < len(text_lower):
            next_char = text_lower[next_char_idx]
            # If next char is NOT a space/separator, it's likely a different app
            # (e.g., "Notepad++" should not match "Notepad")
            if next_char not in (' ', '\t', '\n'):
                # Weak match for extended names
                length_ratio = len(target_lower) / len(text_lower)
                return 0.3 + (0.1 * length_ratio)
        
        # Word boundary exists or target is at end - strong match
        length_ratio = len(target_lower) / len(text_lower)
        return 0.6 + (0.2 * length_ratio)
    
    # Contains target
    if target_lower in text_lower:
        # Check for word boundary
        import re
        pattern = r'\b' + re.escape(target_lower) + r'\b'
        if re.search(pattern, text_lower):
            return 0.6
        else:
            return 0.4
    
    return 0.0


def create_timestamp_string() -> str:
    """
    Create timestamp string for file naming.
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(
    log_file: str = "automation.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> None:
    """
    Set up dual logging to console and file.
    
    Args:
        log_file: Path to log file
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info("=" * 60)
    logger.info("Logging system initialized")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        np.ndarray: Loaded image in BGR format, or None if failed
    """
    try:
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        logger.debug(f"Loaded image: {image_path} {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def estimate_icon_position(
    text_x: int,
    text_y: int,
    text_width: int,
    text_height: int,
    icon_offset: int = 60
) -> Tuple[int, int]:
    """
    Estimate icon center position from text label position.
    
    Desktop icons typically appear above their text labels.
    
    Args:
        text_x: Text bounding box x coordinate
        text_y: Text bounding box y coordinate
        text_width: Text bounding box width
        text_height: Text bounding box height
        icon_offset: Typical vertical offset from text to icon center
        
    Returns:
        Tuple[int, int]: Estimated (icon_x, icon_y) coordinates
    """
    # Icon is centered horizontally with text
    icon_x = text_x + text_width // 2
    
    # Icon is above text, typically 60px
    icon_y = text_y - max(text_height * 3, icon_offset) // 2
    
    return icon_x, icon_y
