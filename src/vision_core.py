import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
import pytesseract
from PIL import Image

from .utils import (
    capture_screenshot,
    validate_coordinates,
    calculate_text_similarity,
    estimate_icon_position,
    annotate_detection,
    save_annotated_image
)

# Configure module logger
logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection results with metadata."""
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        confidence: float,
        method: str,
        scale: Optional[float] = None
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.method = method  # "template" or "ocr"
        self.scale = scale
        
        # Calculate center coordinates
        self.center_x = x + width // 2
        self.center_y = y + height // 2
    
    def __repr__(self):
        return (
            f"DetectionResult(center=({self.center_x}, {self.center_y}), "
            f"confidence={self.confidence:.2f}, method={self.method})"
        )


class IconDetector:
    """
    Robust icon detection using template matching and OCR fallback.
    
    Implements a two-stage detection strategy:
    1. Template matching with multi-scale search (primary)
    2. OCR-based text detection (fallback)
    """
    
    def __init__(
        self,
        template_path: Path,
        template_confidence: float = 0.85,
        ocr_confidence: float = 0.6,
        target_name: str = "Notepad",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize icon detector.
        
        Args:
            template_path: Path to template image file
            template_confidence: Minimum confidence for template matching
            ocr_confidence: Minimum confidence for OCR detection
            target_name: Name of icon to detect (for OCR)
            max_retries: Maximum detection retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.template_path = template_path
        self.template_confidence = template_confidence
        self.ocr_confidence = ocr_confidence
        self.target_name = target_name.lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Multi-scale search parameters
        # Optimized for 46x54 template - covers 32px to 138px icon sizes
        # Includes smaller scales for different DPI/icon size settings
        self.scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
        
        # Load template
        self.template = None
        self.template_gray = None
        self._load_template()
        
        logger.info(f"IconDetector initialized: target='{target_name}'")
        logger.info(f"Template confidence threshold: {template_confidence}")
        logger.info(f"OCR confidence threshold: {ocr_confidence}")
    
    def _load_template(self) -> bool:
        """Load and validate template image."""
        try:
            if not self.template_path.exists():
                logger.warning(f"Template not found: {self.template_path}")
                logger.warning("Template matching will be disabled")
                return False
            
            self.template = cv2.imread(str(self.template_path))
            
            if self.template is None:
                logger.error(f"Failed to load template: {self.template_path}")
                return False
            
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            
            logger.info(f"Template loaded: {self.template_path}")
            logger.info(f"Template size: {self.template.shape[:2]}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            return False
    
    def detect(
        self,
        screenshot: Optional[np.ndarray] = None,
        save_debug: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Optional[DetectionResult]:
        """
        Detect icon using template matching with OCR fallback.
        
        Args:
            screenshot: Screenshot to search in (captures new if None)
            save_debug: Whether to save debug images
            debug_dir: Directory for debug output
            
        Returns:
            DetectionResult: Detection result if found, None otherwise
        """
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Detection attempt {attempt}/{self.max_retries}")
            
            try:
                # Capture screenshot if not provided
                if screenshot is None:
                    screenshot = capture_screenshot()
                
                # Try template matching first
                if self.template is not None:
                    result, candidates = self._detect_by_template(screenshot, return_candidates=True)
                    
                    # Check for ambiguous matches OR high-confidence matches that need verification
                    if result and len(candidates) > 1:
                        top_confidence = candidates[0]['confidence']
                        
                        # ALWAYS verify if we have multiple candidates and top confidence is very high (0.91-1.0)
                        # This catches cases like Notepad vs Notepad++ with identical icons
                        force_ocr_check = top_confidence >= 0.91
                        
                        # Check if top candidates are within 0.1 confidence of each other
                        ambiguous_candidates = [
                            c for c in candidates 
                            if abs(c['confidence'] - top_confidence) < 0.1
                        ]
                        
                        if len(ambiguous_candidates) > 1 or (force_ocr_check and len(candidates) > 1):
                            if force_ocr_check:
                                logger.warning(
                                    f"High confidence ({top_confidence:.3f}) with {len(candidates)} candidates detected, "
                                    "performing OCR verification to ensure correct match..."
                                )
                                # Use all candidates for OCR verification when forcing check
                                ocr_candidates = candidates
                            else:
                                logger.warning(
                                    f"Found {len(ambiguous_candidates)} ambiguous matches "
                                    f"(confidence range: {ambiguous_candidates[-1]['confidence']:.3f} - {top_confidence:.3f}), "
                                    "using OCR verification..."
                                )
                                ocr_candidates = ambiguous_candidates
                            
                            # Use OCR to verify which one matches the target text
                            verified_result = self._verify_candidates_with_ocr(
                                screenshot, ocr_candidates, save_debug, debug_dir
                            )
                            if verified_result:
                                logger.info(f"✓ OCR verification successful: {verified_result}")
                                if save_debug and debug_dir:
                                    self._save_debug_image(screenshot, verified_result, debug_dir)
                                return verified_result
                            else:
                                logger.warning("OCR verification failed, using best template match")
                    
                    if result:
                        logger.info(f"✓ Template matching successful: {result}")
                        
                        if save_debug and debug_dir:
                            self._save_debug_image(screenshot, result, debug_dir)
                        
                        return result
                    else:
                        logger.warning("Template matching failed, trying OCR...")
                
                # Fallback to OCR
                result = self._detect_by_ocr(screenshot, save_debug, debug_dir)
                if result:
                    logger.info(f"✓ OCR detection successful: {result}")
                    
                    if save_debug and debug_dir:
                        self._save_debug_image(screenshot, result, debug_dir)
                    
                    return result
                
                logger.warning(f"Detection attempt {attempt} failed")
                
                # Retry with delay
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    screenshot = None  # Capture fresh screenshot
                
            except Exception as e:
                logger.error(f"Detection attempt {attempt} error: {e}", exc_info=True)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        logger.error("All detection attempts failed")
        return None
    
    def _detect_by_template(
        self,
        screenshot: np.ndarray,
        return_candidates: bool = False
    ) -> Tuple[Optional[DetectionResult], List[Dict]]:
        """
        Detect icon using multi-scale template matching.
        
        Args:
            screenshot: Screenshot to search in
            return_candidates: Whether to return all high-confidence candidates
            
        Returns:
            Tuple of (DetectionResult, candidates_list):
            - DetectionResult: Best match if confidence threshold met, None otherwise
            - candidates_list: List of all matches above threshold (if return_candidates=True)
        """
        try:
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            best_match = None
            best_confidence = 0.0
            best_scale = 1.0
            best_location = (0, 0)
            all_candidates = []  # Collect all high-confidence matches
            
            template_h, template_w = self.template_gray.shape
            
            # Multi-scale search
            for scale in self.scales:
                # Resize template
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)
                
                # Skip if template larger than screenshot
                if new_w > screenshot_gray.shape[1] or new_h > screenshot_gray.shape[0]:
                    continue
                
                resized_template = cv2.resize(
                    self.template_gray,
                    (new_w, new_h),
                    interpolation=cv2.INTER_CUBIC
                )
                
                # Perform template matching
                result = cv2.matchTemplate(
                    screenshot_gray,
                    resized_template,
                    cv2.TM_CCOEFF_NORMED
                )
                
                # Find ALL matches above threshold, not just the best one
                # This is crucial for detecting multiple similar icons (Notepad vs Notepad++)
                locations = np.where(result >= self.template_confidence)
                
                # Group nearby detections (non-maximum suppression)
                detections_at_scale = []
                for pt in zip(*locations[::-1]):  # Switch x and y
                    x, y = pt
                    conf = result[y, x]
                    
                    # Check if this detection is too close to an existing one (duplicate)
                    is_duplicate = False
                    for existing in detections_at_scale:
                        ex, ey = existing['location']
                        distance = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
                        # If within 20% of template size, consider it the same detection
                        if distance < min(new_w, new_h) * 0.2:
                            # Keep the one with higher confidence
                            if conf > existing['confidence']:
                                existing['confidence'] = conf
                                existing['location'] = (x, y)
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detections_at_scale.append({
                            'confidence': float(conf),
                            'scale': scale,
                            'location': (x, y),
                            'size': (new_w, new_h)
                        })
                
                # Add all detections from this scale to candidates
                all_candidates.extend(detections_at_scale)
                
                # Track absolute best for fallback
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # Track best match
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_scale = scale
                    best_location = max_loc
                    best_match = (new_w, new_h)
            
            # Deduplicate candidates across scales (same icon at different scales)
            # Group by spatial proximity
            deduplicated = []
            for candidate in all_candidates:
                cx = candidate['location'][0] + candidate['size'][0] // 2
                cy = candidate['location'][1] + candidate['size'][1] // 2
                
                # Check if this is near an existing detection
                is_duplicate = False
                for existing in deduplicated:
                    ex = existing['location'][0] + existing['size'][0] // 2
                    ey = existing['location'][1] + existing['size'][1] // 2
                    distance = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                    
                    # If centers are within 50px, consider it the same icon
                    if distance < 50:
                        # Keep the one with higher confidence
                        if candidate['confidence'] > existing['confidence']:
                            deduplicated.remove(existing)
                            deduplicated.append(candidate)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    deduplicated.append(candidate)
            
            # Sort deduplicated candidates by confidence (highest first)
            deduplicated.sort(key=lambda c: c['confidence'], reverse=True)
            
            logger.debug(f"Found {len(deduplicated)} unique icon candidates after deduplication")
            
            # Use deduplicated list
            all_candidates = deduplicated
            
            # Check confidence threshold
            if best_confidence >= self.template_confidence:
                x, y = best_location
                width, height = best_match
                
                # Validate coordinates
                if validate_coordinates(x + width // 2, y + height // 2):
                    result = DetectionResult(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        confidence=best_confidence,
                        method="template",
                        scale=best_scale
                    )
                    return (result, all_candidates) if return_candidates else (result, [])
            
            logger.debug(
                f"Template matching: best_confidence={best_confidence:.3f} "
                f"(threshold={self.template_confidence})"
            )
            return (None, []) if return_candidates else (None, [])
            
        except Exception as e:
            logger.error(f"Template matching error: {e}")
            return (None, [])
    
    def _verify_candidates_with_ocr(
        self,
        screenshot: np.ndarray,
        candidates: List[Dict],
        save_debug: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Optional[DetectionResult]:
        """
        Verify ambiguous template matches using OCR to find exact text match.
        
        When multiple template matches have similar confidence scores,
        use OCR to check which one actually has the target text label.
        
        Args:
            screenshot: Screenshot to search in
            candidates: List of candidate matches from template matching
            save_debug: Whether to save debug images
            debug_dir: Directory for debug output
            
        Returns:
            DetectionResult: Candidate with matching text label, None if no match
        """
        try:
            # Preprocess image for OCR
            preprocessed = self._preprocess_for_ocr(screenshot)
            
            # Convert to PIL Image for better Tesseract compatibility
            pil_image = Image.fromarray(preprocessed)
            
            # Run Tesseract OCR with PSM 11 (sparse text - best for desktop icons)
            ocr_data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config='--psm 11'
            )
            
            # Build list of text detections with positions
            text_detections = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                
                # Skip empty text
                if not text:
                    continue
                
                # Get OCR confidence (skip if invalid)
                conf = float(ocr_data['conf'][i])
                if conf < 0:
                    continue
                
                # Calculate text similarity to target
                similarity = calculate_text_similarity(text, self.target_name)
                
                # Only keep matches with good similarity
                if similarity >= 0.6:  # High similarity threshold for verification
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Text label center
                    label_center_x = x + w // 2
                    label_center_y = y + h // 2
                    
                    text_detections.append({
                        'text': text,
                        'x': label_center_x,
                        'y': label_center_y,
                        'similarity': similarity,
                        'ocr_conf': conf
                    })
            
            if not text_detections:
                logger.warning("No matching text labels found by OCR")
                return None
            
            logger.debug(f"Found {len(text_detections)} text labels matching target")
            
            # Match each candidate with nearest text label
            best_match = None
            best_combined_score = 0.0
            
            for candidate in candidates:
                x, y = candidate['location']
                w, h = candidate['size']
                template_center_x = x + w // 2
                template_center_y = y + h // 2
                
                # Find closest text label
                min_distance = float('inf')
                closest_text = None
                
                for text_det in text_detections:
                    # Calculate distance between template match and text label
                    # Text is typically 40-100px below icon
                    dx = text_det['x'] - template_center_x
                    dy = text_det['y'] - template_center_y
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_text = text_det
                
                # Score this candidate: template confidence + text match - distance penalty
                if closest_text and min_distance < 200:  # Must be reasonably close
                    # Normalize distance (closer is better)
                    distance_score = max(0, 1 - (min_distance / 200))
                    
                    # Combined score: 40% template, 40% text similarity, 20% proximity
                    combined_score = (
                        candidate['confidence'] * 0.4 +
                        closest_text['similarity'] * 0.4 +
                        distance_score * 0.2
                    )
                    
                    logger.debug(
                        f"Candidate at ({template_center_x}, {template_center_y}): "
                        f"template={candidate['confidence']:.3f}, "
                        f"text='{closest_text['text']}' (sim={closest_text['similarity']:.2f}), "
                        f"distance={min_distance:.0f}px, "
                        f"combined={combined_score:.3f}"
                    )
                    
                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_match = {
                            'candidate': candidate,
                            'text': closest_text,
                            'score': combined_score
                        }
            
            if best_match:
                candidate = best_match['candidate']
                x, y = candidate['location']
                w, h = candidate['size']
                
                logger.info(
                    f"OCR verification selected match at ({x + w//2}, {y + h//2}) "
                    f"with text '{best_match['text']['text']}' (score={best_match['score']:.3f})"
                )
                
                return DetectionResult(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=best_match['score'],
                    method="template+ocr",
                    scale=candidate['scale']
                )
            
            logger.warning("No candidate matched with nearby text label")
            return None
            
        except Exception as e:
            logger.error(f"OCR verification error: {e}", exc_info=True)
            return None
    
    def _detect_by_ocr(
        self,
        screenshot: np.ndarray,
        save_candidates: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Optional[DetectionResult]:
        """
        Detect icon by finding text label using OCR.
        
        Args:
            screenshot: Screenshot to search in
            save_candidates: Whether to save candidate images
            debug_dir: Directory for candidate images
            
        Returns:
            DetectionResult: Best match if found, None otherwise
        """
        try:
            # Preprocess image for OCR
            preprocessed = self._preprocess_for_ocr(screenshot)
            
            # Convert to PIL Image for better Tesseract compatibility
            pil_image = Image.fromarray(preprocessed)
            
            # Run Tesseract OCR with PSM 11 (sparse text - best for desktop icons)
            ocr_data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config='--psm 11'
            )
            
            # Find candidate matches
            candidates = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                
                # Skip empty text
                if not text:
                    continue
                
                # Get OCR confidence (skip if invalid)
                conf = float(ocr_data['conf'][i])
                if conf < 0:
                    continue
                
                # Calculate text similarity
                similarity = calculate_text_similarity(text, self.target_name)
                
                # Only process matches with some similarity
                if similarity > 0:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Estimate icon position above text label
                    # Desktop icons typically have labels 40-80px below icon center
                    label_center_x = x + w // 2
                    label_center_y = y + h // 2
                    
                    # Calculate offset: use max of 3x label height or 60px default
                    icon_offset_y = max(h * 3, 60) // 2
                    icon_x = label_center_x
                    icon_y = label_center_y - icon_offset_y
                    
                    # Combined score: 90% similarity, 10% OCR confidence
                    # (text matching is more important than OCR confidence)
                    combined_score = (similarity * 0.9) + (conf / 100.0 * 0.1)
                    
                    candidates.append({
                        'text': text,
                        'icon_x': icon_x,
                        'icon_y': icon_y,
                        'text_x': x,
                        'text_y': y,
                        'text_w': w,
                        'text_h': h,
                        'similarity': similarity,
                        'ocr_conf': conf,
                        'combined_score': combined_score
                    })
            
            if not candidates:
                logger.warning("No OCR candidates found")
                return None
            
            # Sort by combined score
            candidates.sort(key=lambda c: c['combined_score'], reverse=True)
            
            logger.debug(f"Found {len(candidates)} OCR candidates")
            
            # Save debug images
            if save_candidates and debug_dir:
                self._save_candidate_images(screenshot, candidates, debug_dir)
            
            # Return best candidate if meets threshold
            best = candidates[0]
            
            if best['combined_score'] >= self.ocr_confidence:
                # Validate coordinates
                if validate_coordinates(best['icon_x'], best['icon_y']):
                    # Use typical icon size
                    icon_size = 64
                    
                    return DetectionResult(
                        x=best['icon_x'] - icon_size // 2,
                        y=best['icon_y'] - icon_size // 2,
                        width=icon_size,
                        height=icon_size,
                        confidence=best['combined_score'],
                        method="ocr"
                    )
            
            logger.warning(
                f"Best OCR candidate score too low: {best['combined_score']:.3f} "
                f"(threshold={self.ocr_confidence})"
            )
            return None
            
        except Exception as e:
            logger.error(f"OCR detection error: {e}", exc_info=True)
            return None
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to improve OCR accuracy.
        
        Pipeline:
        1. Grayscale conversion
        2. Bilateral filtering (noise reduction)
        3. CLAHE (contrast enhancement)
        4. Otsu's thresholding (binarization)
        
        Args:
            image: Input BGR image
            
        Returns:
            np.ndarray: Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter - preserve edges while reducing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # CLAHE - enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # Otsu's thresholding - binarization
        _, binary = cv2.threshold(
            contrast,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary
    
    def _save_debug_image(
        self,
        screenshot: np.ndarray,
        result: DetectionResult,
        debug_dir: Path
    ) -> None:
        """Save annotated detection result image."""
        try:
            annotated = annotate_detection(
                screenshot,
                result.x,
                result.y,
                result.width,
                result.height,
                result.method.upper(),
                result.confidence
            )
            
            filename = f"detection_{result.method}_{int(time.time())}.png"
            save_annotated_image(annotated, debug_dir, filename)
            
        except Exception as e:
            logger.error(f"Failed to save debug image: {e}")
    
    def _save_candidate_images(
        self,
        screenshot: np.ndarray,
        candidates: List[Dict],
        debug_dir: Path
    ) -> None:
        """Save annotated images for OCR candidates."""
        try:
            candidates_dir = debug_dir / "candidates"
            candidates_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, candidate in enumerate(candidates[:5], 1):  # Top 5
                annotated = screenshot.copy()
                
                # Draw text bounding box
                cv2.rectangle(
                    annotated,
                    (candidate['text_x'], candidate['text_y']),
                    (candidate['text_x'] + candidate['text_w'],
                     candidate['text_y'] + candidate['text_h']),
                    (255, 0, 0),
                    2
                )
                
                # Draw estimated icon position
                cv2.drawMarker(
                    annotated,
                    (candidate['icon_x'], candidate['icon_y']),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    30,
                    3
                )
                
                # Add label
                label = (
                    f"#{idx}: {candidate['text']} "
                    f"(sim={candidate['similarity']:.2f}, "
                    f"conf={candidate['ocr_conf']:.0f})"
                )
                cv2.putText(
                    annotated,
                    label,
                    (candidate['text_x'], candidate['text_y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                
                filename = f"candidate_{idx}_{candidate['text']}.png"
                save_annotated_image(annotated, candidates_dir, filename)
                
        except Exception as e:
            logger.error(f"Failed to save candidate images: {e}")
