import logging
import time
from typing import Optional, Tuple
from pathlib import Path
import pyautogui
import pygetwindow as gw

# Configure module logger
logger = logging.getLogger(__name__)

# Configure PyAutoGUI
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small delay between actions


class AutomationController:
    """
    Desktop automation controller for interacting with applications.
    
    Provides high-level automation functions with error handling and validation.
    """
    
    def __init__(
        self,
        move_duration: float = 0.5,
        type_interval: float = 0.05,
        window_timeout: int = 5
    ):
        """
        Initialize automation controller.
        
        Args:
            move_duration: Mouse movement duration in seconds
            type_interval: Typing interval between keystrokes
            window_timeout: Timeout for window detection in seconds
        """
        self.move_duration = move_duration
        self.type_interval = type_interval
        self.window_timeout = window_timeout
        
        logger.info("AutomationController initialized")
        logger.info(f"Move duration: {move_duration}s, Type interval: {type_interval}s")
    
    def launch_notepad(
        self,
        icon_x: int,
        icon_y: int
    ) -> bool:
        """
        Launch Notepad by double-clicking detected icon.
        
        Args:
            icon_x: Icon center X coordinate
            icon_y: Icon center Y coordinate
            
        Returns:
            bool: True if Notepad launched successfully, False otherwise
        """
        logger.info(f"Launching Notepad at ({icon_x}, {icon_y})...")
        
        try:
            # Move mouse to icon
            logger.debug(f"Moving mouse to ({icon_x}, {icon_y})")
            pyautogui.moveTo(icon_x, icon_y, duration=self.move_duration)
            time.sleep(0.3)  # Wait for movement to complete
            
            # Double-click
            logger.debug("Double-clicking icon")
            pyautogui.doubleClick(interval=0.25)
            
            # Move mouse away to avoid obstructing view for next iteration
            screen_width, screen_height = self.get_screen_size()
            mid_x = screen_width // 2
            mid_y = screen_height // 2
            
            # Determine which quadrant and calculate new position
            new_x = icon_x
            new_y = icon_y
            
            if icon_x < mid_x:  # Left half
                new_x = icon_x + 480
            else:  # Right half
                new_x = icon_x - 480
            
            if icon_y < mid_y:  # Upper half
                new_y = icon_y + 270
            else:  # Lower half
                new_y = icon_y - 270
            
            logger.debug(f"Moving mouse away to ({new_x}, {new_y})")
            pyautogui.moveTo(new_x, new_y, duration=0.2)
            time.sleep(0.07) #experimental
            pyautogui.click()  # Click to clear any hover effects, new here
            
            # Wait for window to appear
            logger.debug("Waiting for Notepad window...")
            if self._wait_for_window("Notepad", timeout=self.window_timeout):
                logger.info("✓ Notepad launched successfully")
                return True
            else:
                logger.error("Notepad window did not appear")
                return False
                
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe triggered (mouse moved to corner)")
            return False
            
        except Exception as e:
            logger.error(f"Error launching Notepad: {e}", exc_info=True)
            return False
    
    def type_text(self, text: str, clear_first: bool = True) -> bool:
        """
        Type text into active window.
        
        Args:
            text: Text content to type
            clear_first: Whether to clear existing content first
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Typing {len(text)} characters...")
        
        try:
            # Ensure Notepad window is active
            if not self._ensure_notepad_active():
                logger.error("Notepad window not active")
                return False
            
            time.sleep(0.3)
            
            # Clear existing content
            if clear_first:
                logger.debug("Clearing existing content (Ctrl+A, Delete)")
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.2)
                pyautogui.press('delete')
                time.sleep(0.2)
            
            # Type content
            logger.debug("Typing content...")
            pyautogui.write(text, interval=self.type_interval)
            
            logger.info("✓ Text typed successfully")
            return True
            
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe triggered")
            return False
            
        except Exception as e:
            logger.error(f"Error typing text: {e}", exc_info=True)
            return False
    
    def save_file(
        self,
        filepath: Path,
        handle_overwrite: bool = True
    ) -> bool:
        """
        Save file using Save As dialog.
        
        Args:
            filepath: Full path where file should be saved
            handle_overwrite: Whether to handle overwrite confirmation
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        logger.info(f"Saving file: {filepath}")
        
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {filepath.parent}")
            
            # Ensure Notepad window is active
            if not self._ensure_notepad_active():
                logger.error("Notepad window not active")
                return False
            
            time.sleep(0.3)
            
            # Open Save As dialog (Ctrl+Shift+S or Ctrl+S)
            logger.debug("Opening Save As dialog (Ctrl+Shift+S)")
            pyautogui.hotkey('ctrl', 'shift', 's')
            time.sleep(1.0)  # Wait for dialog to fully load
            
            # Clear any existing text in filename field
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.2)
            
            # Type full file path with quotes (helps with spaces)
            filepath_str = str(filepath.absolute())
            logger.debug(f"Typing filepath: {filepath_str}")
            
            # Type path character by character for reliability
            for char in filepath_str:
                pyautogui.press(char)
                time.sleep(0.02)
            
            time.sleep(0.5)
            
            # Press Enter to save
            logger.debug("Pressing Enter to save")
            pyautogui.press('enter')
            time.sleep(1.0)  # Wait longer for file operation
            
            # Handle overwrite confirmation if needed
            if handle_overwrite:
                logger.debug("Checking for overwrite confirmation dialog")
                time.sleep(0.3)
                
                # Look for various possible confirmation window titles
                confirm_titles = ["Confirm Save As", "Save As", "Replace", "Notepad"]
                confirmation_found = False
                
                for title in confirm_titles:
                    confirm_windows = gw.getWindowsWithTitle(title)
                    if confirm_windows and len(confirm_windows) > 0:
                        confirmation_found = True
                        break
                
                if confirmation_found:
                    logger.debug("Overwrite confirmation detected, selecting Yes")
                    time.sleep(0.3)
                    # Press left arrow to ensure "Yes" is selected, then Enter
                    pyautogui.press('left')
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(0.5)
                else:
                    # Try pressing enter anyway in case dialog exists but title didn't match
                    pyautogui.press('enter')
                    time.sleep(0.3)
            
            # Verify file was saved (check multiple times)
            for attempt in range(3):
                time.sleep(0.5)
                if filepath.exists():
                    logger.info(f"✓ File saved successfully: {filepath}")
                    # Click to clear any dialogs or focus issues
                    time.sleep(0.2)
                    pyautogui.click()
                    logger.debug("Clicked after save")
                    return True
                logger.debug(f"Verification attempt {attempt + 1}/3: File not found yet")
            
            logger.warning(f"File not found after save attempt: {filepath}")
            logger.warning(f"Directory exists: {filepath.parent.exists()}")
            logger.warning(f"Directory contents: {list(filepath.parent.iterdir()) if filepath.parent.exists() else 'N/A'}")
            return False
                
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe triggered")
            return False
            
        except Exception as e:
            logger.error(f"Error saving file: {e}", exc_info=True)
            return False
    
    def close_notepad(self, save_changes: bool = False) -> bool:
        """
        Close Notepad window.
        
        Args:
            save_changes: Whether to save changes before closing
            
        Returns:
            bool: True if closed successfully, False otherwise
        """
        logger.info("Closing Notepad...")
        
        try:
            # Find Notepad windows
            notepad_windows = gw.getWindowsWithTitle("Notepad")
            
            if not notepad_windows:
                logger.warning("No Notepad windows found")
                return True  # Already closed
            
            # Activate first Notepad window
            notepad_window = notepad_windows[0]
            notepad_window.activate()
            time.sleep(0.3)
            
            # Close window (Alt+F4)
            logger.debug("Sending Alt+F4 to close window")
            pyautogui.hotkey('alt', 'F4')
            time.sleep(1.0)  # Increased wait for close/prompt dialog
            
            # Handle save changes prompt if appears
            # Check for actual save prompt dialog, not just any Notepad window
            prompt_detected = False
            try:
                # Look for the save changes dialog specifically
                all_windows = gw.getAllTitles()
                for title in all_windows:
                    if "Notepad" in title and ("want to save" in title.lower() or title == "Notepad"):
                        # Additional check: if there's still a Notepad window after Alt+F4, it's likely a prompt
                        notepad_windows_after = gw.getWindowsWithTitle("Notepad")
                        if notepad_windows_after:
                            prompt_detected = True
                            break
            except Exception as e:
                logger.debug(f"Error checking for save prompt: {e}")
            
            if prompt_detected:
                logger.debug("Save changes prompt detected")
                time.sleep(0.2)  # Small delay before responding to prompt
                if save_changes:
                    logger.debug("Pressing 'Y' to save changes")
                    pyautogui.press('y')

                    time.sleep(0.3)
                    pyautogui.click()
                    logger.debug("Clicked after accepting save prompt")
                else:
                    logger.debug("Pressing 'N' to discard changes")
                    pyautogui.press('n')
                    time.sleep(0.3)
                    pyautogui.click()
                    logger.debug("Clicked after dismissing save prompt")
                time.sleep(0.5)
            else:
                logger.debug("No save prompt detected, window closed cleanly")
            
            # Verify window closed
            time.sleep(0.3)
            remaining_windows = gw.getWindowsWithTitle("Notepad")
            if not remaining_windows:
                logger.info("✓ Notepad closed successfully")
                return True
            else:
                logger.warning("Notepad window still open")
                return False
                
        except Exception as e:
            logger.error(f"Error closing Notepad: {e}", exc_info=True)
            return False
    
    def cleanup_all_notepad_windows(self) -> int:
        """
        Close all open Notepad windows without saving.
        
        Returns:
            int: Number of windows closed
        """
        logger.info("Cleaning up all Notepad windows...")
        
        closed_count = 0
        max_attempts = 10  # Prevent infinite loop
        
        for attempt in range(max_attempts):
            notepad_windows = gw.getWindowsWithTitle("Notepad")
            
            if not notepad_windows:
                break
            
            try:
                window = notepad_windows[0]
                window.activate()
                time.sleep(0.2)
                
                pyautogui.hotkey('alt', 'F4')
                time.sleep(0.3)
                
                # Discard changes if prompted
                pyautogui.press('n')
                time.sleep(0.3)
                
                closed_count += 1
                
            except Exception as e:
                logger.error(f"Error closing window: {e}")
                break
        
        logger.info(f"Cleaned up {closed_count} Notepad windows")
        return closed_count
    
    def _wait_for_window(
        self,
        window_title: str,
        timeout: int = 5
    ) -> bool:
        """
        Wait for window with title to appear.
        
        Args:
            window_title: Partial window title to match
            timeout: Maximum wait time in seconds
            
        Returns:
            bool: True if window found, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                logger.debug(f"Window found: {window_title}")
                return True
            time.sleep(0.2)
        
        logger.warning(f"Window not found after {timeout}s: {window_title}")
        return False
    
    def _ensure_notepad_active(self) -> bool:
        """
        Ensure Notepad window is active/focused.
        
        Returns:
            bool: True if Notepad is active, False otherwise
        """
        try:
            notepad_windows = gw.getWindowsWithTitle("Notepad")
            
            if not notepad_windows:
                logger.error("No Notepad windows found")
                return False
            
            # Activate first window
            notepad_window = notepad_windows[0]
            notepad_window.activate()
            time.sleep(0.3)
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating Notepad: {e}")
            return False
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get screen resolution.
        
        Returns:
            Tuple[int, int]: Screen width and height
        """
        size = pyautogui.size()
        logger.debug(f"Screen size: {size}")
        return size
    
    def move_mouse_safe(self, x: int, y: int) -> bool:
        """
        Move mouse to coordinates with validation.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate coordinates
            screen_width, screen_height = self.get_screen_size()
            if not (0 <= x <= screen_width and 0 <= y <= screen_height):
                logger.error(f"Coordinates out of bounds: ({x}, {y})")
                return False
            
            pyautogui.moveTo(x, y, duration=self.move_duration)
            return True
            
        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
            return False
