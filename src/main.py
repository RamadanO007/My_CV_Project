import logging
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
import pyautogui


from .utils import setup_logging, ensure_directory, create_timestamp_string
from .vision_core import IconDetector, DetectionResult
from .data_provider import DataProvider, test_api_connection
from .bot_controller import AutomationController

# Configure module logger
logger = logging.getLogger(__name__)

"""
.\scripts\run.bat
"""
class AutomationWorkflow:
    """
    Main workflow orchestrator for CV-based desktop automation.
    
    Coordinates icon detection, data fetching, and application automation.
    """
    
    def __init__(
        self,
        template_path: Path,
        output_dir: Path,
        debug_dir: Optional[Path] = None,
        num_posts: int = 10
    ):
        """
        Initialize automation workflow.
        
        Args:
            template_path: Path to icon template image
            output_dir: Directory for output text files
            debug_dir: Directory for debug screenshots
            num_posts: Number of posts to process
        """
        self.template_path = template_path
        self.output_dir = output_dir
        self.debug_dir = debug_dir or (output_dir / "detection_screenshots")
        self.num_posts = num_posts
        
        # Statistics
        self.stats = {
            'total_posts': 0,
            'successful': 0,
            'failed': 0,
            'detection_failures': 0,
            'automation_failures': 0
        }
        
        # Initialize components
        self.detector: Optional[IconDetector] = None
        self.data_provider: Optional[DataProvider] = None
        self.controller: Optional[AutomationController] = None
        
        logger.info("=" * 60)
        logger.info("AutomationWorkflow initialized")
        logger.info(f"Template: {template_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Debug directory: {self.debug_dir}")
        logger.info(f"Posts to process: {num_posts}")
        logger.info("=" * 60)
    
    def run(self) -> bool:
        """
        Execute complete automation workflow.
        
        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING AUTOMATION WORKFLOW")
        logger.info("=" * 60 + "\n")
        
        start_time = time.time()
        
        try:
            # Step 0: Minimize all windows to show desktop
            logger.info("Minimizing all windows to show desktop...")
            pyautogui.hotkey('win', 'd')
            time.sleep(2)  # Wait for desktop to appear
            logger.info("✓ Desktop visible\n")
            
            # Step 1: Initialize components
            if not self._initialize_components():
                logger.error("Component initialization failed")
                return False
            
            # Step 2: Ensure output directories exist
            if not self._setup_directories():
                logger.error("Directory setup failed")
                return False
            
            # Step 3: Test API connection
            if not test_api_connection():
                logger.error("API connection test failed")
                return False
            
            # Step 4: Fetch posts from API
            posts = self._fetch_posts()
            if not posts:
                logger.error("No posts fetched from API")
                return False
            
            # Step 5: Process each post
            logger.info(f"\nProcessing {len(posts)} posts...\n")
            for idx, post in enumerate(posts, 1):
                logger.info(f"{'=' * 60}")
                logger.info(f"PROCESSING POST {idx}/{len(posts)}: {post}")
                logger.info(f"{'=' * 60}")
                
                success = self._process_post(post, idx)
                
                if success:
                    self.stats['successful'] += 1
                    logger.info(f"✓ Post {idx} completed successfully\n")
                else:
                    self.stats['failed'] += 1
                    logger.error(f"✗ Post {idx} failed\n")
                
                # Small delay between posts
                if idx < len(posts):
                    time.sleep(2)
            
            # Step 6: Cleanup
            self._cleanup()
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Print summary
            self._print_summary(duration)
            
            return self.stats['successful'] > 0
            
        except KeyboardInterrupt:
            logger.warning("\n\nWorkflow interrupted by user (Ctrl+C)")
            self._cleanup()
            return False
            
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            self._cleanup()
            return False
    
    def _initialize_components(self) -> bool:
        """Initialize detector, data provider, and controller."""
        logger.info("Initializing components...")
        
        try:
            # Initialize icon detector
            self.detector = IconDetector(
                template_path=self.template_path,
                template_confidence=0.85,
                ocr_confidence=0.65,
                target_name="notepad",
                max_retries=3
            )
            
            # Initialize data provider
            self.data_provider = DataProvider(timeout=10)
            
            # Initialize automation controller
            self.controller = AutomationController(
                move_duration=0.5,
                type_interval=0.01,
                window_timeout=5
            )
            
            logger.info("✓ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization error: {e}", exc_info=True)
            return False
    
    def _setup_directories(self) -> bool:
        """Ensure output directories exist."""
        logger.info("Setting up directories...")
        
        try:
            ensure_directory(self.output_dir)
            ensure_directory(self.debug_dir)
            ensure_directory(self.debug_dir / "candidates")
            
            logger.info("✓ Directories ready")
            return True
            
        except Exception as e:
            logger.error(f"Directory setup error: {e}")
            return False
    
    def _fetch_posts(self):
        """Fetch posts from API."""
        logger.info("\nFetching posts from API...")
        
        try:
            posts = self.data_provider.fetch_posts(limit=self.num_posts)
            self.stats['total_posts'] = len(posts)
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch posts: {e}")
            return []
    
    def _process_post(self, post, post_number: int) -> bool:
        """
        Process single post: detect icon, launch Notepad, type content, save file.
        
        Args:
            post: Post object to process
            post_number: Sequential post number for file naming
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Step 1: Detect icon
            logger.info("Step 1: Detecting Notepad icon...")
            detection = self.detector.detect(
                save_debug=True,
                debug_dir=self.debug_dir
            )
            
            if not detection:
                logger.error("Icon detection failed")
                self.stats['detection_failures'] += 1
                return False
            
            logger.info(f"✓ Icon detected at ({detection.center_x}, {detection.center_y})")
            
            # Step 2: Launch Notepad
            logger.info("Step 2: Launching Notepad...")
            if not self.controller.launch_notepad(detection.center_x, detection.center_y):
                logger.error("Failed to launch Notepad")
                self.stats['automation_failures'] += 1
                return False
            
            logger.info("✓ Notepad launched")
            time.sleep(1)  # Wait for window to fully load
            
            # Step 3: Type content
            logger.info("Step 3: Typing post content...")
            content = post.format_content()
            if not self.controller.type_text(content, clear_first=True):
                logger.error("Failed to type content")
                self.stats['automation_failures'] += 1
                self._safe_close_notepad()
                return False
            
            logger.info("✓ Content typed")
            time.sleep(0.5)
            
            # Step 4: Save file
            logger.info("Step 4: Saving file...")
            filename = f"post_{post.id}.txt"
            filepath = self.output_dir / filename
            
            if not self.controller.save_file(filepath, handle_overwrite=True):
                logger.error("Failed to save file")
                self.stats['automation_failures'] += 1
                self._safe_close_notepad()
                return False
            
            logger.info(f"✓ File saved: {filepath}")
            time.sleep(0.5)
            
            # Step 5: Close Notepad
            logger.info("Step 5: Closing Notepad...")
            if not self.controller.close_notepad(save_changes=False):
                logger.warning("Failed to close Notepad cleanly")
                # Not critical - continue
            
            logger.info("✓ Notepad closed")
            
            # Step 6: Show desktop again for next post (minimize all windows)
            logger.debug("Showing desktop for next icon detection...")
            pyautogui.hotkey('win', 'd')
            time.sleep(1.0)  # Wait for desktop to appear
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing post: {e}", exc_info=True)
            self._safe_close_notepad()
            return False
    
    def _safe_close_notepad(self):
        """Safely close Notepad windows in case of error."""
        try:
            logger.debug("Attempting safe Notepad cleanup...")
            self.controller.close_notepad(save_changes=False)
        except Exception as e:
            logger.debug(f"Safe close failed: {e}")
    
    def _cleanup(self):
        """Cleanup any remaining Notepad windows."""
        logger.info("\nPerforming cleanup...")
        
        try:
            if self.controller:
                closed = self.controller.cleanup_all_notepad_windows()
                if closed > 0:
                    logger.info(f"Cleaned up {closed} Notepad windows")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _print_summary(self, duration: float):
        """Print workflow execution summary."""
        logger.info("\n" + "=" * 60)
        logger.info("WORKFLOW SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total posts:           {self.stats['total_posts']}")
        logger.info(f"Successful:            {self.stats['successful']} ✓")
        logger.info(f"Failed:                {self.stats['failed']} ✗")
        logger.info(f"  - Detection failures: {self.stats['detection_failures']}")
        logger.info(f"  - Automation failures: {self.stats['automation_failures']}")
        
        if self.stats['total_posts'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_posts']) * 100
            logger.info(f"Success rate:          {success_rate:.1f}%")
        
        logger.info(f"Execution time:        {duration:.1f}s")
        logger.info(f"Output directory:      {self.output_dir}")
        logger.info(f"Debug directory:       {self.debug_dir}")
        logger.info("=" * 60)


def main():
    """Main entry point for the automation application."""
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    TEMPLATE_PATH = PROJECT_ROOT / "assets" / "notepad_icon.png"
    
    # Create output directory with timestamp
    timestamp = create_timestamp_string()
    OUTPUT_BASE = Path.home() / "Desktop" / "tjm-project"
    OUTPUT_DIR = OUTPUT_BASE
    
    # Number of posts to process
    NUM_POSTS = 2  # Testing with 2 posts (change back to 10 when ready)
    
    # Setup logging
    LOG_FILE = PROJECT_ROOT / "automation.log"
    setup_logging(str(LOG_FILE))
    
    logger.info("=" * 60)
    logger.info("CV DESKTOP AUTOMATION APPLICATION")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Template path: {TEMPLATE_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60 + "\n")
    
    # Check template exists
    if not TEMPLATE_PATH.exists():
        logger.error(f"Template image not found: {TEMPLATE_PATH}")
        logger.error("Please create a template image of the Notepad icon")
        logger.error("and save it as 'assets/notepad_icon.png'")
        logger.error("\nInstructions:")
        logger.error("1. Take a screenshot of your desktop")
        logger.error("2. Crop just the Notepad icon (about 64x64 pixels)")
        logger.error("3. Save as assets/notepad_icon.png")
        return 1
    
    try:
        # Create and run workflow
        workflow = AutomationWorkflow(
            template_path=TEMPLATE_PATH,
            output_dir=OUTPUT_DIR,
            num_posts=NUM_POSTS
        )
        
        success = workflow.run()
        
        if success:
            logger.info("\n✓✓✓ Workflow completed successfully! ✓✓✓\n")
            return 0
        else:
            logger.error("\n✗✗✗ Workflow completed with errors ✗✗✗\n")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
