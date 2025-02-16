import tkinter as tk
from pathlib import Path
import logging
import sys
from typing import Optional
import argparse
from datetime import datetime
import chess
import subprocess
from mss import mss

from chess_gui import ChessGUI, GuiConfig
from file_manager import FileManager, FileConfig
from board_state import BoardState
from board_detector import BoardDetector
from piece_detector import PieceDetector
from threaded_capture import ThreadedCapture

def setup_logging(log_level: str = "INFO") -> None:
    # Reset logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chess_session_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )

class ChessApplication:
    """
    Main application class that coordinates the chess components
    and manages the application lifecycle.
    """
    
    def __init__(
        self,
        gui_config: Optional[GuiConfig] = None,
        file_config: Optional[FileConfig] = None
    ):
        """
        Initialize the chess application
        
        Args:
            gui_config: Optional GUI configuration
            file_config: Optional file management configuration
        """
        self.logger = logging.getLogger("__main__")
        
        # Initialize screen capture
        self.screen = mss()
        
        # Initialize root window
        self.root = tk.Tk()
        self.setup_window()
        
        # Process templates on startup
        self.process_templates()
        
        # Initialize computer vision components
        self.board_detector = BoardDetector()
        self.piece_detector = PieceDetector("templates")
        
        # Initialize components
        self.file_manager = FileManager(file_config)
        self.gui = ChessGUI(self.root, gui_config)
        
        # Set up callbacks
        self.gui.set_move_callback(self.on_move_made)
        
        # Set up spacebar binding
        self.root.bind('<space>', self.handle_screenshot_capture)
        
        # Set up application exit handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.auto_capture = False
        self.threaded_capture = ThreadedCapture(
            capture_func=self.capture_screenshot,
            process_func=self.process_board_position,
            gui=self.root  # Change this from self.gui to self.root
        )
        self.root.bind('<space>', self.toggle_auto_capture)
        
    def setup_window(self) -> None:
        """Configure the main window"""
        self.root.title("Chess Application")
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position for center of screen
        window_width = 800  # Estimated width
        window_height = 600  # Estimated height
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def on_move_made(self, move) -> None:
        """
        Callback for when a move is made
        
        Args:
            move: The chess move that was made
        """
        self.logger.info(f"Move made: {move}")
        # Add any additional move processing here
    
    def on_closing(self) -> None:
        """Handle application shutdown"""
        try:
            # Archive current session
            self.file_manager.archive_session()
            self.logger.info("Session archived successfully")
            
            # Cleanup and exit
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.root.destroy()
    
    def run(self) -> None:
        """Start the application"""
        self.logger.info("Starting chess application")
        self.root.mainloop()

    def process_templates(self):
        """Process piece templates on application startup"""
        try:
            self.logger.info("Processing piece templates...")
            # Run the template processor script
            subprocess.run(["python", "template_processor.py"], check=True)
            self.logger.info("Template processing completed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Template processing failed: {e}")
            # You might want to handle this error more gracefully
            raise

    def capture_screenshot(self):
        """Capture screenshot from primary monitor and save it"""
        try:
            self.logger.info("Capturing screenshot...")
            # Use monitor 1 instead of 2
            screenshot = self.screen.grab(self.screen.monitors[2])
            
            # Save screenshot using PIL
            from PIL import Image
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            screenshot_path = "chess_screenshot.png"
            img.save(screenshot_path)
            
            return screenshot_path
            
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            raise

    def process_board_position(self):
        """Process screenshot to detect chess position and update the board"""
        try:
            self.logger.info("Processing board position...")
            
            # Process the image through board detector
            board_image, squares = self.board_detector.process_image("chess_screenshot.png")
            
            # Detect pieces and get FEN string
            fen = self.piece_detector.detect_board(squares)
            self.logger.info(f"Detected FEN: {fen}")
            
            # Add required FEN fields for a complete FEN string if not present
            if len(fen.split()) == 1:
                fen = f"{fen} w KQkq - 0 1"
                
            # Update the board state with new position
            self.gui.board_state = BoardState()  # Reset board state
            self.gui.board_state.board.set_fen(fen)
            self.logger.info(f"Current board FEN: {self.gui.board_state.get_fen()}")
            
            # Force GUI refresh
            self.root.update_idletasks()
            self.gui.update_display()
            
            self.logger.info(f"Board position updated from image: {fen}")
            
        except Exception as e:
            self.logger.error(f"Board position processing failed: {e}")
            self.gui.set_status(f"Error processing board position: {str(e)}")

    def handle_screenshot_capture(self, event=None):
        """Handle spacebar press to capture and process board position"""
        try:
            self.gui.set_status("Capturing and processing board position...")
            
            # Capture screenshot
            self.capture_screenshot()
            
            # Process the board position
            self.process_board_position()
            
            self.gui.set_status("Board position updated from screenshot")
            
        except Exception as e:
            self.logger.error(f"Screenshot capture and processing failed: {e}")
            self.gui.set_status(f"Error: {str(e)}")

    def toggle_auto_capture(self, event=None):
        """Toggle automatic board position capture"""
        self.auto_capture = not self.auto_capture
        if self.auto_capture:
            self.gui.set_status("Auto-capture enabled")
            self.threaded_capture.start()
        else:
            self.gui.set_status("Auto-capture disabled")
            self.threaded_capture.stop()

    def on_closing(self):
        """Handle application shutdown"""
        try:
            # Stop auto-capture if running
            if self.auto_capture:
                self.threaded_capture.stop()
            
            # Archive current session
            self.file_manager.archive_session()
            self.logger.info("Session archived successfully")
            
            # Cleanup and exit
            self.root.destroy()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.root.destroy()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Chess Application")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    parser.add_argument(
        "--theme",
        default="default",
        choices=["default", "blue", "green"],
        help="Set the color theme"
    )
    return parser.parse_args()

def get_theme_config(theme: str) -> GuiConfig:
    """Get GUI configuration for specified theme"""
    themes = {
        "default": GuiConfig(),
        "blue": GuiConfig(
            LIGHT_SQUARE="#B6D0E2",
            DARK_SQUARE="#4682B4",
            HIGHLIGHT_COLOR="#FFD700"
        ),
        "green": GuiConfig(
            LIGHT_SQUARE="#C8E6C9",
            DARK_SQUARE="#2E7D32",
            HIGHLIGHT_COLOR="#FFA000"
        )
    }
    return themes.get(theme, GuiConfig())

def main() -> None:
    """Main entry point for the chess application"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        
        # Get theme configuration
        gui_config = get_theme_config(args.theme)
        
        # Create and run application
        app = ChessApplication(gui_config=gui_config)
        app.run()
        
    except Exception as e:
        logging.critical(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
