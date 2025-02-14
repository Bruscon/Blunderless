import tkinter as tk
from pathlib import Path
import logging
import sys
from typing import Optional
import argparse
from datetime import datetime

from chess_gui import ChessGUI, GuiConfig
from file_manager import FileManager, FileConfig
from board_state import BoardState
#from board_detection import ChessBoardDetector

# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chess_session_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
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
        self.logger = logging.getLogger(__name__)
        self.root = tk.Tk()
        self.setup_window()
        
        # Initialize components
        self.file_manager = FileManager(file_config)
        self.gui = ChessGUI(self.root, gui_config)
        
        # Set up callbacks
        self.gui.set_move_callback(self.on_move_made)
        
        # Set up application exit handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
