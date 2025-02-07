from pathlib import Path
from dataclasses import dataclass
import logging
import shutil
from datetime import datetime
import chess
from typing import Optional, Dict, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileConfig:
    """Configuration for file paths and storage options"""
    base_dir: Path = Path('rev_1')
    position_dir: Path = Path('rev_1/position_history')
    archive_dir: Path = Path('archive')
    max_positions_to_keep: int = 1000  # Prevent unlimited file growth

class FileManager:
    """
    Handles all file operations for the chess application including:
    - Saving and loading positions
    - Managing directory structure
    - Position history cleanup
    - Game session archiving
    """
    
    def __init__(self, config: Optional[FileConfig] = None):
        """
        Initialize FileManager with optional custom configuration
        
        Args:
            config: Optional custom configuration, uses defaults if None
        """
        self.config = config or FileConfig()
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        try:
            self.config.base_dir.mkdir(parents=True, exist_ok=True)
            self.config.position_dir.mkdir(parents=True, exist_ok=True)
            self.config.archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Create other required subdirectories
            (self.config.base_dir / 'questionnaire').mkdir(exist_ok=True)
            (self.config.base_dir / 'prompts').mkdir(exist_ok=True)
            (self.config.base_dir / 'responses').mkdir(exist_ok=True)
            (self.config.base_dir / 'keep_an_eye_on').mkdir(exist_ok=True)
            
            logger.info("Directory structure verified/created successfully")
        except Exception as e:
            logger.error(f"Error creating directory structure: {e}")
            raise
    
    def save_position(self, board: chess.Board, move_num: int) -> None:
        """
        Save current board position and FEN string
        
        Args:
            board: Current chess board state
            move_num: Current move number
        """
        try:
            # Determine color that just moved (opposite of current turn)
            color = 'b' if board.turn == chess.WHITE else 'w'
            
            # Create position data
            position_data = {
                'fen': board.fen(),
                'ascii': str(board),
                'timestamp': datetime.now().isoformat(),
                'move_number': move_num,
                'color': color
            }
            
            # Save position file
            position_file = self.config.position_dir / f'board_position_move_{move_num}_{color}.json'
            position_file.write_text(json.dumps(position_data, indent=2))
            
            logger.info(f"Position saved: {position_file}")
            
            # Cleanup old positions if needed
            self._cleanup_old_positions()
            
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            raise
    
    def _cleanup_old_positions(self) -> None:
        """Remove oldest position files if count exceeds max_positions_to_keep"""
        try:
            position_files = sorted(self.config.position_dir.glob('board_position_move_*.json'))
            if len(position_files) > self.config.max_positions_to_keep:
                files_to_remove = position_files[:-self.config.max_positions_to_keep]
                for file in files_to_remove:
                    file.unlink()
                logger.info(f"Cleaned up {len(files_to_remove)} old position files")
        except Exception as e:
            logger.error(f"Error during position cleanup: {e}")
    
    def archive_session(self) -> Path:
        """
        Archive current session directory with timestamp
        
        Returns:
            Path: Path to the archived directory
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"rev_1_{timestamp}"
            archive_path = self.config.archive_dir / archive_name
            
            # Move current session to archive
            shutil.move(str(self.config.base_dir), str(archive_path))
            
            # Recreate directory structure
            self._ensure_directories()
            
            logger.info(f"Session archived to: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Error archiving session: {e}")
            raise
    
    def get_position_history(self) -> List[Dict]:
        """
        Get list of all saved positions in chronological order
        
        Returns:
            List[Dict]: List of position data dictionaries
        """
        try:
            position_files = sorted(self.config.position_dir.glob('board_position_move_*.json'))
            positions = []
            
            for file in position_files:
                position_data = json.loads(file.read_text())
                positions.append(position_data)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error reading position history: {e}")
            return []
    
    def load_position(self, move_num: int, color: str) -> Optional[Dict]:
        """
        Load a specific position from history
        
        Args:
            move_num: Move number to load
            color: Color that moved ('w' or 'b')
            
        Returns:
            Optional[Dict]: Position data if found, None otherwise
        """
        try:
            position_file = self.config.position_dir / f'board_position_move_{move_num}_{color}.json'
            if position_file.exists():
                return json.loads(position_file.read_text())
            return None
            
        except Exception as e:
            logger.error(f"Error loading position: {e}")
            return None
    
    def copy_file(self, source: Path, dest: Path) -> None:
        """
        Copy a file from source to destination
        
        Args:
            source: Source file path
            dest: Destination file path
        """
        try:
            shutil.copy2(str(source), str(dest))
            logger.info(f"Copied {source} to {dest}")
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            raise
