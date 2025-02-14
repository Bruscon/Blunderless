#import mss
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import chess
import logging

logger = logging.getLogger(__name__)

class ChessBoardDetector:
    """
    Handles chess board detection and FEN string generation from screen captures
    using tensorflow_chessbot.
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize the chess board detector
        
        Args:
            model_path: Path to the tensorflow_chessbot model file
        """
        self.model = tf.saved_model.load(str(model_path))
        self.sct = mss.mss()
        logger.info("ChessBoardDetector initialized")
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture the entire screen
        
        Returns:
            np.ndarray: Screenshot as a numpy array
        """
        screenshot = self.sct.grab(self.sct.monitors[1])  # Primary monitor
        return np.array(screenshot)
    
    def detect_board(self, image: np.ndarray) -> tuple[bool, str]:
        """
        Detect chess board and generate FEN string from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            tuple[bool, str]: (Success flag, FEN string if successful)
        """
        try:
            # Preprocess image for model
            processed_img = self._preprocess_image(image)
            
            # Run detection
            predictions = self.model(processed_img)
            
            # Process predictions to FEN string
            fen = self._predictions_to_fen(predictions)
            
            if self._validate_fen(fen):
                return True, fen
            return False, ""
            
        except Exception as e:
            logger.error(f"Error detecting board: {e}")
            return False, ""
    
    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess image for the model
        
        Args:
            image: Input image
            
        Returns:
            tf.Tensor: Preprocessed image tensor
        """
        # Convert to RGB if needed
        if image.shape[-1] == 4:  # RGBA
            image = Image.fromarray(image).convert('RGB')
            image = np.array(image)
        
        # Resize and normalize
        image = tf.image.resize(image, [512, 512])
        image = image / 255.0
        
        # Add batch dimension
        return tf.expand_dims(image, 0)
    
    def _predictions_to_fen(self, predictions) -> str:
        """
        Convert model predictions to FEN string
        
        Args:
            predictions: Model output tensor
            
        Returns:
            str: FEN string
        """
        # This will need to be implemented based on the specific model output format
        # For now, returning a placeholder
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def _validate_fen(self, fen: str) -> bool:
        """
        Validate FEN string
        
        Args:
            fen: FEN string to validate
            
        Returns:
            bool: True if valid FEN string
        """
        try:
            chess.Board(fen)
            return True
        except ValueError:
            return False

    def monitor_for_board(self, callback):
        """
        Continuously monitor screen for chess boards
        
        Args:
            callback: Function to call when board is detected with FEN string
        """
        while True:
            try:
                screen = self.capture_screen()
                success, fen = self.detect_board(screen)
                
                if success:
                    callback(fen)
                    
            except Exception as e:
                logger.error(f"Error in board monitoring: {e}")
