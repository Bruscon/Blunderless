import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import os

class ChessPieceDetector:
    def __init__(self, template_dir: str, threshold: float = 0.6):
        """
        Initialize the chess piece detector with templates.
        
        Args:
            template_dir: Directory containing piece template images
            threshold: Minimum confidence threshold for piece detection
        """
        self.threshold = threshold
        self.templates = self._load_templates(template_dir)
        
        # Cache for processed templates
        self._template_cache = {}
        
    def _load_templates(self, template_dir: str) -> Dict[str, np.ndarray]:
        """Load piece templates from the specified directory."""
        templates = {}
        
        # Map template filenames to chess notation
        piece_map = {
            'white_king': 'K',
            'white_queen': 'Q',
            'white_rook': 'R',
            'white_bishop': 'B',
            'white_knight': 'N',
            'white_pawn': 'P',
            'black_king': 'k',
            'black_queen': 'q',
            'black_rook': 'r',
            'black_bishop': 'b',
            'black_knight': 'n',
            'black_pawn': 'p',
            'empty_light': None,
            'empty_dark': None
        }
        
        for template_name, piece_symbol in piece_map.items():
            path = os.path.join(template_dir, f"{template_name}.png")
            if not os.path.exists(path):
                raise ValueError(f"Missing template for piece: {template_name}")
                
            # Read directly as grayscale
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise ValueError(f"Could not read template: {path}")
                
            templates[template_name] = template
                
        return templates

    def _preprocess_square(self, square: np.ndarray) -> np.ndarray:
        """
        Preprocess a square image for template matching.
        
        Args:
            square: Square image (64x64 BGR)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(square.shape) == 3:
            square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        
        # Normalize using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(square)
        
        return normalized
    
    def _get_processed_template(self, piece: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get or compute processed versions of a template.
        
        Args:
            piece: Piece symbol (e.g., 'K', 'q')
            
        Returns:
            Tuple of (grayscale_template, edge_template)
        """
        if piece in self._template_cache:
            return self._template_cache[piece]
            
        # Get grayscale template
        gray_template = self.templates[piece]
        
        # Create edge-detected version for additional matching
        edge_template = cv2.Canny(gray_template, 50, 150)
        
        # Cache the results
        self._template_cache[piece] = (gray_template, edge_template)
        return gray_template, edge_template
    
    def detect_piece(self, square: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect chess piece in a square using template matching.
        
        Args:
            square: Square image (64x64 BGR)
            
        Returns:
            Tuple of (piece_symbol, confidence) where piece_symbol is None for empty squares
        """
        square_normalized = self._preprocess_square(square)
        square_edges = cv2.Canny(square_normalized, 50, 150)
        
        best_match = None
        best_score = float('-inf')
        best_template = None
        all_scores = {}
        
        # Try matching against all templates (including empty squares)
        for template_name, template in self.templates.items():
            template_edges = cv2.Canny(template, 50, 150)
            
            # Match against normalized images
            gray_result = cv2.matchTemplate(
                square_normalized,
                template,
                cv2.TM_CCOEFF_NORMED
            )
            edge_result = cv2.matchTemplate(
                square_edges,
                template_edges,
                cv2.TM_CCOEFF_NORMED
            )
            
            gray_score = np.max(gray_result)
            edge_score = np.max(edge_result)
            combined_score = 0.7 * gray_score + 0.3 * edge_score
            
            all_scores[template_name] = combined_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_template = template_name
        
        # Map the template name back to chess notation
        piece_map = {
            'white_king': 'K',
            'white_queen': 'Q',
            'white_rook': 'R',
            'white_bishop': 'B',
            'white_knight': 'N',
            'white_pawn': 'P',
            'black_king': 'k',
            'black_queen': 'q',
            'black_rook': 'r',
            'black_bishop': 'b',
            'black_knight': 'n',
            'black_pawn': 'p',
            'empty_light': None,
            'empty_dark': None
        }
        
        # Print debug info
        if best_score > 0.3:  # Keep debug output for scores in interesting range
            print(f"Debug - Best match: {best_template} (score: {best_score:.3f})")
            print("All scores:", {k: f"{v:.3f}" for k, v in all_scores.items()})
        
        return piece_map[best_template], best_score
    
    def detect_board(self, squares: List[np.ndarray]) -> List[List[Optional[str]]]:
        """
        Detect pieces on entire chess board.
        
        Args:
            squares: List of 64 square images (rank-major order)
            
        Returns:
            8x8 matrix of piece symbols (or None for empty squares)
        """
        if len(squares) != 64:
            raise ValueError(f"Expected 64 squares, got {len(squares)}")
            
        # Process all squares
        board = []
        for rank in range(8):
            rank_pieces = []
            for file in range(8):
                square_idx = rank * 8 + file
                piece, _ = self.detect_piece(squares[square_idx])
                rank_pieces.append(piece)
            board.append(rank_pieces)
            
        return board
    
    def validate_detection(self, board: List[List[Optional[str]]]) -> bool:
        """
        Validate detected board position.
        
        Args:
            board: 8x8 matrix of piece symbols
            
        Returns:
            True if position is valid, False otherwise
        """
        # Count pieces
        piece_counts = {
            'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0, 'P': 0,
            'k': 0, 'q': 0, 'r': 0, 'b': 0, 'n': 0, 'p': 0
        }
        
        for rank in board:
            for piece in rank:
                if piece is not None:
                    piece_counts[piece] += 1
        
        # Basic validation rules
        if piece_counts['K'] != 1 or piece_counts['k'] != 1:
            return False  # Must have exactly one king of each color
            
        if piece_counts['Q'] > 1 or piece_counts['q'] > 1:
            return False  # Can't have more than one queen of each color
            
        if piece_counts['P'] > 8 or piece_counts['p'] > 8:
            return False  # Can't have more than 8 pawns of each color
            
        # Add more validation rules as needed
        
        return True