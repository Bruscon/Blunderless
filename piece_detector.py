import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import chess
import chess.svg

class PieceDetector:
    def __init__(self, template_dir: str, threshold: float = 0.2, debug_level: int = 0):
        """
        Initialize the chess piece detector with templates.
        
        Args:
            template_dir: Directory containing piece template images
            threshold: Minimum confidence threshold for piece detection
            debug_level: Debug output level (0-3)
                0: No debug output
                1: Print confidence scores matrix
                2: Add terminal debug output
                3: Full debug output including images
        """
        self.threshold = threshold
        self.debug_level = debug_level
        self.debug_dir = "images/piece_detection_debug"
        
        # First load templates
        self.templates = self._load_templates(template_dir)
        
        # Then create debug output if needed
        if debug_level == 3:
            os.makedirs(self.debug_dir, exist_ok=True)
            self._save_template_debug()
    
    def _save_template_debug(self):
        """Save visualization of loaded templates."""
        template_viz = np.zeros((64*2, 64*6, 3), dtype=np.uint8)
        
        # Arrange templates in a grid
        piece_types = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']
        colors = ['white', 'black']
        
        for i, color in enumerate(colors):
            for j, piece in enumerate(piece_types):
                template_name = f"{color}_{piece}"
                if template_name in self.templates:
                    template = self.templates[template_name]
                    # Convert to BGR for visualization
                    template_viz[i*64:(i+1)*64, j*64:(j+1)*64] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(os.path.join(self.debug_dir, "00_loaded_templates.png"), template_viz)
        
    def _load_templates(self, template_dir: str) -> Dict[str, np.ndarray]:
        """Load piece templates from the specified directory."""
        templates = {}
        
        # Map template filenames to chess notation
        piece_map = {
            'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
            'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
            'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p',
            'empty_light': None, 'empty_dark': None
        }
        
        missing_templates = []
        for template_name, piece_symbol in piece_map.items():
            path = os.path.join(template_dir, f"{template_name}.png")
            if not os.path.exists(path):
                missing_templates.append(template_name)
                continue
                
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                missing_templates.append(template_name)
                continue
                
            templates[template_name] = template
        
        if missing_templates:
            print("Warning: Missing templates:", missing_templates)
                
        return templates

    def _process_square(self, square: np.ndarray, is_light: bool = True) -> np.ndarray:
        """
        Process a square image to match template format.
        
        Args:
            square: Square image (64x64 BGR)
            is_light: Whether this is a light square
            
        Returns:
            Processed grayscale image
        """
        if self.debug_level >= 2:
            print("\nProcessing square:")
            print(f"Input shape: {square.shape}")
        if self.debug_level == 3:
            cv2.imwrite(os.path.join(self.debug_dir, "00_input_square.png"), square)
        
        # Convert to grayscale if needed
        if len(square.shape) == 3:
            square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        
        # Crop 2 pixels from each edge to remove artifacts
        h, w = square.shape
        square = square[2:h-2, 2:w-2]
        
        # Sample background from top right corner (5x5 region)
        bg_sample = square[0:5, -5:]
        bg_color = int(np.median(bg_sample))
        
        if self.debug_level >= 2:
            print(f"Background color: {bg_color} ({'light' if bg_color > 127 else 'dark'} square)")
        if self.debug_level == 3:
            # Save background sample visualization
            bg_viz = square.copy()
            cv2.rectangle(bg_viz, (w-5, 0), (w, 5), (0, 255, 0), 1)
            cv2.imwrite(os.path.join(self.debug_dir, "01_bg_sample.png"), bg_viz)
        
        # Create a mask for the piece based on background
        if bg_color > 127:  # Light square
            # For light squares, piece pixels are darker
            piece_mask = square < (bg_color - 30)
        else:  # Dark square
            # For dark squares, piece pixels are lighter
            piece_mask = square > (bg_color + 30)
        
        # Create clean image with piece on black background
        clean_image = np.zeros_like(square)
        piece_color = 255  # White piece on black background
        clean_image[piece_mask] = piece_color

        if self.debug_level >= 2:
            print(f"Clean image shape before scaling: {clean_image.shape}")
        if self.debug_level == 3:
            cv2.imwrite(os.path.join(self.debug_dir, "02_clean_image.png"), clean_image)

        # Scale the piece to 80% like in template processing
        h, w = clean_image.shape
        target_size = (64, 64)
        scale = min(target_size[0]/w, target_size[1]/h) * 0.8  # Match template processing
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(clean_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create output image (black background)
        output = np.zeros(target_size, dtype=np.uint8)
        
        # Calculate centering offsets
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        
        # Place resized image in center
        output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        if self.debug_level >= 2:
            print(f"Final output shape: {output.shape}")
            print(f"Scale factor used: {scale:.3f}")
            print(f"New dimensions: {new_w}x{new_h}")
            print(f"Centering offsets: x={x_offset}, y={y_offset}")
        
        if self.debug_level == 3:
            cv2.imwrite(os.path.join(self.debug_dir, "03_processed_square.png"), output)
            
            # Create side-by-side comparison with template
            comparison = np.zeros((64, 128), dtype=np.uint8)
            comparison[:, :64] = output
            if len(self.templates) > 0:
                # Get first template for comparison
                first_template = next(iter(self.templates.values()))
                comparison[:, 64:] = first_template
                cv2.imwrite(os.path.join(self.debug_dir, "04_template_comparison.png"), comparison)
        
        return output
    
    def detect_piece(self, square: np.ndarray, is_light: bool) -> Tuple[Optional[str], float]:
        """
        Detect chess piece in a square using template matching.
        
        Args:
            square: Square image (64x64 BGR)
            is_light: Whether this is a light square
            
        Returns:
            Tuple of (piece_symbol, confidence) where piece_symbol is None for empty squares
        """
        # Process the square to match template format
        processed_square = self._process_square(square, is_light)
        
        best_match = None
        best_score = float('-inf')
        best_template = None
        
        # Create debug visualization
        if self.debug_level == 3:
            match_viz = np.zeros((64, 64*len(self.templates), 3), dtype=np.uint8)
            template_x = 0
        
        # Store match scores for debug output
        if self.debug_level >= 2:
            match_scores = []
        
        # Try matching against all templates
        for template_name, template in self.templates.items():
            # Skip empty square templates entirely
            if template_name.startswith('empty_'):
                continue
            
            # Use TM_SQDIFF_NORMED - Note: this gives 0 for perfect match, 1 for complete mismatch
            result = cv2.matchTemplate(
                processed_square,
                template,
                cv2.TM_SQDIFF_NORMED
            )
            
            # Convert score to be consistent with our expectations (1 = perfect match, 0 = complete mismatch)
            score = 1.0 - np.min(result)
            
            if self.debug_level >= 2:
                match_scores.append((template_name, score))
            
            if self.debug_level == 3:
                # Add template to visualization
                template_viz = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                match_viz[:, template_x:template_x+64] = template_viz
                # Add score text
                cv2.putText(match_viz, f"{score:.2f}", (template_x+2, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                template_x += 64
                
                # Save detailed debug for all pieces
                debug_dir = os.path.join(self.debug_dir, template_name)
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save processed square
                cv2.imwrite(os.path.join(debug_dir, "square.png"), processed_square)
                
                # Save template
                cv2.imwrite(os.path.join(debug_dir, "template.png"), template)
                
                # Create difference visualization
                diff = cv2.absdiff(processed_square, template)
                diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                
                # Create side-by-side comparison
                comparison = np.zeros((64, 192), dtype=np.uint8)
                comparison[:, :64] = processed_square
                comparison[:, 64:128] = template
                comparison[:, 128:] = diff_normalized
                
                # Add labels
                label_img = np.zeros((20, 192), dtype=np.uint8)
                label_img.fill(255)
                comparison = np.vstack([comparison, label_img])
                
                # Add score text
                cv2.putText(comparison, f"Score: {score:.3f}", (5, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                cv2.imwrite(os.path.join(debug_dir, f"comparison_{score:.3f}.png"), comparison)
                
                # Save pixel counts for quantitative comparison
                square_pixels = np.sum(processed_square > 127)
                template_pixels = np.sum(template > 127)
                with open(os.path.join(debug_dir, "pixel_counts.txt"), "w") as f:
                    f.write(f"Square white pixels: {square_pixels}\n")
                    f.write(f"Template white pixels: {template_pixels}\n")
                    f.write(f"Difference: {abs(square_pixels - template_pixels)}\n")
                    f.write(f"Match score: {score:.3f}\n")
            
            if score > best_score:
                best_score = score
                best_template = template_name

        if self.debug_level == 3:
            cv2.imwrite(os.path.join(self.debug_dir, "04_template_matching.png"), match_viz)
        
        if self.debug_level >= 2:
            print("\nTemplate matching scores:")
            for name, score in sorted(match_scores, key=lambda x: x[1], reverse=True):
                print(f"{name}: {score:.3f}")
        
        # Map template name to chess notation
        piece_map = {
            'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
            'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
            'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p',
            'empty_light': None, 'empty_dark': None
        }
        
        # Check if the square is empty based on the processed image content
        piece_content = np.sum(processed_square > 127) / (64 * 64)  # Fraction of white pixels
        if self.debug_level >= 2:
            print(f"Piece content (white pixel fraction): {piece_content:.3f}")
        
        # If less than 5% white pixels, consider it empty
        if piece_content < 0.05:
            return None, 1.0
            
        # Only return piece if score is above threshold
        if best_score < self.threshold:
            return None, best_score
            
        return piece_map[best_template], best_score
    
    def detect_board(self, squares: List[Tuple[int, int, np.ndarray]]) -> str:
        """
        Detect pieces on entire chess board and return FEN string.
        
        Args:
            squares: List of tuples (rank, file, square_image)
            
        Returns:
            FEN string representing the position
        """
        if len(squares) != 64:
            raise ValueError(f"Expected 64 squares, got {len(squares)}")
            
        # Initialize 8x8 board
        board = [[None for _ in range(8)] for _ in range(8)]
        detection_scores = [[0.0 for _ in range(8)] for _ in range(8)]
        
        # Process all squares
        for rank, file, square_img in squares:
            # Determine if light square
            is_light = (rank + file) % 2 == 0
            
            # Detect piece
            piece, score = self.detect_piece(square_img, is_light)
            
            # Store in board array
            board[rank][file] = piece
            detection_scores[rank][file] = score
        
        if self.debug_level >= 1:
            print("\nDetection confidence scores:")
            files = 'abcdefgh'
            for rank in range(8):
                row = []
                for file in range(8):
                    score = detection_scores[rank][file]
                    piece = board[rank][file] or '.'
                    row.append(f"{piece}:{score:.2f}")
                print(f"{8-rank} | {' '.join(row)}")
            print("   " + "-" * 55)
            print("    " + "  ".join(files))
        
        # Convert board array to FEN
        fen_parts = []
        for rank in board:
            empty_count = 0
            rank_str = ""
            
            for piece in rank:
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += piece
            
            if empty_count > 0:
                rank_str += str(empty_count)
            
            fen_parts.append(rank_str)
        
        return "/".join(fen_parts)
