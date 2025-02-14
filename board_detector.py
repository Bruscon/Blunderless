import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class BoardDetector:
    def __init__(self, debug_output: bool = True):
        """Initialize the board detector."""
        self.debug_output = debug_output
        self.debug_dir = "images/debug"
        if debug_output:
            os.makedirs(self.debug_dir, exist_ok=True)

    def save_debug_image(self, name: str, image: np.ndarray) -> None:
        """Save a debug image if debug output is enabled."""
        if self.debug_output:
            cv2.imwrite(os.path.join(self.debug_dir, f"{name}.png"), image)

    def find_main_board(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Find the main chess board in the image using contour detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (cropped board image, (x, y, w, h) of board)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.save_debug_image("01_initial_thresh", thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and look for largest square-like contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Check top 5 largest contours
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            area_ratio = w*h / (image.shape[0]*image.shape[1])
            
            # Look for square-ish contour that takes up reasonable portion of image
            if 0.8 < aspect_ratio < 1.2 and 0.2 < area_ratio < 0.9:
                # Draw detected board region
                debug_img = image.copy()
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0,255,0), 2)
                self.save_debug_image("02_detected_board", debug_img)
                
                return image[y:y+h, x:x+w], (x, y, w, h)
                
        raise ValueError("Could not find main chess board")

    def snap_to_edge(self, pos: int, is_vertical: bool, image_width: int, image_height: int, edges: np.ndarray, window: int = 3) -> int:
        """
        Snap a grid line position to the nearest strong edge.
        
        Args:
            pos: Initial line position
            is_vertical: True if vertical line, False if horizontal
            image_width: Width of the image
            image_height: Height of the image
            edges: Edge detection output
            window: Pixel window to search for edges
        
        Returns:
            Adjusted line position
        """
        # Handle boundary cases specifically
        if is_vertical and (pos <= window or pos >= image_width - window):
            return pos  # Keep exact position for first and last vertical lines
        if not is_vertical and (pos <= window or pos >= image_height - window):
            return pos  # Keep exact position for first and last horizontal lines
            
        if is_vertical:
            # Look in vertical strip around proposed x position
            strip = edges[:, max(0, pos-window):min(image_width, pos+window)]
            if strip.any():  # If any edges found in strip
                # Find the strongest edge in the strip
                edge_strength = np.sum(strip, axis=0)
                offset = np.argmax(edge_strength)
                return pos - window + offset
        else:
            # Look in horizontal strip around proposed y position
            strip = edges[max(0, pos-window):min(image_height, pos+window), :]
            if strip.any():  # If any edges found in strip
                # Find the strongest edge in the strip
                edge_strength = np.sum(strip, axis=1)
                offset = np.argmax(edge_strength)
                return pos - window + offset
        return pos

    def find_grid_lines(self, board_image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find grid lines using advanced edge detection and line snapping.
        
        Args:
            board_image: Cropped chess board image
            
        Returns:
            Tuple of (x_coordinates, y_coordinates) for grid lines
        """
        height, width = board_image.shape[:2]
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multi-scale edge detection
        edges_low = cv2.Canny(enhanced, 30, 90)
        edges_high = cv2.Canny(enhanced, 90, 270)
        edges = cv2.addWeighted(edges_low, 0.5, edges_high, 0.5, 0)
        
        # Enhance horizontal and vertical edges separately
        kernel_h = np.ones((1, 3), np.uint8)
        kernel_v = np.ones((3, 1), np.uint8)
        edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
        edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
        
        self.save_debug_image("03a_edges_horizontal", edges_h)
        self.save_debug_image("03b_edges_vertical", edges_v)
        
        # Calculate initial grid positions
        x_coords = [round(width * i / 8) for i in range(9)]
        y_coords = [round(height * i / 8) for i in range(9)]
        
        # Snap lines to edges with adaptive window size
        def adaptive_window(pos: int, max_dim: int) -> int:
            # Use larger window in middle of board, smaller near edges
            rel_pos = min(pos, max_dim - pos) / max_dim
            return max(3, min(7, round(5 * rel_pos)))
        
        # Snap vertical lines using vertical edge map
        x_coords = [
            self.snap_to_edge(x, True, width, height, edges_v, 
                             window=adaptive_window(x, width))
            for x in x_coords
        ]
        
        # Snap horizontal lines using horizontal edge map
        y_coords = [
            self.snap_to_edge(y, False, width, height, edges_h,
                             window=adaptive_window(y, height))
            for y in y_coords
        ]
        
        # Ensure minimum spacing between lines
        def enforce_spacing(coords: List[int], min_spacing: int) -> List[int]:
            for i in range(1, len(coords)):
                if coords[i] - coords[i-1] < min_spacing:
                    coords[i] = coords[i-1] + min_spacing
            return coords
        
        min_spacing = min(width, height) // 10  # Minimum 1/10th of board size
        x_coords = enforce_spacing(x_coords, min_spacing)
        y_coords = enforce_spacing(y_coords, min_spacing)
        
        # Draw debug visualization
        debug_img = board_image.copy()
        
        # Draw vertical lines
        for x in x_coords:
            cv2.line(debug_img, (x, 0), (x, height), (0, 0, 255), 1)
            
        # Draw horizontal lines
        for y in y_coords:
            cv2.line(debug_img, (0, y), (width, y), (0, 255, 0), 1)
            
        self.save_debug_image("04_grid_lines", debug_img)
        
        return x_coords, y_coords

    def extract_squares(self, board_image: np.ndarray, x_coords: List[int], y_coords: List[int]) -> List[Tuple[int, int, np.ndarray]]:
        """
        Extract individual squares using grid line coordinates.
        
        Args:
            board_image: Cropped chess board image
            x_coords: X coordinates of vertical grid lines
            y_coords: Y coordinates of horizontal grid lines
            
        Returns:
            List of tuples (rank, file, square_image) in row-major order (a8 to h1)
        """
        height, width = board_image.shape[:2]
        squares = []
        debug_img = board_image.copy()
        
        # Extract each square
        for rank in range(8):
            for file in range(8):
                # Get square corners with boundary checks
                x1 = min(max(x_coords[file], 0), width - 1)
                x2 = min(max(x_coords[file + 1], x1 + 1), width)
                y1 = min(max(y_coords[rank], 0), height - 1)
                y2 = min(max(y_coords[rank + 1], y1 + 1), height)
                
                # Extract and resize square
                square = board_image[y1:y2, x1:x2]
                square = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
                
                # Store square with its position
                squares.append((rank, file, square))
                
                # Draw square outline on debug image
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # Add coordinate labels for debugging
                label = f"{rank},{file}"
                cv2.putText(debug_img, label, (x1 + 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        self.save_debug_image("05_extracted_squares", debug_img)
        return squares

    def process_image(self, image_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process a chess board screenshot.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (board_image, list_of_squares)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Find and extract the main board region
        board_image, board_rect = self.find_main_board(image)
        
        # Find grid lines
        x_coords, y_coords = self.find_grid_lines(board_image)
        
        # Extract squares
        squares = self.extract_squares(board_image, x_coords, y_coords)
        
        return board_image, squares