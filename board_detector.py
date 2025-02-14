import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class BoardDetector:
    def __init__(self, target_size: int = 800):
        self.target_size = target_size
        self.square_size = target_size // 8

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the screenshot specifically for chess.com board detection.
        Uses edge detection to find the white border of the board.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            Processed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("images/debug_gray.png", gray)
        
        # Detect edges with lower threshold to catch the board border
        edges = cv2.Canny(gray, 30, 100)
        cv2.imwrite("images/debug_edges.png", edges)
        
        # Dilate edges to connect them
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cv2.imwrite("images/debug_dilated.png", dilated)
        
        return dilated

    def detect_board_bounds(self, binary_image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Detect chess board boundaries by finding the consistent checkerboard pattern and rank/file markers.
        
        Args:
            binary_image: Edge-detected and dilated image
        
        Returns:
            Tuple of ((min_x, min_y), (max_x, max_y)) representing board bounds
        """
        height, width = binary_image.shape
        print(f"Processing image of size: {width}x{height}")
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")
        
        # Draw all contours for debugging
        debug_contours = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.drawContours(debug_contours, contours, -1, (0,255,0), 2)
        cv2.imwrite("images/debug_all_contours.png", debug_contours)
        
        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to find a square-like contour among the largest contours
        for contour in contours[:10]:  # Only check the 10 largest contours
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If it has 4 vertices, it might be our board
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = w * h
                area_ratio = area / (width * height)
                
                print(f"Found potential board: aspect_ratio={aspect_ratio:.2f}, area_ratio={area_ratio:.2f}")
                
                # The board should be a significant portion of the image
                # and should be roughly square
                if area_ratio > 0.2 and 0.9 < aspect_ratio < 1.1:
                    # Calculate the size of one square (board is 8x8)
                    square_size = w // 8
                    
                    # Adjust bounds to perfectly contain 8x8 squares plus a small margin
                    adjusted_x = x - 2  # Small margin for the rank numbers
                    adjusted_y = y - 2  # Small margin for any top border
                    adjusted_w = square_size * 8 + 4  # Add margins
                    adjusted_h = square_size * 8 + 4  # Add margins
                    
                    # Draw final detected board
                    debug_final = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.drawContours(debug_final, [approx], -1, (0,0,255), 2)
                    cv2.rectangle(debug_final, 
                                (adjusted_x, adjusted_y), 
                                (adjusted_x + adjusted_w, adjusted_y + adjusted_h), 
                                (255,0,0), 2)
                    cv2.imwrite("images/debug_final_detection.png", debug_final)
                    
                    return (adjusted_x, adjusted_y), (adjusted_x + adjusted_w, adjusted_y + adjusted_h)
        
        raise ValueError("Could not detect chess board")
        
        raise ValueError("Could not detect chess board")

    def extract_board(self, image: np.ndarray, bounds: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        """
        Extract and resize the chess board from the image.
        
        Args:
            image: Original image
            bounds: Board boundaries as ((min_x, min_y), (max_x, max_y))
        
        Returns:
            Extracted and resized board image
        """
        (min_x, min_y), (max_x, max_y) = bounds
        
        # Extract board region
        board = image[min_y:max_y, min_x:max_x]
        
        # Resize to target size
        board = cv2.resize(board, (self.target_size, self.target_size))
        
        return board

    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual squares from the board image.
        
        Args:
            board_image: Chess board image
        
        Returns:
            List of 64 square images
        """
        squares = []
        MARGIN = 4  # Must match template processor margin
        
        # Get all squares
        files = 'abcdefgh'
        for rank in range(8):
            for file in range(8):
                # Calculate square coordinates
                x = file * self.square_size
                y = rank * self.square_size
                
                # Extract square
                square = board_image[
                    y:y + self.square_size,
                    x:x + self.square_size
                ]
                
                # Resize to 64x64
                square = cv2.resize(square, (64, 64))
                
                # Crop margins
                h, w = square.shape[:2]
                square = square[MARGIN:h-MARGIN, MARGIN:w-MARGIN]
                
                # Resize back to 64x64 after cropping
                square = cv2.resize(square, (64, 64))
                
                # Convert to grayscale
                square_gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                
                # Save processed square with chess notation (e.g., 'e4')
                square_name = f"{files[file]}{8-rank}"
                cv2.imwrite(f"images/square_{square_name}.png", square_gray)
                
                squares.append((rank, file, square_gray))
        
        return squares

    def process_image(self, image_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process a chess board screenshot and extract squares.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Tuple of (board_image, list_of_squares)
        """
        # Create images directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Preprocess image
        binary = self.preprocess_image(image)
        
        # Detect board boundaries
        bounds = self.detect_board_bounds(binary)
        
        # Extract board
        board = self.extract_board(image, bounds)
        
        # Extract squares
        squares = self.extract_squares(board)
        
        return board, squares