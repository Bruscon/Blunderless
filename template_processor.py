import cv2
import numpy as np
import os
from typing import Dict, List

MARGIN = 4  # Pixels to crop from each edge

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to improve template matching.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Normalized image
    """
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    normalized = clahe.apply(image)
    
    return normalized

def process_template(image: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Process a template image into the format needed for matching.
    Includes cropping borders, resizing, and normalization.
    
    Args:
        image: Input piece image
        target_size: Target size for resizing (default 64x64)
    
    Returns:
        Processed template image
    """
    # Crop margins to remove borders
    h, w = image.shape[:2]
    cropped = image[MARGIN:h-MARGIN, MARGIN:w-MARGIN]
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image
    normalized = normalize_image(gray)
    
    return normalized

def main():
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Process starting position squares for each piece type
    piece_positions = {
        # White pieces (from rank 1 and 2)
        'white_king': 'e1',
        'white_queen': 'd1',
        'white_rook': 'h1',    # Using h1 rook to avoid the 'a' file notation
        'white_bishop': 'f1',   # Using f1 bishop to avoid edge
        'white_knight': 'g1',   # Using g1 knight to avoid edge
        'white_pawn': 'd2',     # Using d2 pawn to avoid edge
        # Black pieces (from rank 7 and 8)
        'black_king': 'e8',
        'black_queen': 'd8',
        'black_rook': 'h8',     # Using h8 rook to avoid the 'a' file notation
        'black_bishop': 'f8',   # Using f8 bishop to avoid edge
        'black_knight': 'g8',   # Using g8 knight to avoid edge
        'black_pawn': 'd7',     # Using d7 pawn to avoid edge
        # Empty squares (from middle of board)
        'empty_light': 'e4',    # Empty light square from middle
        'empty_dark': 'd5'      # Empty dark square from middle
    }
    
    # Process each piece/square
    for template_name, pos in piece_positions.items():
        # Read the square image
        square_path = f'images/square_{pos}.png'
        if not os.path.exists(square_path):
            print(f"Warning: Missing square image for {template_name} at {pos}")
            continue
            
        image = cv2.imread(square_path)
        if image is None:
            print(f"Error: Could not read image {square_path}")
            continue
        
        # Process the template
        template = process_template(image)
        
        # Save with descriptive name
        output_path = f'templates/{template_name}.png'
        cv2.imwrite(output_path, template)
        print(f"Processed template {template_name} from position {pos}")
    
    print("\nTemplate processing completed!")
    print("Created the following templates:")
    for template_name in sorted(piece_positions.keys()):
        print(f"  templates/{template_name}.png")

if __name__ == "__main__":
    main()