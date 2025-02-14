import cv2
import numpy as np
import os
from typing import Dict, List, Tuple

def process_template(image: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Process a template image, maintaining piece detail while removing background.
    
    Args:
        image: Input piece image
        target_size: Target size for resizing (default 64x64)
    
    Returns:
        Processed template image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crop 2 pixels from each edge to remove artifacts
    h, w = image.shape
    image = image[2:h-2, 2:w-2]
    
    # Sample background from top right corner (5x5 region)
    bg_sample = image[0:5, -5:]
    bg_color = int(np.median(bg_sample))
    
    # Create a mask for the piece based on background
    if bg_color > 127:  # Light square
        # For light squares, piece pixels are darker
        piece_mask = image < (bg_color - 30)
    else:  # Dark square
        # For dark squares, piece pixels are lighter
        piece_mask = image > (bg_color + 30)
    
    # Create clean image with piece on black background
    clean_image = np.zeros_like(image)
    piece_color = 255  # White piece on black background
    clean_image[piece_mask] = piece_color
    
    # Simple resize while maintaining aspect ratio
    h, w = clean_image.shape
    scale = min(target_size[0]/w, target_size[1]/h) * 0.8  # 80% of available space
    
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
    
    return output

def create_empty_square_template(is_light: bool = True, size: tuple = (64, 64)) -> np.ndarray:
    """
    Create a template for an empty square with proper intensity.
    
    Args:
        is_light: Whether this is a light square
        size: Size of template
    
    Returns:
        Square template with appropriate intensity
    """
    intensity = 192 if is_light else 64  # Distinct intensities for light/dark squares
    return np.full(size, intensity, dtype=np.uint8)

def main():
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Process starting position squares for each piece type
    piece_positions = {
        # White pieces (from rank 1 and 2)
        'white_king': 'e1',
        'white_queen': 'd1',
        'white_rook': 'h1',    
        'white_bishop': 'f1',   
        'white_knight': 'g1',   
        'white_pawn': 'd2',     
        # Black pieces (from rank 7 and 8)
        'black_king': 'e8',
        'black_queen': 'd8',
        'black_rook': 'h8',     
        'black_bishop': 'f8',   
        'black_knight': 'g8',   
        'black_pawn': 'd7'
    }
    
    # Process each piece
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
        
        try:
            # Process the template
            template = process_template(image)
            
            # Save with descriptive name
            output_path = f'templates/{template_name}.png'
            cv2.imwrite(output_path, template)
            print(f"Processed template {template_name} from position {pos}")
            
        except Exception as e:
            print(f"Error processing {template_name}: {e}")
    
    # Create empty square templates with distinct intensities
    print("\nCreating empty square templates...")
    cv2.imwrite('templates/empty_light.png', create_empty_square_template(True))
    cv2.imwrite('templates/empty_dark.png', create_empty_square_template(False))
    
    print("\nTemplate processing completed!")

if __name__ == "__main__":
    main()