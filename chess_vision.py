import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import subprocess
from piece_detector import PieceDetector
from board_detector import BoardDetector


def print_board(board: List[List[Optional[str]]]):
    """Print the chess board in a readable format."""
    print("  a b c d e f g h")
    print("  ---------------")
    for rank_idx, rank in enumerate(board):
        rank_str = f"{8-rank_idx}|"
        for piece in rank:
            if piece is None:
                rank_str += ". "
            else:
                rank_str += f"{piece} "
        print(rank_str)
    print()


# Example usage:
if __name__ == "__main__":
    # Create detector instances
    board_detector = BoardDetector()

    # process the starting position screenshot to generate template images
    try:
        board_image, squares = board_detector.process_image("starting_position.png")
        
        # Save extracted board
        cv2.imwrite("images/extracted_board.png", board_image)
        
        # Save individual squares with rank and file information
        files = 'abcdefgh'
        for rank, file, square in squares:
            # Convert rank/file to chess notation (e.g., e4, f6)
            square_name = f"{files[file]}{8-rank}"
            cv2.imwrite(f"images/square_{square_name}.png", square)

        #run the template processor
        subprocess.run(["python", "template_processor.py"])
            
        print("Template processing completed successfully!")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


    piece_detector = PieceDetector("templates")
    
    # Process an image
    try:
        board_image, squares = board_detector.process_image("starting_position.png")
        
        # Save extracted board
        cv2.imwrite("images/extracted_board.png", board_image)
        
        # Save individual squares with rank and file information
        files = 'abcdefgh'
        for rank, file, square in squares:
            # Convert rank/file to chess notation (e.g., e4, f6)
            square_name = f"{files[file]}{8-rank}"
            cv2.imwrite(f"images/square_{square_name}.png", square)
            
        print("Board processing completed successfully!")
        print(f"Saved squares in the 'images' directory")

        # Extract just the square images from the tuples
        square_images = [square for _, _, square in squares]
        
        # Detect pieces on the board
        board_state = piece_detector.detect_board(square_images)
        
        # Validate the detection
        if not piece_detector.validate_detection(board_state):
            print("Warning: Detected position may not be valid!")
        
        # Print the detected position
        print("\nDetected chess position:")
        print_board(board_state)
        
        print("Processing completed successfully!")
        print(f"Board image saved as 'images/extracted_board.png'")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")