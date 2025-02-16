import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import subprocess
import chess
from piece_detector import PieceDetector
from board_detector import BoardDetector


def print_board(fen: str):
    """Print the chess board in a readable format using python-chess."""
    board = chess.Board(fen + " w KQkq - 0 1")  # Add required FEN fields
    print(board)
    print()


# Example usage:
if __name__ == "__main__":
    # Create detector instances
    board_detector = BoardDetector()

    # process the starting position screenshot to generate template images

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




    piece_detector = PieceDetector("templates", debug_level=3)
    
    # Process the chessboard image
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
    
    # Detect pieces on the board
    board_state = piece_detector.detect_board(squares)
    
    
    # Print the detected position
    print("\nDetected chess position:")
    print_board(board_state)
    
    print("Processing completed successfully!")
    print(f"Board image saved as 'images/extracted_board.png'")
