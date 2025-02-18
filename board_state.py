from typing import Optional, List, Set
import chess
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SquareControl:
    """Represents the control of a square by both players"""
    white_control: int
    black_control: int
    
    @property
    def is_contested(self) -> bool:
        """A square is only contested if BOTH colors are attacking it"""
        return self.white_control > 0 and self.black_control > 0

class BoardState:
    """
    Handles chess game state, move validation, and game logic.
    This class is responsible for:
    - Maintaining the current board position
    - Validating and making moves
    - Tracking move history
    - Calculating square control
    - Providing game state information
    """
    
    def __init__(self):
        self.board = chess.Board()
        self.move_history: List[chess.Move] = []
        self._selected_square: Optional[int] = None
            
    @property
    def selected_square(self) -> Optional[int]:
        """Currently selected square, if any"""
        return self._selected_square
    
    @selected_square.setter
    def selected_square(self, square: Optional[int]):
        """Set or clear the selected square"""
        self._selected_square = square
        
    def make_move(self, move: chess.Move) -> bool:
        """
        Attempt to make a move on the board
        
        Args:
            move: The move to make
            
        Returns:
            bool: True if move was legal and made, False otherwise
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            logger.info(f"Move made: {move}")
            return True
        logger.warning(f"Illegal move attempted: {move}")
        return False
    
    def undo_move(self) -> Optional[chess.Move]:
        """
        Undo the last move if possible
        
        Returns:
            Optional[chess.Move]: The move that was undone, or None if no moves to undo
        """
        if self.move_history:
            self.board.pop()
            undone_move = self.move_history.pop()
            logger.info(f"Move undone: {undone_move}")
            return undone_move
        logger.info("No moves to undo")
        return None
    
    def get_legal_moves(self, square: int) -> Set[chess.Move]:
        """
        Get all legal moves from a given square
        
        Args:
            square: The square to get moves from
            
        Returns:
            Set[chess.Move]: Set of legal moves from this square
        """
        return {move for move in self.board.legal_moves 
                if move.from_square == square}
    
    def calculate_square_control(self, square: int) -> SquareControl:
        """
        Calculate how many pieces of each color attack a square
        
        Args:
            square: The square to analyze
            
        Returns:
            SquareControl: Control information for the square
        """
        try:
            white_control = len(list(self.board.attackers(chess.WHITE, square)))
            black_control = len(list(self.board.attackers(chess.BLACK, square)))
            
                
            return SquareControl(white_control, black_control)
        except Exception as e:
            logger.error(f"Error calculating square control: {e}")
            return SquareControl(0, 0)
    
    def get_game_status(self) -> str:
        """
        Get the current game status as a string
        
        Returns:
            str: Description of current game state
        """
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate - {winner} wins!"
        elif self.board.is_stalemate():
            return "Stalemate!"
        elif self.board.is_insufficient_material():
            return "Draw - Insufficient material"
        elif self.board.is_check():
            return "Check!"
        else:
            return "White to move" if self.board.turn == chess.WHITE else "Black to move"
    
    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """
        Get the piece at a given square
        
        Args:
            square: The square to check
            
        Returns:
            Optional[chess.Piece]: The piece at that square, or None if empty
        """
        return self.board.piece_at(square)
    
    def get_fen(self) -> str:
        """
        Get the current position in FEN notation
        
        Returns:
            str: FEN string of current position
        """
        return self.board.fen()
    
    def get_move_number(self) -> int:
        """
        Get the current move number
        
        Returns:
            int: Current move number
        """
        return self.board.fullmove_number
    
    def get_side_to_move(self) -> chess.Color:
        """
        Get the color to move
        
        Returns:
            chess.Color: Color that has the move
        """
        return self.board.turn

    def __str__(self) -> str:
        """String representation of the current board position"""
        return str(self.board)
