import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Callable, Tuple
from pathlib import Path
import chess
import logging
from dataclasses import dataclass
from board_state import BoardState
from file_manager import FileManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GuiConfig:
    """Configuration for GUI appearance and behavior"""
    SQUARE_SIZE: int = 64
    LIGHT_SQUARE: str = "#DDB88C"
    DARK_SQUARE: str = "#A66D4F"
    HIGHLIGHT_COLOR: str = "#ffff00"
    MOVE_HIGHLIGHT_COLOR: str = "#00ff00"
    LAST_MOVE_COLOR: str = "#aaaaff"
    PIECE_FONT: str = "Arial"
    PIECE_SIZE: int = 36

    # Image configuration
    PIECE_SCALE: float = 0.8  # Scale factor for piece images relative to square size
    PIECES_DIR: Path = Path("pieces")  # Directory containing piece images


    # Control visualization colors
    WHITE_CONTROL_COLOR: str = "#00ff0033"  # Green with alpha
    BLACK_CONTROL_COLOR: str = "#ff000033"  # Red with alpha
    CONTESTED_COLOR: str = "#80008033"      # Purple with alpha

class ChessGUI:
    """
    Handles chess board visualization and user interaction.
    This class is responsible for:
    - Drawing the board and pieces
    - Handling user input
    - Visual feedback (highlights, move indicators)
    - Status updates
    """

    def __init__(self, root: tk.Tk, config: Optional[GuiConfig] = None):
        """
        Initialize the chess GUI
        
        Args:
            root: Tkinter root window
            config: Optional GUI configuration
        """
        self.root = root
        self.config = config or GuiConfig()
        self.piece_images: Dict[str, tk.PhotoImage] = {}
        
        # Initialize components
        self.board_state = BoardState()
        self.file_manager = FileManager()
        self.move_callback: Optional[Callable] = None
        
        self._setup_gui()
        self._load_piece_images()
        self._setup_bindings()
        self.update_display()

    def _load_piece_images(self) -> None:
        """Load piece images from the pieces directory"""
        try:
            # Map of piece symbols to filenames using the dt/lt naming convention
            piece_files = {
                'P': 'Chess_plt45.svg.png',  # white pawn
                'N': 'Chess_nlt45.svg.png',  # white knight
                'B': 'Chess_blt45.svg.png',  # white bishop
                'R': 'Chess_rlt45.svg.png',  # white rook
                'Q': 'Chess_qlt45.svg.png',  # white queen
                'K': 'Chess_klt45.svg.png',  # white king
                'p': 'Chess_pdt45.svg.png',  # black pawn
                'n': 'Chess_ndt45.svg.png',  # black knight
                'b': 'Chess_bdt45.svg.png',  # black bishop
                'r': 'Chess_rdt45.svg.png',  # black rook
                'q': 'Chess_qdt45.svg.png',  # black queen
                'k': 'Chess_kdt45.svg.png'   # black king
            }
            
            # Calculate piece size based on square size and scale factor
            piece_size = int(self.config.SQUARE_SIZE * self.config.PIECE_SCALE)
            
            # Load each piece image
            for symbol, filename in piece_files.items():
                image_path = self.config.PIECES_DIR / filename
                if image_path.exists():
                    # Load and resize the image
                    image = tk.PhotoImage(file=str(image_path))
                    # Note: subsample or zoom might be needed depending on original image size
                    self.piece_images[symbol] = image
                else:
                    logger.warning(f"Piece image not found: {image_path}")
                    
        except Exception as e:
            logger.error(f"Error loading piece images: {e}")
            # Fall back to Unicode pieces if image loading fails
            self._setup_unicode_fallback()


    def _setup_unicode_fallback(self) -> None:
        """Set up Unicode pieces as fallback if images fail to load"""
        self.use_unicode = True
        self.UNICODE_PIECES = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }

    def _setup_gui(self) -> None:
        """Set up the GUI components"""
        self.root.title("Chess GUI")
        
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Chess board canvas
        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.config.SQUARE_SIZE * 8,
            height=self.config.SQUARE_SIZE * 8
        )
        self.canvas.pack(side='left', padx=5, pady=5)
        
        # Control panel
        self.control_panel = self._setup_control_panel()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')

    def _setup_control_panel(self) -> ttk.Frame:
        """Set up the control panel with buttons and info"""
        panel = ttk.Frame(self.main_frame)
        panel.pack(side='right', fill='y', padx=5)
        
        # Buttons
        ttk.Button(panel, text="New Game", command=self._new_game).pack(pady=5)
        ttk.Button(panel, text="Undo Move", command=self._undo_move).pack(pady=5)
        ttk.Button(panel, text="Save Position", command=self._save_position).pack(pady=5)
        
        # Game info
        self.info_frame = ttk.LabelFrame(panel, text="Game Info")
        self.info_frame.pack(pady=10, fill='x')
        
        self.move_label = ttk.Label(self.info_frame, text="Move: 1")
        self.move_label.pack(pady=2)
        
        self.turn_label = ttk.Label(self.info_frame, text="Turn: White")
        self.turn_label.pack(pady=2)
        
        return panel

    def _setup_bindings(self) -> None:
        """Set up event bindings"""
        self.canvas.bind('<Button-1>', self._on_square_clicked)
        self.root.bind('<Control-z>', lambda e: self._undo_move())
        self.root.bind('<Escape>', lambda e: self._deselect())

    def _get_square_from_coords(self, x: int, y: int) -> int:
        """Convert canvas coordinates to chess square"""
        file = x // self.config.SQUARE_SIZE
        rank = 7 - (y // self.config.SQUARE_SIZE)
        return chess.square(file, rank)

    def _get_coords_from_square(self, square: int) -> Tuple[int, int]:
        """Convert chess square to canvas coordinates"""
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        return (file * self.config.SQUARE_SIZE, rank * self.config.SQUARE_SIZE)

    def _on_square_clicked(self, event) -> None:
        """Handle mouse clicks on the board"""
        square = self._get_square_from_coords(event.x, event.y)
        
        if self.board_state.selected_square is None:
            # First click - select piece if present
            piece = self.board_state.get_piece_at(square)
            if piece:
                if piece.color == self.board_state.get_side_to_move():
                    self.board_state.selected_square = square
                    self._highlight_square(square)
                    self._highlight_legal_moves(square)
                else:
                    self.set_status("Wrong color to move")
        else:
            # Second click - attempt move
            move = chess.Move(self.board_state.selected_square, square)
            if self.board_state.make_move(move):
                self._handle_successful_move(move)
            else:
                self._handle_failed_move()

    def _handle_successful_move(self, move: chess.Move) -> None:
        """Handle successful move completion"""
        self._save_position()
        self.update_display()
        self._deselect()
        
        if self.move_callback:
            self.move_callback(move)

    def _handle_failed_move(self) -> None:
        """Handle failed move attempt"""
        self.set_status("Illegal move!")
        self._deselect()

    def _highlight_square(self, square: int) -> None:
        """Highlight a single square"""
        x, y = self._get_coords_from_square(square)
        self.canvas.create_rectangle(
            x, y,
            x + self.config.SQUARE_SIZE,
            y + self.config.SQUARE_SIZE,
            outline=self.config.HIGHLIGHT_COLOR,
            width=2,
            tags="highlight"
        )

    def _highlight_legal_moves(self, square: int) -> None:
        """Highlight squares for legal moves from selected square"""
        for move in self.board_state.get_legal_moves(square):
            x, y = self._get_coords_from_square(move.to_square)
            self.canvas.create_oval(
                x + self.config.SQUARE_SIZE // 4,
                y + self.config.SQUARE_SIZE // 4,
                x + 3 * self.config.SQUARE_SIZE // 4,
                y + 3 * self.config.SQUARE_SIZE // 4,
                fill=self.config.MOVE_HIGHLIGHT_COLOR,
                tags="highlight"
            )

    def _deselect(self) -> None:
        """Clear selection and highlights"""
        self.board_state.selected_square = None
        self.canvas.delete("highlight")

    def draw_board(self) -> None:
        """Draw the chess board with grid lines"""
        # First draw base squares
        for row in range(8):
            for col in range(8):
                x1 = col * self.config.SQUARE_SIZE
                y1 = row * self.config.SQUARE_SIZE
                x2 = x1 + self.config.SQUARE_SIZE
                y2 = y1 + self.config.SQUARE_SIZE
                
                # Use light gray as base color for better piece visibility
                base_color = "#E0E0E0"  # Light gray background
                
                # Apply control visualization
                square = chess.square(col, 7 - row)
                control = self.board_state.calculate_square_control(square)
                color = self._modify_color_for_control(base_color, control)
                
                # Draw square with color but no outline
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        
        # Then draw grid lines
        for i in range(9):  # Draw 9 lines to create 8 squares
            # Calculate coordinates
            coord = i * self.config.SQUARE_SIZE
            
            # Draw horizontal line
            self.canvas.create_line(
                0, coord,  # Start point
                self.config.SQUARE_SIZE * 8, coord,  # End point
                fill="#808080",  # Gray color
                width=1
            )
            
            # Draw vertical line
            self.canvas.create_line(
                coord, 0,  # Start point
                coord, self.config.SQUARE_SIZE * 8,  # End point
                fill="#808080",  # Gray color
                width=1
            )

    def draw_pieces(self) -> None:
        """Draw the chess pieces on the board"""
        self.canvas.delete("piece")
        for square in chess.SQUARES:
            piece = self.board_state.get_piece_at(square)
            if piece:
                x = chess.square_file(square) * self.config.SQUARE_SIZE
                y = (7 - chess.square_rank(square)) * self.config.SQUARE_SIZE
                
                if hasattr(self, 'use_unicode'):
                    # Fallback to Unicode pieces
                    self.canvas.create_text(
                        x + self.config.SQUARE_SIZE // 2,
                        y + self.config.SQUARE_SIZE // 2,
                        text=self.UNICODE_PIECES[piece.symbol()],
                        font=("Arial", 36),
                        fill="white" if piece.color else "#666666",
                        tags="piece"
                    )
                else:
                    # Use piece images
                    image = self.piece_images.get(piece.symbol())
                    if image:
                        self.canvas.create_image(
                            x + self.config.SQUARE_SIZE // 2,
                            y + self.config.SQUARE_SIZE // 2,
                            image=image,
                            tags="piece"
                        )

    def update_display(self) -> None:
        """Update the entire display"""
        self.draw_board()
        self.draw_pieces()
        self.update_status()
        self.update_info()

    def update_status(self) -> None:
        """Update the status bar with game state"""
        status = self.board_state.get_game_status()
        self.set_status(status)

    def update_info(self) -> None:
        """Update game info display"""
        self.move_label.config(text=f"Move: {self.board_state.get_move_number()}")
        self.turn_label.config(text=f"Turn: {'White' if self.board_state.get_side_to_move() else 'Black'}")

    def set_status(self, message: str) -> None:
        """Set status bar message"""
        self.status_bar.config(text=message)

    def set_move_callback(self, callback: Callable) -> None:
        """Set callback for move completion"""
        self.move_callback = callback

    def _new_game(self) -> None:
        """Start a new game"""
        self.board_state = BoardState()
        self._deselect()
        self.update_display()
        self.set_status("New game started")

    def _undo_move(self) -> None:
        """Undo the last move"""
        if self.board_state.undo_move():
            self._deselect()
            self.update_display()
            self.set_status("Move undone")
        else:
            self.set_status("No moves to undo")

    def _save_position(self) -> None:
        """Save the current position"""
        try:
            self.file_manager.save_position(
                self.board_state.board,
                self.board_state.get_move_number()
            )
            self.set_status("Position saved")
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            self.set_status("Error saving position")

    def _modify_color_for_control(self, base_color: str, control) -> str:
        """
        Modify square color based on control information with net control calculation.
        Returns base_color for uncontrolled squares or equally contested squares.
        Returns green for net white control, red for net black control.
        """
        # Calculate net control (positive = white advantage, negative = black advantage)
        net_control = control.white_control - control.black_control
        value = max(250 - abs(net_control)*50, 0)
        
        if net_control == 0:
            # Equal control or uncontrolled - use base color
            return base_color
        elif net_control > 0:
            # White has more control

            return f"#{value:02x}FF{value:02x}"
        else:
            # Black has more control
            return f"#FF{value:02x}{value:02x}"

def main():
    """Main entry point for running the chess GUI"""
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
