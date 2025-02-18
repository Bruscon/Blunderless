import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Callable, Tuple
from pathlib import Path
import chess
import logging
from dataclasses import dataclass
from board_state import BoardState
from file_manager import FileManager
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
            self.root = root
            self.config = config or GuiConfig()
            self.piece_images: Dict[str, tk.PhotoImage] = {}
            self.auto_capture_active = False
            self.detecting_white_on_bottom = True

            # Initialize components
            self.board_state = BoardState()
            self.file_manager = FileManager()
            self.move_callback: Optional[Callable] = None
            
            # Setup GUI first
            self._setup_gui()
            
            # Then initialize double buffer
            self._setup_double_buffer()
            
            self._load_piece_images()
            self._setup_bindings()
            self.update_display()


    def _setup_double_buffer(self):
        """Initialize the off-screen buffer for double buffering"""
        # Create buffer with extra space for labels
        buffer_width = self.config.SQUARE_SIZE * 8 + 40
        buffer_height = self.config.SQUARE_SIZE * 8 + 40
        self.buffer_image = Image.new('RGB', (buffer_width, buffer_height), 'white')
        self.buffer_photo = None
        
        # Configure canvas for buffer display
        self.canvas.config(width=buffer_width, height=buffer_height)
        
        # Create buffer image on canvas
        self.buffer_id = self.canvas.create_image(
            buffer_width//2, buffer_height//2,
            anchor='center',
            tags='buffer'
        )

    def _toggle_auto_capture(self):
        """Toggle auto-capture state and update button text"""
        self.auto_capture_active = not self.auto_capture_active
        if self.auto_capture_active:
            self.auto_capture_button.config(text="Auto-Capture")
            self.set_status("Auto-capture enabled")
            if hasattr(self, 'auto_capture_callback'):
                self.auto_capture_callback(True)
        else:
            self.auto_capture_button.config(text="Auto-Capture")
            self.set_status("Auto-capture disabled")
            if hasattr(self, 'auto_capture_callback'):
                self.auto_capture_callback(False)

    def set_auto_capture_callback(self, callback):
        """Set callback for auto-capture toggle"""
        self.auto_capture_callback = callback

    def _load_piece_images(self) -> None:
        """Load piece images"""
        try:
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
            
            piece_size = int(self.config.SQUARE_SIZE * self.config.PIECE_SCALE)
            
            for symbol, filename in piece_files.items():
                image_path = self.config.PIECES_DIR / filename
                if image_path.exists():
                    # Load with PIL preserving transparency
                    img = Image.open(str(image_path)).convert('RGBA')
                    img = img.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
                    self.piece_images[symbol] = ImageTk.PhotoImage(img)
                else:
                    logger.warning(f"Piece image not found: {image_path}")
                    
        except Exception as e:
            logger.error(f"Error loading piece images: {e}")
            self._setup_unicode_fallback()

    def _draw_pieces_to_buffer(self, draw: ImageDraw.Draw):
        """Draw pieces to buffer"""
        board_offset = 20
        
        for square in chess.SQUARES:
            piece = self.board_state.get_piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                
                if self.detecting_white_on_bottom:
                    x = file * self.config.SQUARE_SIZE + board_offset
                    y = (7 - rank) * self.config.SQUARE_SIZE + board_offset
                else:
                    x = (7 - file) * self.config.SQUARE_SIZE + board_offset
                    y = rank * self.config.SQUARE_SIZE + board_offset
                
                if hasattr(self, 'use_unicode'):
                    # Unicode pieces
                    draw.text(
                        (x + self.config.SQUARE_SIZE//2, y + self.config.SQUARE_SIZE//2),
                        self.UNICODE_PIECES[piece.symbol()],
                        font=self._get_font(36),
                        fill="white" if piece.color else "#666666",
                        anchor="mm"
                    )
                else:
                    # Image pieces
                    piece_img = self.piece_images.get(piece.symbol())
                    if piece_img:
                        # Convert piece PhotoImage back to PIL for pasting
                        piece_photo = ImageTk.getimage(piece_img)
                        x_pos = x + (self.config.SQUARE_SIZE - piece_photo.width) // 2
                        y_pos = y + (self.config.SQUARE_SIZE - piece_photo.height) // 2
                        self.buffer_image.paste(piece_photo, (x_pos, y_pos), piece_photo)


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
            width=self.config.SQUARE_SIZE * 8 + 40,
            height=self.config.SQUARE_SIZE * 8 + 40,
            background='white'  # Add this
        )
        self.canvas.pack(side='left', padx=5, pady=5)
        
        # Control panel
        self.control_panel = self._setup_control_panel()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')

    def _setup_control_panel(self) -> ttk.Frame:
        panel = ttk.Frame(self.main_frame)
        panel.pack(side='right', fill='y', padx=5)
        
        # Buttons
        ttk.Button(panel, text="New Game", command=self._new_game).pack(pady=5)
        ttk.Button(panel, text="Undo Move", command=self._undo_move).pack(pady=5)
        ttk.Button(panel, text="Save Position", command=self._save_position).pack(pady=5)
        # Detection orientation button
        self.detect_orientation_button = ttk.Button(
            panel, 
            text="Detecting White Bottom", 
            command=self._toggle_detection_orientation
        )
        self.detect_orientation_button.pack(pady=5)

        # Auto-capture toggle button
        self.auto_capture_button = ttk.Button(panel, text="Auto-Capture", command=self._toggle_auto_capture)
        self.auto_capture_button.pack(pady=5)
        
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
        rank = chess.square_rank(square)
        return (file * self.config.SQUARE_SIZE, (7 - rank) * self.config.SQUARE_SIZE)

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


    def _toggle_detection_orientation(self):
        """Toggle whether we're detecting a board with white or black on bottom"""
        if hasattr(self, 'detection_orientation_callback'):
            # Get current state from button text
            is_currently_white = "White" in self.detect_orientation_button['text']
            # Toggle it
            new_is_white = not is_currently_white
            # Update button text
            self.detect_orientation_button['text'] = f"Detecting {'White' if new_is_white else 'Black'} Bottom"
            # Update local state
            self.detecting_white_on_bottom = new_is_white
            # Call the callback
            self.detection_orientation_callback(new_is_white)
            self.set_status(f"Now detecting board with {'White' if new_is_white else 'Black'} on bottom")

    def set_detection_orientation_callback(self, callback):
        """Set callback for detection orientation changes"""
        self.detection_orientation_callback = callback

    def _draw_board_to_buffer(self, draw: ImageDraw.Draw):
        """Draw board squares and grid to buffer"""
        board_offset = 20  # Space for labels
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                x1 = col * self.config.SQUARE_SIZE + board_offset
                y1 = row * self.config.SQUARE_SIZE + board_offset
                x2 = x1 + self.config.SQUARE_SIZE
                y2 = y1 + self.config.SQUARE_SIZE
                
                # Get square color
                if self.detecting_white_on_bottom:
                    square = chess.square(col, 7 - row)
                else:
                    square = chess.square(7 - col, row)
                
                control = self.board_state.calculate_square_control(square)
                color = self._modify_color_for_control("#E0E0E0", control)
                
                # Draw square
                draw.rectangle([x1, y1, x2, y2], fill=color, outline="#808080")
                
                # Draw control number if needed
                net_control = control.white_control - control.black_control
                if net_control != 0:
                    draw.text((x2 - 5, y1 + 5), f"{net_control:+d}", 
                            font=self._get_font(10), fill="black", anchor="rt")
        
        # Draw coordinates
        font = self._get_font(12)
        for i in range(8):
            # Rank numbers
            y = i * self.config.SQUARE_SIZE + board_offset + self.config.SQUARE_SIZE//2
            rank = 8 - i if self.detecting_white_on_bottom else i + 1
            draw.text((10, y), str(rank), font=font, fill="black", anchor="rm")
            draw.text((board_offset + 8 * self.config.SQUARE_SIZE + 10, y), 
                     str(rank), font=font, fill="black", anchor="lm")
            
            # File letters
            x = i * self.config.SQUARE_SIZE + board_offset + self.config.SQUARE_SIZE//2
            file = chr(97 + i) if self.detecting_white_on_bottom else chr(97 + (7-i))
            draw.text((x, 10), file, font=font, fill="black", anchor="ms")
            draw.text((x, board_offset + 8 * self.config.SQUARE_SIZE + 10), 
                     file, font=font, fill="black", anchor="ms")


    def _draw_coordinates(self):
        """Draw rank numbers and file letters"""
        # Add rank numbers (1-8)
        for rank in range(8):
            y = (7 - rank) * self.config.SQUARE_SIZE + self.config.SQUARE_SIZE // 2
            # Use correct rank number based on orientation
            rank_num = rank + 1 if self.detecting_white_on_bottom else 8 - rank
            # Add rank number on both sides of the board
            self.canvas.create_text(
                -10, y,
                text=str(rank_num),
                font=("Arial", 12),
                fill="black",
                anchor="e",
                tags="label"
            )
            self.canvas.create_text(
                self.config.SQUARE_SIZE * 8 + 10, y,
                text=str(rank_num),
                font=("Arial", 12),
                fill="black",
                anchor="w",
                tags="label"
            )

        # Add file letters (a-h)
        for file in range(8):
            x = file * self.config.SQUARE_SIZE + self.config.SQUARE_SIZE // 2
            # Use correct file letter based on orientation
            file_letter = chr(97 + file) if self.detecting_white_on_bottom else chr(97 + (7-file))
            # Add file letter on both top and bottom of the board
            self.canvas.create_text(
                x, -10,
                text=file_letter,
                font=("Arial", 12),
                fill="black",
                anchor="s",
                tags="label"
            )
            self.canvas.create_text(
                x, self.config.SQUARE_SIZE * 8 + 10,
                text=file_letter,
                font=("Arial", 12),
                fill="black",
                anchor="n",
                tags="label"
            )

    def draw_pieces(self) -> None:
        """Draw the chess pieces on the board"""
        # No need to delete "piece" tag items as canvas.delete("all") in draw_board cleared everything
        for square in chess.SQUARES:
            piece = self.board_state.get_piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                                
                if self.detecting_white_on_bottom:
                    x = file * self.config.SQUARE_SIZE
                    y = (7 - rank) * self.config.SQUARE_SIZE
                else:
                    x = (7 - file) * self.config.SQUARE_SIZE
                    y = rank * self.config.SQUARE_SIZE
                
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
        """Update display using double buffering"""
        # Create a new buffer image 
        buffer_width = self.config.SQUARE_SIZE * 8 + 40
        buffer_height = self.config.SQUARE_SIZE * 8 + 40
        self.buffer_image = Image.new('RGBA', (buffer_width, buffer_height), 'white')
        draw = ImageDraw.Draw(self.buffer_image)

        # Draw everything to buffer
        self._draw_board_to_buffer(draw)
        self._draw_pieces_to_buffer(draw)
        
        # Convert buffer to PhotoImage only once
        self.buffer_photo = ImageTk.PhotoImage(self.buffer_image)
        
        # Update canvas with single operation
        self.canvas.delete('all')
        self.canvas.create_image(buffer_width//2, buffer_height//2, 
                               image=self.buffer_photo, anchor='center')
        
        # Update status
        self.update_status()
        self.update_info()

    def _draw_to_buffer(self):
        """Draw all chess elements to the off-screen buffer"""
        # Create a drawing context for the buffer
        draw = ImageDraw.Draw(self.buffer_image)
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                x1 = col * self.config.SQUARE_SIZE + 20  # Add offset for labels
                y1 = row * self.config.SQUARE_SIZE + 20
                x2 = x1 + self.config.SQUARE_SIZE
                y2 = y1 + self.config.SQUARE_SIZE
                
                # Get square color with control visualization
                if self.detecting_white_on_bottom:
                    square = chess.square(col, 7 - row)
                else:
                    square = chess.square(7 - col, row)
                    
                control = self.board_state.calculate_square_control(square)
                base_color = "#E0E0E0"
                color = self._modify_color_for_control(base_color, control)
                
                # Draw square
                draw.rectangle([x1, y1, x2, y2], fill=color, outline="#808080")
                
                # Draw control numbers if needed
                net_control = control.white_control - control.black_control
                if net_control != 0:
                    text = f"{net_control:+d}"
                    draw.text((x2 - 10, y1 + 5), text, 
                            fill="black", anchor="rt",
                            font=self._get_font(10))
        
        # Draw grid lines
        for i in range(9):
            coord = i * self.config.SQUARE_SIZE + 20
            # Horizontal
            draw.line([(20, coord), 
                      (self.config.SQUARE_SIZE * 8 + 20, coord)], 
                     fill="#808080")
            # Vertical
            draw.line([(coord, 20), 
                      (coord, self.config.SQUARE_SIZE * 8 + 20)], 
                     fill="#808080")
        
        # Draw coordinates
        self._draw_coordinates_to_buffer(draw)
        
        # Draw pieces
        self._draw_pieces_to_buffer(draw)

    def _get_font(self, size: int):
        """Helper to get PIL ImageFont"""
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

    def _draw_coordinates_to_buffer(self, draw: ImageDraw.Draw):
        """Draw rank numbers and file letters to buffer"""
        font = self._get_font(12)
        
        # Ranks
        for rank in range(8):
            y = (7 - rank) * self.config.SQUARE_SIZE + self.config.SQUARE_SIZE//2 + 20
            rank_num = rank + 1 if self.detecting_white_on_bottom else 8 - rank
            # Left side
            draw.text((10, y), str(rank_num), fill="black", anchor="rm", font=font)
            # Right side
            draw.text((self.config.SQUARE_SIZE * 8 + 30, y), 
                     str(rank_num), fill="black", anchor="lm", font=font)
        
        # Files
        for file in range(8):
            x = file * self.config.SQUARE_SIZE + self.config.SQUARE_SIZE//2 + 20
            file_letter = chr(97 + file) if self.detecting_white_on_bottom else chr(97 + (7-file))
            # Top
            draw.text((x, 10), file_letter, fill="black", anchor="ms", font=font)
            # Bottom
            draw.text((x, self.config.SQUARE_SIZE * 8 + 30), 
                     file_letter, fill="black", anchor="ms", font=font)


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
