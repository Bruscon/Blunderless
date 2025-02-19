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
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GuiConfig:
    """Configuration for GUI appearance and behavior"""
    SQUARE_SIZE: int = 64
    BOARD_PADDING: int = 20  # Space for labels
    PIECE_SCALE: float = 0.8  # Scale factor for piece images
    PIECES_DIR: Path = Path("pieces")

class ChessGUI:
    """
    Handles chess board visualization and user interaction with proper double buffering.
    """
    def __init__(self, root: tk.Tk, config: Optional[GuiConfig] = None):
        self.root = root
        self.config = config or GuiConfig()
        self.board_state = BoardState()
        self.file_manager = FileManager()
        
        # GUI state
        self.piece_images: Dict[str, ImageTk.PhotoImage] = {}
        self.detecting_white_on_bottom = True
        self.auto_capture_active = False
        self.buffer_lock = threading.Lock()
        
        # Callbacks
        self.move_callback: Optional[Callable] = None
        self.auto_capture_callback: Optional[Callable] = None
        self.detection_orientation_callback: Optional[Callable] = None
        
        # Setup GUI
        self._setup_gui()
        self._setup_double_buffer()
        self._load_piece_images()
        self._setup_bindings()
        
        # Initial display
        self.update_display()

    def _setup_gui(self) -> None:
        """Initialize GUI components"""
        self.root.title("Chess GUI")
        
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Chess board canvas - size includes padding for labels
        canvas_size = self.config.SQUARE_SIZE * 8 + 2 * self.config.BOARD_PADDING
        self.canvas = tk.Canvas(
            self.main_frame,
            width=canvas_size,
            height=canvas_size,
            background='white'
        )
        self.canvas.pack(side='left', padx=5, pady=5)
        
        # Control panel
        self._setup_control_panel()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')

    def _setup_double_buffer(self) -> None:
        """Initialize off-screen buffer for double buffering"""
        buffer_size = self.config.SQUARE_SIZE * 8 + 2 * self.config.BOARD_PADDING
        self.buffer_image = Image.new('RGB', (buffer_size, buffer_size), 'white')
        self.buffer_draw = ImageDraw.Draw(self.buffer_image)
        self.buffer_photo = None

    def _setup_control_panel(self) -> None:
        """Initialize control panel with buttons and info display"""
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
        
        # Auto-capture button
        self.auto_capture_button = ttk.Button(
            panel,
            text="Auto-Capture",
            command=self._toggle_auto_capture
        )
        self.auto_capture_button.pack(pady=5)
        
        # Game info frame
        self.info_frame = ttk.LabelFrame(panel, text="Game Info")
        self.info_frame.pack(pady=10, fill='x')
        
        self.move_label = ttk.Label(self.info_frame, text="Move: 1")
        self.move_label.pack(pady=2)
        
        self.turn_label = ttk.Label(self.info_frame, text="Turn: White")
        self.turn_label.pack(pady=2)

    def _load_piece_images(self) -> None:
        """Load and scale piece images"""
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
            try:
                image_path = self.config.PIECES_DIR / filename
                if image_path.exists():
                    img = Image.open(str(image_path)).convert('RGBA')
                    img = img.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
                    self.piece_images[symbol] = ImageTk.PhotoImage(img)
                else:
                    logger.warning(f"Piece image not found: {image_path}")
            except Exception as e:
                logger.error(f"Error loading piece image {filename}: {e}")

    def _setup_bindings(self) -> None:
        """Set up event bindings"""
        self.canvas.bind('<Button-1>', self._on_square_clicked)
        self.root.bind('<Control-z>', lambda e: self._undo_move())
        self.root.bind('<Escape>', lambda e: self._deselect())

    def update_display(self) -> None:
        """Update the display using double buffering with thread safety"""
        with self.buffer_lock:
            # Clear buffer
            self.buffer_image = Image.new('RGB', self.buffer_image.size, 'white')
            self.buffer_draw = ImageDraw.Draw(self.buffer_image)
            
            # Draw everything to buffer
            self._draw_board()
            self._draw_coordinates()
            self._draw_pieces()
            
            # Only update canvas when we have a complete frame
            self.buffer_photo = ImageTk.PhotoImage(self.buffer_image)
            
            # If this is our first frame, create the image
            if not hasattr(self, 'canvas_image_id'):
                self.canvas_image_id = self.canvas.create_image(
                    self.buffer_image.size[0]//2,
                    self.buffer_image.size[1]//2,
                    image=self.buffer_photo
                )
            # Otherwise, just update the existing image
            else:
                self.canvas.itemconfig(self.canvas_image_id, image=self.buffer_photo)
        
        # Update status and info
        self.update_status()
        self.update_info()

    def _draw_board(self) -> None:
        """Draw the chess board squares with control visualization"""
        for row in range(8):
            for col in range(8):
                # Calculate square position
                x1 = col * self.config.SQUARE_SIZE + self.config.BOARD_PADDING
                y1 = row * self.config.SQUARE_SIZE + self.config.BOARD_PADDING
                x2 = x1 + self.config.SQUARE_SIZE
                y2 = y1 + self.config.SQUARE_SIZE
                
                # Get square and its control information
                if self.detecting_white_on_bottom:
                    square = chess.square(col, 7 - row)
                else:
                    square = chess.square(7 - col, row)
                
                control = self.board_state.calculate_square_control(square)
                color = self._get_square_color(square, control)
                
                # Draw square
                self.buffer_draw.rectangle([x1, y1, x2, y2], fill=color, outline="#808080")
                
                # Draw control indicator if needed
                net_control = control.white_control - control.black_control
                if net_control != 0:
                    self.buffer_draw.text(
                        (x2 - 5, y1 + 5),
                        f"{net_control:+d}",
                        font=self._get_font(10),
                        fill="black",
                        anchor="rt"
                    )

    def _draw_coordinates(self) -> None:
        """Draw rank numbers and file letters"""
        font = self._get_font(12)
        
        for i in range(8):
            # Rank numbers
            y = i * self.config.SQUARE_SIZE + self.config.BOARD_PADDING + self.config.SQUARE_SIZE//2
            rank = 8 - i if self.detecting_white_on_bottom else i + 1
            
            # Draw rank numbers on both sides
            self.buffer_draw.text((10, y), str(rank), font=font, fill="black", anchor="rm")
            self.buffer_draw.text(
                (self.config.BOARD_PADDING + 8 * self.config.SQUARE_SIZE + 10, y),
                str(rank), font=font, fill="black", anchor="lm"
            )
            
            # File letters
            x = i * self.config.SQUARE_SIZE + self.config.BOARD_PADDING + self.config.SQUARE_SIZE//2
            file = chr(97 + i) if self.detecting_white_on_bottom else chr(97 + (7-i))
            
            # Draw file letters on top and bottom
            self.buffer_draw.text((x, 10), file, font=font, fill="black", anchor="ms")
            self.buffer_draw.text(
                (x, self.config.BOARD_PADDING + 8 * self.config.SQUARE_SIZE + 10),
                file, font=font, fill="black", anchor="ms"
            )

    def _draw_pieces(self) -> None:
        """Draw pieces on the board"""
        for square in chess.SQUARES:
            piece = self.board_state.get_piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                
                if self.detecting_white_on_bottom:
                    x = file * self.config.SQUARE_SIZE + self.config.BOARD_PADDING
                    y = (7 - rank) * self.config.SQUARE_SIZE + self.config.BOARD_PADDING
                else:
                    x = (7 - file) * self.config.SQUARE_SIZE + self.config.BOARD_PADDING
                    y = rank * self.config.SQUARE_SIZE + self.config.BOARD_PADDING
                
                piece_img = self.piece_images.get(piece.symbol())
                if piece_img:
                    # Convert PhotoImage back to PIL for pasting
                    piece_photo = ImageTk.getimage(piece_img)
                    x_pos = x + (self.config.SQUARE_SIZE - piece_photo.width) // 2
                    y_pos = y + (self.config.SQUARE_SIZE - piece_photo.height) // 2
                    self.buffer_image.paste(piece_photo, (x_pos, y_pos), piece_photo)

    def _get_square_color(self, square: int, control) -> str:
        """Get square color based on its position and control information"""
        net_control = control.white_control - control.black_control
        value = max(250 - abs(net_control)*50, 0)
        
        if net_control == 0:
            return "#E0E0E0"  # Neutral color
        elif net_control > 0:
            return f"#{value:02x}FF{value:02x}"  # Green tint
        else:
            return f"#FF{value:02x}{value:02x}"  # Red tint

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Get PIL ImageFont with fallback"""
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

    def _on_square_clicked(self, event: tk.Event) -> None:
        """Handle mouse clicks on the board"""
        # Convert click coordinates to square
        file = (event.x - self.config.BOARD_PADDING) // self.config.SQUARE_SIZE
        rank = 7 - (event.y - self.config.BOARD_PADDING) // self.config.SQUARE_SIZE
        
        if 0 <= file <= 7 and 0 <= rank <= 7:
            square = chess.square(file, rank)
            
            if self.board_state.selected_square is None:
                # First click - select piece
                piece = self.board_state.get_piece_at(square)
                if piece and piece.color == self.board_state.get_side_to_move():
                    self.board_state.selected_square = square
                    self.update_display()
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
        self.board_state.selected_square = None
        self.update_display()
        
        if self.move_callback:
            self.move_callback(move)

    def _handle_failed_move(self) -> None:
        """Handle failed move attempt"""
        self.set_status("Illegal move!")
        self.board_state.selected_square = None
        self.update_display()

    def _new_game(self) -> None:
        """Start a new game"""
        self.board_state = BoardState()
        self.board_state.selected_square = None
        self.update_display()
        self.set_status("New game started")

    def _undo_move(self) -> None:
        """Undo the last move"""
        if self.board_state.undo_move():
            self.board_state.selected_square = None
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

    def _toggle_auto_capture(self) -> None:
        """Toggle auto-capture state"""
        self.auto_capture_active = not self.auto_capture_active
        button_text = "Stop Auto-Capture" if self.auto_capture_active else "Start Auto-Capture"
        self.auto_capture_button.config(text=button_text)
        
        if self.auto_capture_callback:
            self.auto_capture_callback(self.auto_capture_active)
        
        status = "Auto-capture enabled" if self.auto_capture_active else "Auto-capture disabled"
        self.set_status(status)

    def _toggle_detection_orientation(self) -> None:
        """Toggle board detection orientation"""
        self.detecting_white_on_bottom = not self.detecting_white_on_bottom
        button_text = f"Detecting {'White' if self.detecting_white_on_bottom else 'Black'} Bottom"
        self.detect_orientation_button.config(text=button_text)
        
        if self.detection_orientation_callback:
            self.detection_orientation_callback(self.detecting_white_on_bottom)
        
        self.set_status(f"Now detecting board with {'White' if self.detecting_white_on_bottom else 'Black'} on bottom")

    def _deselect(self) -> None:
        """Clear current selection"""
        self.board_state.selected_square = None
        self.update_display()

    def update_status(self) -> None:
        """Update the status bar with game state"""
        status = self.board_state.get_game_status()
        self.set_status(status)

    def update_info(self) -> None:
        """Update game info display"""
        self.move_label.config(text=f"Move: {self.board_state.get_move_number()}")
        turn_text = "White" if self.board_state.get_side_to_move() else "Black"
        self.turn_label.config(text=f"Turn: {turn_text}")

    def set_status(self, message: str) -> None:
        """Set status bar message"""
        self.status_bar.config(text=message)

    # Callback setters
    def set_move_callback(self, callback: Callable) -> None:
        """Set callback for move completion"""
        self.move_callback = callback

    def set_auto_capture_callback(self, callback: Callable) -> None:
        """Set callback for auto-capture toggle"""
        self.auto_capture_callback = callback

    def set_detection_orientation_callback(self, callback: Callable) -> None:
        """Set callback for detection orientation changes"""
        self.detection_orientation_callback = callback