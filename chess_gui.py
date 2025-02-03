import tkinter as tk
import chess
import os
import time

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Chess GUI")
        
        # Initialize chess board
        self.board = chess.Board()
        
        # Move callback
        self.move_callback = None
        
        # Size constants
        self.SQUARE_SIZE = 64
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.BOARD_SIZE, height=self.BOARD_SIZE)
        self.canvas.pack()
        
        # Base colors for squares
        self.LIGHT_SQUARE = "#DDB88C"
        self.DARK_SQUARE = "#A66D4F"
        
        # Control colors with alpha values
        self.WHITE_CONTROL = "#00ff0033"  # Green with low alpha
        self.BLACK_CONTROL = "#ff000033"  # Red with low alpha
        self.BOTH_CONTROL = "#80008033"   # Purple with low alpha
        
        # Unicode chess pieces
        self.pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_square_clicked)
        self.selected_square = None
        
        # Draw initial board
        self.draw_board()
        self.draw_pieces()
        
        # Start update cycle
        self.update()
    
    def update(self):
        """Periodic update function"""
        try:
            self.draw_board()
            self.draw_pieces()
        except Exception as e:
            print(f"Error in update cycle: {e}")
        finally:
            # Schedule next update with a longer interval (e.g., 500ms instead of 100ms)
            self.root.after(100, self.update)
    
    def set_move_callback(self, callback):
        """Set a function to be called when a move is made"""
        self.move_callback = callback
    
    def draw_board(self):
        """Draw the chess board squares with control visualization"""
        for row in range(8):
            for col in range(8):
                x1 = col * self.SQUARE_SIZE
                y1 = row * self.SQUARE_SIZE
                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE
                
                # Calculate the chess square
                square = chess.square(col, 7 - row)
                
                # Calculate control of this square
                white_control, black_control = self.calculate_square_control(square)
                
                # Get square color based on control
                color = self.get_square_color(row, col, white_control, black_control)
                
                # Draw the square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
    
    def draw_pieces(self):
        """Draw the chess pieces on the board"""
        self.canvas.delete("piece")  # Remove all pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x = (chess.square_file(square) * self.SQUARE_SIZE) + self.SQUARE_SIZE // 2
                y = ((7 - chess.square_rank(square)) * self.SQUARE_SIZE) + self.SQUARE_SIZE // 2
                color = "white" if piece.color else "black"
                self.canvas.create_text(x, y, text=self.pieces[piece.symbol()],
                                     font=('Arial', 36), fill=color, tags="piece")
    
    def get_square_from_coords(self, x, y):
        """Convert canvas coordinates to chess square"""
        file = x // self.SQUARE_SIZE
        rank = 7 - (y // self.SQUARE_SIZE)
        return chess.square(file, rank)

    def on_square_clicked(self, event):
        """Handle mouse clicks on the board"""
        print(f"Click detected at ({event.x}, {event.y})")  # Debug print
        square = self.get_square_from_coords(event.x, event.y)
        
        if self.selected_square is None:
            # First click - select piece
            piece = self.board.piece_at(square)
            if piece:
                print(f"Selected piece: {piece.symbol()} at {chess.square_name(square)}")  # Debug print
                self.selected_square = square
                # Clear any existing highlights
                self.canvas.delete("highlight")
                # Highlight selected square
                x = (chess.square_file(square) * self.SQUARE_SIZE)
                y = ((7 - chess.square_rank(square)) * self.SQUARE_SIZE)
                self.canvas.create_rectangle(x, y, x + self.SQUARE_SIZE, y + self.SQUARE_SIZE,
                                          outline="yellow", width=2, tags="highlight")
        else:
            # Second click - try to make move
            move = chess.Move(self.selected_square, square)
            print(f"Attempting move from {chess.square_name(self.selected_square)} to {chess.square_name(square)}")  # Debug print
            if move in self.board.legal_moves:
                self.board.push(move)
                print(f"Move successful: {move}")  # Debug print
                self.save_position()
                self.canvas.delete("highlight")
                
                # Notify callback if set
                if self.move_callback:
                    self.move_callback(move)
            else:
                print(f"Illegal move attempted: {move}")  # Debug print
            
            self.selected_square = None
            self.canvas.delete("highlight")

    
    def get_board_state(self):
        """Return the current state of the board"""
        return self.board
    
    def get_move_number(self):
        """Return the current move number"""
        return self.board.fullmove_number
    
    def save_position(self):
        """Save the current board position and FEN string to files"""
        # Get current move number and position details
        move_num = self.get_move_number() if self.board.turn == chess.BLACK else self.get_move_number() - 1
        position = str(self.board)
        fen = self.board.fen()
        
        # Determine if it's White's or Black's move that just happened
        # Note: board.turn gives the color to move next, so we need the opposite
        color = 'w' if self.board.turn == chess.BLACK else 'b'
        
        
        # Create filenames with color indicator
        position_filename = os.path.join('rev_1/position_history', f'board_position_move_{move_num}_{color}')
        fen_filename = os.path.join('rev_1/position_history', f'board_fen_move_{move_num}_{color}')
        
        # Save position to file
        with open(position_filename, 'w') as f:
            f.write(fen)
            f.write('\n')
            f.write(position)
            
    
    def make_move(self, move):
        """Make a move programmatically"""
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.draw_pieces()
            
            # Notify callback if set
            if self.move_callback:
                self.move_callback(move)
            return True
        return False

    def calculate_square_control(self, square):
        try:
            white_control = len(list(self.board.attackers(chess.WHITE, square)))
            black_control = len(list(self.board.attackers(chess.BLACK, square)))
            return white_control, black_control
        except Exception as e:
            print(f"Error calculating control: {e}")
            return 0, 0
        
    def get_square_color(self, row, col, white_control, black_control):
        """Get the color for a square based on control and base color"""
        # Base color
        base_color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
        
        # Convert hex to RGB
        base_r = int(base_color[1:3], 16)
        base_g = int(base_color[3:5], 16)
        base_b = int(base_color[5:7], 16)
        
        # Calculate control intensity (adjust these factors to change color intensity)
        white_intensity = min(white_control * 0.2, 1.0)  # 20% more green per controlling piece
        black_intensity = min(black_control * 0.2, 1.0)  # 20% more red per controlling piece
        
        if white_control > 0 and black_control > 0:
            # Purple tint for contested squares
            r = min(base_r + int(128 * max(white_intensity, black_intensity)), 255)
            g = base_g
            b = min(base_b + int(128 * max(white_intensity, black_intensity)), 255)
        else:
            # Green tint for white control
            r = base_r
            g = min(base_g + int(128 * white_intensity), 255)
            b = base_b
            
            # Red tint for black control
            r = min(r + int(128 * black_intensity), 255)
            g = g
            b = b
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"