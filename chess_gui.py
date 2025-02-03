import tkinter as tk
import chess
import os

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
        # Add any periodic updates here
        
        # Schedule next update
        self.root.after(100, self.update)
    
    def set_move_callback(self, callback):
        """Set a function to be called when a move is made"""
        self.move_callback = callback
    
    def draw_board(self):
        """Draw the chess board squares"""
        for row in range(8):
            for col in range(8):
                x1 = col * self.SQUARE_SIZE
                y1 = row * self.SQUARE_SIZE
                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE
                
                # Alternate square colors
                color = "#DDB88C" if (row + col) % 2 == 0 else "#A66D4F"
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
        square = self.get_square_from_coords(event.x, event.y)
        
        if self.selected_square is None:
            # First click - select piece
            piece = self.board.piece_at(square)
            if piece:
                self.selected_square = square
                # Highlight selected square
                x = (chess.square_file(square) * self.SQUARE_SIZE)
                y = ((7 - chess.square_rank(square)) * self.SQUARE_SIZE)
                self.canvas.create_rectangle(x, y, x + self.SQUARE_SIZE, y + self.SQUARE_SIZE,
                                          outline="yellow", width=2, tags="highlight")
        else:
            # Second click - try to make move
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                print(f"Move number: {self.get_move_number()}")
                self.board.push(move)
                self.save_position()
                self.canvas.delete("highlight")
                self.draw_pieces()
                
                # Notify callback if set
                if self.move_callback:
                    self.move_callback(move)
                
                # Check for game end conditions
                if self.board.is_game_over():
                    result = "1-0" if self.board.is_checkmate() and self.board.turn == chess.BLACK else "0-1"
                    print(f"Game Over! Result: {result}")
            
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