import subprocess
import threading
import tkinter as tk
import queue
import time
from manage_rev_1 import manage_folders
from chess_gui import ChessGUI
import chess

def read_from_file(file_path: str) -> str:
    """Reads a prompt from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        return f"Error: The file at '{file_path}' was not found."
    except Exception as e:
        return f"Error reading file: {e}"

def write_to_file(response: str, file_path: str) -> None:
    """Saves the model's response to a text file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response)
        print(f"file saved to {file_path}")
    except Exception as e:
        print(f"Error saving to file: {e}")


def ask_LLM(question):
    model_name = "deepseek-r1:32b"
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=question,
            text=True,
            capture_output=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            return f"Error running Ollama: {result.stderr.strip()}"
        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: file not found"
    except UnicodeDecodeError:
        return "Error decoding output. Ensure the model's response is UTF-8 compatible."

def process_moves_thread(gui, move_queue):
    """Process moves from the queue"""
    try:
        # # Check for moves without blocking
        # move = move_queue.get_nowait()

        # Block until a move is available
        move = move_queue.get()
        
        # Print move and new position
        print(f"\nMove made: {move}")
        print("New position:")
        print(gui.get_board_state())
        
        # Your move processing code here
        # For example:
        # analyze_position()
        # make_computer_move()
        # etc.
        
    except queue.Empty:
        pass


    prompt =    "FEN chess position: " + gui.board.fen() + \
                "\nASCII chess position: " + str(gui.get_board_state()) + "\n" +\
                read_from_file("prompts/questions.txt")

    # Determine if it's White's or Black's move that just happened
    # Note: board.turn gives the color to move next, so we need the opposite
    color = 'w' if gui.board.turn == chess.BLACK else 'b'
    write_to_file(prompt, f"rev_1/prompts/move_{gui.get_move_number()}_{color}_prompt.txt")

    answer = ask_LLM(prompt)
    print(answer)
    write_to_file(answer, f"rev_1/responses/move_{gui.get_move_number()}_{color}_analysis.txt")
    
    # Schedule the next check
    gui.root.after(100, process_moves_thread, gui, move_queue)

if __name__ == "__main__":
    # Archive the old rev_1 and make a new one
    manage_folders()
    
    # Create move queue
    move_queue = queue.Queue()
    
    # Initialize tkinter and chess GUI
    root = tk.Tk()
    gui = ChessGUI(root)
    
    # Set up move callback
    def on_move(move):
        move_queue.put(move)
    gui.set_move_callback(on_move)
    
    # Start move processing in a separate thread
    move_thread = threading.Thread(target=process_moves_thread, 
                                 args=(gui, move_queue),
                                 daemon=True)
    move_thread.start()
    
    # Start the GUI main loop
    root.mainloop()