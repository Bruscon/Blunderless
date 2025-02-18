# threaded_capture.py
import tkinter as tk
from threading import Thread, Event, Lock
import queue
import time
import logging
from dataclasses import dataclass
from typing import Callable, Optional
from chess_gui import ChessGUI  # Add this import

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Container for processing results to pass between threads"""
    success: bool
    fen: Optional[str] = None
    error: Optional[str] = None

class ThreadedCapture:
    """
    Handles chess board capture and processing using separate threads to maintain
    GUI responsiveness.
    """
    def __init__(self, capture_func: Callable, process_func: Callable, chess_gui: ChessGUI):
        self.capture_func = capture_func
        self.process_func = process_func
        self.chess_gui = chess_gui  # Store ChessGUI instance instead of root
        
        # Thread control
        self.running = Event()
        self.capture_lock = Lock()
        self.processing_lock = Lock()
        
        # Communication queues
        self.capture_queue = queue.Queue(maxsize=1)  # Only need latest capture
        self.result_queue = queue.Queue(maxsize=2)   # Allow some buffering of results
        
        # Thread references
        self.capture_thread = None
        self.process_thread = None
        
        # Performance tracking
        self.last_capture_time = 0
        self.last_process_time = 0
        self.error_count = 0
        
        # Configuration
        self.min_capture_interval = 0.1
        self.max_capture_interval = 2.0
        self.error_backoff = 1.5
        self.max_errors = 5

    def start(self):
        """Start capture and processing threads"""
        if not self.running.is_set():
            logger.info("Starting capture and processing threads")
            self.running.set()
            self.error_count = 0
            
            # Clear queues
            self._clear_queue(self.capture_queue)
            self._clear_queue(self.result_queue)
            
            # Start threads
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.process_thread = Thread(target=self._process_loop, daemon=True)
            
            self.capture_thread.start()
            self.process_thread.start()
            
            # Start GUI update loop - use root window for after()
            self.chess_gui.root.after(100, self._update_gui)

    def stop(self):
        """Stop all threads and cleanup"""
        logger.info("Stopping capture and processing")
        self.running.clear()
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            
        self.capture_thread = None
        self.process_thread = None

    def _clear_queue(self, q: queue.Queue):
        """Safely clear a queue"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _capture_loop(self):
        """Dedicated thread for capturing screenshots"""
        while self.running.is_set():
            try:
                with self.capture_lock:
                    # Calculate appropriate delay
                    capture_interval = min(
                        max(self.last_process_time * 0.8, self.min_capture_interval),
                        self.max_capture_interval
                    )
                    
                    # Apply error backoff if needed
                    if self.error_count > 0:
                        capture_interval *= (self.error_backoff ** self.error_count)
                    
                    # Ensure minimum time between captures
                    time_since_last = time.time() - self.last_capture_time
                    if time_since_last < capture_interval:
                        time.sleep(capture_interval - time_since_last)
                    
                    # Perform capture
                    capture_start = time.time()
                    screenshot_path = self.capture_func()
                    self.last_capture_time = time.time()
                    
                    # Put in queue, dropping old if necessary
                    try:
                        self.capture_queue.put_nowait(screenshot_path)
                    except queue.Full:
                        try:
                            self.capture_queue.get_nowait()  # Remove old
                            self.capture_queue.put_nowait(screenshot_path)
                        except queue.Empty:
                            pass  # Queue was already cleared
                    
                    logger.debug(f"Capture completed in {time.time() - capture_start:.3f}s")
                    
            except Exception as e:
                self.error_count = min(self.error_count + 1, self.max_errors)
                logger.error(f"Capture error: {e}")
                
                if self.error_count >= self.max_errors:
                    logger.error("Maximum capture errors reached")
                    self.stop()
                    break
                
                time.sleep(0.5 * self.error_count)  # Back off on errors

    def _process_loop(self):
        """Dedicated thread for processing captured images"""
        while self.running.is_set():
            try:
                # Get latest capture
                screenshot_path = self.capture_queue.get(timeout=1.0)
                
                with self.processing_lock:
                    process_start = time.time()
                    
                    try:
                        # Process the image
                        self.process_func()
                        self.last_process_time = time.time() - process_start
                        
                        # Get the FEN from the GUI's board state
                        fen = self.chess_gui.board_state.get_fen()
                        
                        # Put successful result in queue
                        result = ProcessingResult(success=True, fen=fen)
                        self.error_count = max(0, self.error_count - 1)
                        
                    except Exception as e:
                        self.error_count = min(self.error_count + 1, self.max_errors)
                        result = ProcessingResult(success=False, error=str(e))
                        logger.error(f"Processing error: {e}")
                    
                    # Update result queue
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        try:
                            self.result_queue.get_nowait()  # Remove old
                            self.result_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                    
            except queue.Empty:
                continue  # No new captures to process
                
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                time.sleep(0.5)  # Prevent tight error loop

    def _update_gui(self):
        """Update GUI with latest processing results"""
        if not self.running.is_set():
            return
            
        try:
            # Check for new results
            while True:  # Process all available results
                try:
                    result = self.result_queue.get_nowait()
                    
                    if result.success:
                        # Update status with successful detection
                        self.chess_gui.set_status("Board position updated")
                    else:
                        # Update status with error
                        self.chess_gui.set_status(f"Error: {result.error}")
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"GUI update error: {e}")
            
        finally:
            # Schedule next update if still running
            if self.running.is_set():
                self.chess_gui.root.after(100, self._update_gui)