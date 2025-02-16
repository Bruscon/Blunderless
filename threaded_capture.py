# threaded_capture.py
import tkinter as tk
from threading import Thread, Event
import queue
import time

class ThreadedCapture:
    """
    Handles auto-capture functionality in a separate thread to prevent GUI blocking
    """
    def __init__(self, capture_func, process_func, gui):
        self.capture_func = capture_func
        self.process_func = process_func
        self.gui = gui
        self.running = Event()
        self.queue = queue.Queue()
        self.worker_thread = None

    def start(self):
        """Start the auto-capture thread"""
        if not self.running.is_set():
            self.running.set()
            self.worker_thread = Thread(target=self._capture_loop, daemon=True)
            self.worker_thread.start()
            self.gui.after(100, self._process_queue)

    def stop(self):
        """Stop the auto-capture thread"""
        self.running.clear()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.running.is_set():
            try:
                # Capture screenshot
                self.capture_func()
                # Add processing task to queue
                self.queue.put("process")
                # Wait before next capture
                time.sleep(0.5)  # 500ms delay
            except Exception as e:
                self.queue.put(("error", str(e)))

    def _process_queue(self):
        """Process the queue on the main thread"""
        try:
            while True:
                item = self.queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "error":
                    self.gui.set_status(f"Error: {item[1]}")
                else:
                    self.process_func()
        except queue.Empty:
            pass
        
        if self.running.is_set():
            # Schedule next queue check
            self.gui.after(100, self._process_queue)