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
        self.queue = queue.Queue(maxsize=2)  # Only keep latest 2 frames
        self.worker_thread = None
        self.processing = False

    def start(self):
        """Start the auto-capture thread"""
        if not self.running.is_set():
            self.running.set()
            self.worker_thread = Thread(target=self._capture_loop, daemon=True)
            self.worker_thread.start()
            self.gui.after(250, self._process_queue)

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
                self.capture_func()
                try:
                    self.queue.put_nowait("process")  # Non-blocking put
                except queue.Full:
                    # Queue is full, clear it and put new frame
                    try:
                        while True:
                            self.queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.queue.put("process")
                time.sleep(0.1)
            except Exception as e:
                self.queue.put(("error", str(e)))

    def _process_queue(self):
        """Process the queue on the main thread"""
        if self.processing:
            # Skip this update if still processing previous frame
            if self.running.is_set():
                self.gui.after(250, self._process_queue)
            return

        try:
            self.processing = True
            item = self.queue.get_nowait()
            if isinstance(item, tuple) and item[0] == "error":
                self.gui.set_status(f"Error: {item[1]}")
            else:
                self.process_func()
        except queue.Empty:
            pass
        finally:
            self.processing = False

        if self.running.is_set():
            self.gui.after(250, self._process_queue)