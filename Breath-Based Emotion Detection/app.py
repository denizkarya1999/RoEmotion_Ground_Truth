"""Application launcher for the breathing emotion detector.

Run this file to start the desktop app. The actual interface is implemented in
``gui.py``; this module only creates the Tkinter window and starts the event loop.
"""

from __future__ import annotations

import tkinter as tk

from gui import BreathingEmotionGUI


def main() -> None:
    """Create the main window and hand control to Tkinter."""
    root = tk.Tk()

    # Basic window settings keep the app usable on both small and large screens.
    root.title("Breath-based Emotion Detection")
    root.geometry("860x620")
    root.minsize(720, 540)

    gui = BreathingEmotionGUI(root)

    # Use the GUI's close method so the microphone stream is stopped before the
    # window is destroyed.
    root.protocol("WM_DELETE_WINDOW", gui.close)
    root.mainloop()


if __name__ == "__main__":
    # This guard lets other files import main() without immediately opening a GUI.
    main()
