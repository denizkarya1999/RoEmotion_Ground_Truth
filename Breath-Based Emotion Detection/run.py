"""Application launcher for the breathing emotion detector.

Run this file to start the desktop app. The actual interface is implemented in
``gui.py``; this module only creates the Tkinter window and starts the event loop.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path


def use_project_environment() -> None:
    """Restart with the project venv and make its local PortAudio visible."""
    project_root = Path(__file__).resolve().parent.parent
    virtual_environment = project_root / ".venv"
    interpreter = virtual_environment / "bin" / "python"
    library_directory = virtual_environment / "lib"

    current_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_paths = current_library_path.split(":") if current_library_path else []
    if str(library_directory) not in library_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [str(library_directory), *library_paths]
        )

    if interpreter.exists() and Path(sys.prefix).resolve() != virtual_environment:
        os.execve(
            str(interpreter),
            [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ,
        )


def main() -> None:
    """Create the main window and hand control to Tkinter."""
    use_project_environment()
    from gui import BreathingEmotionGUI

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
