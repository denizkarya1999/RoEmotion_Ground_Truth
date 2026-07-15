#!/usr/bin/env python3
"""Launch the Arduino heart-rate monitor GUI."""

import os
import sys
from pathlib import Path


def use_project_environment() -> None:
    """Restart with the project venv and make local native libraries visible."""
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
    use_project_environment()
    from bpm_gui import run

    run()


if __name__ == "__main__":
    main()
