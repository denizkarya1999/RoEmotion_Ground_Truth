"""Small timestamped text recorder used by the live inference GUI."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TextIO


class TxtRateRecorder:
    """Write one tab-separated text file for each inference session."""

    def __init__(
        self,
        output_directory: Path,
        filename_prefix: str,
        fieldnames: tuple[str, ...],
    ) -> None:
        self.output_directory = output_directory
        self.filename_prefix = filename_prefix
        self.fieldnames = fieldnames
        self.path: Path | None = None
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None

    @property
    def is_recording(self) -> bool:
        return self._file is not None

    def start(self) -> Path:
        """Open a new session file and write its header."""
        self.stop()
        self.path = None
        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_directory / f"{self.filename_prefix}_{timestamp}.txt"
        try:
            self._file = path.open("x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            self._writer.writeheader()
            self._file.flush()
        except OSError:
            self.stop()
            raise
        self.path = path
        return self.path

    def record(self, **values: object) -> None:
        """Append and immediately flush a measurement row."""
        if self._writer is None or self._file is None:
            return
        self._writer.writerow({name: values.get(name, "") for name in self.fieldnames})
        self._file.flush()

    def stop(self) -> None:
        """Close the active session file, if any."""
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
