"""Tkinter GUI for sound-only breathing emotion prediction.

This file owns the user interface: microphone selection, start/stop controls,
prediction labels, and the live decibel graph. The signal-processing logic lives
in ``breathe_analyzer.py`` so the GUI code can focus on display and interaction.
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from breathe_analyzer import (
    DECIBEL_BANDS,
    DECIBEL_MAX,
    DECIBEL_MIN,
    DECIBEL_THRESHOLDS,
    HISTORY_SECONDS,
    PREDICTION_WINDOW_SECONDS,
    SENSITIVITY_BOOST_MAX_DB,
    SENSITIVITY_BOOST_MIN_DB,
    BreathingAnalyzer,
    EmotionEstimate,
)


# Fallback audio settings used when the selected device does not provide better
# information.
DEFAULT_SAMPLE_RATE = 16_000
BLOCK_SIZE = 1024

# Canvas layout constants for the decibel graph.
RIGHT_AXIS_WIDTH = 18
TEXT_SAFE_MARGIN = 10
MIN_READABLE_WIDTH = 460

# Colors used for the background bands behind the live graph.
THRESHOLD_COLORS = {
    "unknown": "#334155",
    "calm": "#113f67",
    "focused": "#5a3e09",
    "stressed": "#5a1726",
}

# Compact labels fit better inside the colored graph bands.
SHORT_BAND_LABELS = {
    "very quiet / unclear": "UNCLEAR",
    "quiet / calm": "CALM",
    "moderate / active": "FOCUS",
    "loud / stressed": "STRESS",
}

# The threshold lines and labels make the decibel categories visible.
THRESHOLD_LINE_COLORS = {
    -55: "#38bdf8",
    -42: "#fbbf24",
    -30: "#fb7185",
}
THRESHOLD_EMOTION_LABELS = {
    -55: "-55 calm",
    -42: "-42 active",
    -30: "-30 stress",
}


class BreathingEmotionGUI(ttk.Frame):
    """Microphone controls, prediction display, and live sound trace."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=18)
        # The analyzer handles all audio math. The GUI asks it for display-ready
        # estimates whenever new microphone data arrives.
        self.analyzer = BreathingAnalyzer()

        # The sounddevice callback runs outside Tkinter's main thread. It pushes
        # audio blocks into this queue, and poll_audio reads them safely later.
        self.audio_queue: queue.Queue[tuple[float | None, np.ndarray]] = queue.Queue()
        self.microphones: list[tuple[int, str, float]] = []
        self.audio_stream = None
        self.stream_warning = ""
        self.current_sample_rate = DEFAULT_SAMPLE_RATE

        self._create_variables()
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self._build_ui()
        self.refresh_microphones()
        self.after(150, self.poll_audio)
        self.show_estimate(self.analyzer.analyze())
        self.draw_sound_trace()

    def _create_variables(self) -> None:
        """Create Tkinter variables that keep widgets and code in sync."""
        self.mic_var = tk.StringVar()
        self.emotion_var = tk.StringVar()
        self.message_var = tk.StringVar()
        self.live_db_var = tk.StringVar(value="-- dBFS")
        self.average_db_var = tk.StringVar(value="-- dBFS")
        self.band_var = tk.StringVar(value="--")
        self.window_var = tk.StringVar(value=f"0/{PREDICTION_WINDOW_SECONDS}s")
        self.confidence_var = tk.StringVar(value="0%")
        self.sensitivity_value_var = tk.DoubleVar(value=SENSITIVITY_BOOST_MAX_DB)
        self.sensitivity_boost_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Microphone stopped")
        self._update_sensitivity_label(SENSITIVITY_BOOST_MAX_DB)

    def _build_ui(self) -> None:
        """Create and arrange every visible widget in the main window."""
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 22, "bold"))
        style.configure("Prediction.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Metric.TLabel", font=("TkDefaultFont", 13))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        # Header text gives the app name and reminds users that this is only a
        # rough rule-based signal, not a medical or psychological diagnosis.
        ttk.Label(self, text="Breath-based Emotion Detection", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self,
            text="32-second voice-band decibel rules from microphone audio. Not a diagnosis.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        # Audio controls: choose a microphone, start/stop recording, clear data,
        # and adjust the extra sensitivity boost used by the analyzer.
        controls = ttk.LabelFrame(self, text="Audio input", padding=12)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Microphone").grid(row=0, column=0, sticky="w")
        self.mic_box = ttk.Combobox(controls, textvariable=self.mic_var, state="readonly")
        self.mic_box.grid(row=0, column=1, columnspan=2, sticky="ew", padx=8)
        ttk.Button(controls, text="Refresh", command=self.refresh_microphones).grid(
            row=0, column=3, sticky="ew"
        )

        self.start_button = ttk.Button(
            controls, text="Start Recording", command=self.start_audio
        )
        self.start_button.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self.stop_audio, state="disabled"
        )
        self.stop_button.grid(row=1, column=1, pady=(10, 0), padx=8, sticky="ew")
        ttk.Button(controls, text="Clear", command=self.clear).grid(
            row=1, column=2, pady=(10, 0), sticky="ew"
        )
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        ttk.Label(controls, text="Microphone sensitivity").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Scale(
            controls,
            from_=SENSITIVITY_BOOST_MIN_DB,
            to=SENSITIVITY_BOOST_MAX_DB,
            variable=self.sensitivity_value_var,
            command=self.set_sensitivity_from_slider,
        ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=8, pady=(10, 0))
        ttk.Label(controls, textvariable=self.sensitivity_boost_var).grid(
            row=2, column=3, sticky="w", pady=(10, 0)
        )

        # Prediction panel: all values here come from one EmotionEstimate object.
        result = ttk.LabelFrame(self, text="Prediction", padding=14)
        result.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        result.columnconfigure(0, weight=1)
        ttk.Label(result, textvariable=self.emotion_var, style="Prediction.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(result, textvariable=self.message_var, wraplength=760).grid(
            row=1, column=0, sticky="w", pady=(6, 12)
        )

        metrics = ttk.Frame(result)
        metrics.grid(row=2, column=0, sticky="ew")
        for column in range(5):
            metrics.columnconfigure(column, weight=1)
        self._metric(metrics, "Live voice dBFS", self.live_db_var, 0)
        self._metric(metrics, "Average dB Per Emotion", self.average_db_var, 1)
        self._metric(metrics, "Decibel band", self.band_var, 2)
        self._metric(metrics, "Next window", self.window_var, 3)
        self._metric(metrics, "Confidence", self.confidence_var, 4)

        # Canvas graph: drawn manually so threshold bands, live points, and
        # average points can share the same decibel scale.
        trace_panel = ttk.LabelFrame(self, text="Decibel graph", padding=12)
        trace_panel.grid(row=4, column=0, sticky="nsew", pady=(16, 0))
        trace_panel.columnconfigure(0, weight=1)
        trace_panel.rowconfigure(0, weight=1)
        self.trace_canvas = tk.Canvas(
            trace_panel,
            background="#101828",
            height=260,
            highlightthickness=0,
        )
        self.trace_canvas.grid(row=0, column=0, sticky="nsew")
        self.trace_canvas.bind("<Configure>", lambda event: self.draw_sound_trace())

    @staticmethod
    def _metric(parent: ttk.Frame, label: str, value: tk.StringVar, column: int) -> None:
        """Add one labeled number to the prediction metrics row."""
        block = ttk.Frame(parent)
        block.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 10, 0))
        ttk.Label(block, text=label).grid(row=0, column=0, sticky="w")
        ttk.Label(block, textvariable=value, style="Metric.TLabel").grid(
            row=1, column=0, sticky="w", pady=(3, 0)
        )

    def refresh_microphones(self) -> None:
        """Reload available input devices and select the system default."""
        if self.audio_stream is not None:
            self.status_var.set("Stop recording before refreshing microphones")
            return

        # sounddevice is optional at import time, so load it only when audio is
        # actually needed and show a helpful error if it is unavailable.
        sd, error = load_sounddevice()
        if sd is None:
            self._set_audio_device_error(error)
            return

        try:
            devices = sd.query_devices()
            default_input = default_input_device(sd)
        except Exception as exc:
            self._set_audio_device_error(f"No microphones found: {exc}")
            return

        # Store only devices that can record audio. Each entry keeps the
        # sounddevice id, display name, and preferred sample rate.
        self.microphones = [
            (index, device["name"], float(device["default_samplerate"] or 0))
            for index, device in enumerate(devices)
            if device["max_input_channels"] > 0
        ]

        if not self.microphones:
            self._set_audio_device_error("No input microphones found")
            return

        self.mic_box["values"] = [
            f"{index}: {name}" for index, name, _ in self.microphones
        ]
        # Prefer the operating system's default input device when it is present.
        selected = next(
            (
                offset
                for offset, (index, _, _) in enumerate(self.microphones)
                if index == default_input
            ),
            0,
        )
        self.mic_box.current(selected)

        self.start_button.configure(state="normal")
        self.status_var.set(f"{len(self.microphones)} microphone(s) available")

    def _set_audio_device_error(self, message: str | None) -> None:
        """Show an audio setup problem and disable recording controls."""
        self.microphones = []
        self.mic_box["values"] = [message or "Audio unavailable"]
        self.mic_box.current(0)
        self.start_button.configure(state="disabled")
        self.status_var.set("Audio unavailable")

    def start_audio(self) -> None:
        """Open the selected microphone and begin feeding audio to the analyzer."""
        sd, error = load_sounddevice()
        if sd is None:
            messagebox.showerror("Audio unavailable", error)
            return

        microphone = self.selected_microphone()
        if microphone is None:
            messagebox.showerror("No microphone", "Select a microphone first.")
            return

        device_id, device_name, input_sample_rate = microphone

        def queue_input(indata, callback_time, status) -> None:
            # This runs in an audio thread, so only queue data here.
            if status:
                self.stream_warning = str(status)
            timestamp = getattr(callback_time, "inputBufferAdcTime", None)
            if timestamp is None:
                timestamp = getattr(callback_time, "currentTime", None)
            # Copy the buffer because sounddevice may reuse ``indata`` after the
            # callback returns.
            self.audio_queue.put(
                (float(timestamp) if timestamp is not None else None, indata.copy())
            )

        def input_callback(indata, frames, callback_time, status) -> None:
            # Keep the callback tiny. Tkinter updates happen in poll_audio.
            queue_input(indata, callback_time, status)

        self.audio_stream, self.current_sample_rate, input_error = (
            open_started_input_stream(
                sd,
                device_id,
                input_sample_rate,
                input_callback,
            )
        )
        if self.audio_stream is None:
            messagebox.showerror("Microphone error", str(input_error))
            return

        self.analyzer.clear()
        # Lock the microphone selector while recording so the stream and selected
        # device cannot get out of sync.
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.mic_box.configure(state="disabled")
        self.status_var.set(
            f"Recording: {device_name} at {self.current_sample_rate} Hz"
        )
        self.show_estimate(self.analyzer.analyze())
        self.draw_sound_trace()

    def stop_audio(self) -> None:
        """Stop recording and re-enable microphone selection."""
        close_stream(self.audio_stream)
        self.audio_stream = None

        self.start_button.configure(state="normal" if self.microphones else "disabled")
        self.stop_button.configure(state="disabled")
        self.mic_box.configure(state="readonly")
        self.status_var.set("Microphone stopped")

    def clear(self) -> None:
        """Clear collected samples and redraw the empty prediction state."""
        self.analyzer.clear()
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        self.show_estimate(self.analyzer.analyze())
        self.draw_sound_trace()

    def set_sensitivity_from_slider(self, value: str | float) -> None:
        """Handle slider movement and restart the current prediction window."""
        current_boost = int(round(float(value)))
        self.sensitivity_value_var.set(current_boost)
        self._update_sensitivity_label(current_boost)

        previous_boost = self.analyzer.sensitivity_boost_db
        current_boost = self.analyzer.set_sensitivity_boost(current_boost)
        self._update_sensitivity_label(current_boost)
        # Empty queued audio that was captured under the previous sensitivity.
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()

        if current_boost == previous_boost:
            return
        else:
            percent = sensitivity_boost_percent(current_boost)
            self.status_var.set(
                f"Sensitivity boost set to {percent}% (+{current_boost} dB); "
                "window restarted"
            )

        self.show_estimate(self.analyzer.analyze())
        self.draw_sound_trace()

    def _update_sensitivity_label(self, boost_db: int) -> None:
        """Show the slider value as both percent and dB boost."""
        percent = sensitivity_boost_percent(boost_db)
        self.sensitivity_boost_var.set(f"{percent}% (+{boost_db} dB)")

    def selected_microphone(self) -> tuple[int, str, float] | None:
        """Return the selected microphone tuple, or ``None`` if selection is invalid."""
        selection = self.mic_box.current()
        if selection < 0 or selection >= len(self.microphones):
            return None
        return self.microphones[selection]

    def poll_audio(self) -> None:
        """Move queued microphone blocks into the analyzer on Tkinter's thread."""
        updated = False
        while not self.audio_queue.empty():
            timestamp, audio = self.audio_queue.get_nowait()
            self.analyzer.add_audio_block(
                audio,
                timestamp=timestamp,
                sample_rate=self.current_sample_rate,
            )
            updated = True

        if updated:
            self.show_estimate(self.analyzer.analyze())
            self.draw_sound_trace()

        if self.stream_warning:
            self.status_var.set(self.stream_warning)
            self.stream_warning = ""

        # Schedule the next poll. This keeps the UI responsive without using a
        # second Tkinter thread.
        self.after(150, self.poll_audio)

    def show_estimate(self, estimate: EmotionEstimate) -> None:
        """Copy an analyzer estimate into the visible labels."""
        self.emotion_var.set(estimate.label)
        self.message_var.set(estimate.message)
        self.live_db_var.set(format_value(estimate.volume_db, "-- dBFS", " dBFS", 1))
        self.average_db_var.set(
            format_value(estimate.average_db, "-- dBFS", " dBFS", 1)
        )
        self.band_var.set(format_category(estimate.category))
        self.window_var.set(
            f"{min(round(estimate.window_seconds), PREDICTION_WINDOW_SECONDS)}/"
            f"{PREDICTION_WINDOW_SECONDS}s"
        )
        self.confidence_var.set(f"{round(estimate.confidence * 100)}%")

    def draw_sound_trace(self) -> None:
        """Redraw the decibel graph from the analyzer's current history."""
        self.trace_canvas.delete("all")
        width = max(self.trace_canvas.winfo_width(), 2)
        height = max(self.trace_canvas.winfo_height(), 2)
        padding = 16
        # Leave a small strip on the right for the vertical axis and threshold
        # labels, but keep a minimum plot width on small windows.
        plot_right = max(width - RIGHT_AXIS_WIDTH, padding + 80)
        plot_bottom = height - padding
        plot_top = padding

        estimate = self.analyzer.analyze()
        self._draw_threshold_bands(width, height, padding, plot_right)
        self._draw_trace_grid(plot_right, plot_bottom, padding)
        live_times, live_values = self.analyzer.db_curve()
        average_times, average_values = self.analyzer.average_db_curve()
        if len(live_values) == 0 and len(average_values) == 0:
            # An empty graph should still explain what will appear once audio
            # starts.
            self.trace_canvas.create_text(
                (padding + plot_right) / 2,
                clamp(plot_bottom / 2, plot_top + 36, plot_bottom - 36),
                text="Start recording to see live voice dBFS",
                fill="#e5e7eb",
                font=("TkDefaultFont", 14, "bold"),
                width=max(plot_right - padding - 180, 220),
            )
            self._draw_threshold_overlay(width, height, padding, plot_right, estimate)
            return

        # Draw measured data first, then overlay thresholds and labels so the
        # category markers stay readable.
        self._draw_db_series(
            live_times,
            live_values,
            average_times,
            average_values,
            padding,
            plot_right,
            plot_top,
            plot_bottom,
        )
        self._boxed_text(
            padding,
            padding,
            "live voice dBFS",
            fill="#cdd5df",
            background="#020617",
            font=("TkDefaultFont", 9, "bold"),
            bounds=(padding, plot_top, plot_right, plot_bottom),
            anchor="nw",
            pad_x=4,
            pad_y=2,
        )
        self._draw_threshold_overlay(width, height, padding, plot_right, estimate)

    def _draw_db_series(
        self,
        live_times: np.ndarray,
        live_values: np.ndarray,
        average_times: np.ndarray,
        average_values: np.ndarray,
        padding: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
    ) -> None:
        """Draw live dBFS values and completed-window average points."""
        time_sets = [
            times for times in (live_times, average_times) if len(times) > 0
        ]
        latest_time = max(float(times[-1]) for times in time_sets)
        start_time = latest_time - HISTORY_SECONDS

        def x_for(time_value: float) -> float:
            # The x-axis is a rolling HISTORY_SECONDS-wide time window.
            fraction = (float(time_value) - start_time) / HISTORY_SECONDS
            return padding + clamp(fraction, 0, 1) * (plot_right - padding)

        live_points: list[float] = []
        for time_value, decibel in zip(live_times, live_values):
            live_points.extend(
                (
                    float(x_for(time_value)),
                    float(db_to_y(decibel, plot_top, plot_bottom)),
                )
            )

        if len(live_points) >= 4:
            # Two or more points can be connected into a live trace.
            self.trace_canvas.create_line(
                *live_points,
                fill="#2dd4bf",
                width=3,
                smooth=True,
            )
        elif len(live_points) == 2:
            # With only one point, draw a dot instead of a zero-length line.
            x, y = live_points
            self.trace_canvas.create_oval(
                x - 4,
                y - 4,
                x + 4,
                y + 4,
                fill="#2dd4bf",
                outline="#020617",
                width=2,
            )

        average_points: list[float] = []
        for time_value, decibel in zip(average_times, average_values):
            average_points.extend(
                (
                    float(x_for(time_value)),
                    float(db_to_y(decibel, plot_top, plot_bottom)),
                )
            )

        if len(average_points) >= 4:
            # Completed-window averages update less often, so a dashed line helps
            # separate them from the live trace.
            self.trace_canvas.create_line(
                *average_points,
                fill="#fbbf24",
                width=3,
                dash=(6, 4),
            )

        # Always draw average points as circles so each completed 32-second
        # prediction is easy to spot.
        for index in range(0, len(average_points), 2):
            x = average_points[index]
            y = average_points[index + 1]
            self.trace_canvas.create_oval(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                fill="#fbbf24",
                outline="#020617",
                width=2,
            )

    def _draw_trace_grid(self, plot_right: int, plot_bottom: int, padding: int) -> None:
        """Draw subtle horizontal guide lines behind the graph data."""
        for fraction in (0.25, 0.5, 0.75):
            y = padding + fraction * (plot_bottom - 2 * padding)
            self.trace_canvas.create_line(
                padding,
                y,
                plot_right,
                y,
                fill="#243042",
            )

    def _draw_threshold_bands(
        self,
        width: int,
        height: int,
        padding: int,
        plot_right: int,
    ) -> None:
        """Draw colored background regions for each decibel category."""
        plot_top = padding
        plot_bottom = height - padding

        if width >= MIN_READABLE_WIDTH:
            self._boxed_text(
                padding + 10,
                plot_top + 14,
                "DECIBEL THRESHOLDS",
                fill="#020617",
                background="#e5e7eb",
                font=("TkDefaultFont", 10, "bold"),
                bounds=(padding, plot_top, plot_right, plot_bottom),
            )

        for start, end, label, category in DECIBEL_BANDS:
            # Convert the lower/upper dBFS bounds into canvas y coordinates.
            y1 = db_to_y(end, plot_top, plot_bottom)
            y2 = db_to_y(start, plot_top, plot_bottom)
            color = THRESHOLD_COLORS.get(category, "#94a3b8")
            self.trace_canvas.create_rectangle(
                padding,
                y1,
                plot_right,
                y2,
                fill=color,
                outline="",
            )
            if width >= MIN_READABLE_WIDTH:
                short_label = SHORT_BAND_LABELS.get(label, label.upper())
                self._boxed_text(
                    padding + 12,
                    (y1 + y2) / 2,
                    f"{short_label} {start} to {end} dBFS",
                    fill="#cdd5df",
                    background="#020617",
                    font=("TkDefaultFont", 10, "bold"),
                    bounds=(padding, plot_top, plot_right, plot_bottom),
                )

    def _draw_threshold_overlay(
        self,
        width: int,
        height: int,
        padding: int,
        plot_right: int,
        estimate: EmotionEstimate | None,
    ) -> None:
        """Draw threshold lines plus live and average markers on top of the graph."""
        axis_x = plot_right
        label_x = max(padding + 150, plot_right - 118)
        plot_top = padding
        plot_bottom = height - padding

        self.trace_canvas.create_line(
            axis_x,
            plot_top,
            axis_x,
            plot_bottom,
            fill="#f8fafc",
            width=2,
        )

        for threshold in DECIBEL_THRESHOLDS:
            # Draw a dark line underneath each colored line for contrast against
            # the colored threshold bands.
            y = db_to_y(threshold, plot_top, plot_bottom)
            color = THRESHOLD_LINE_COLORS[threshold]
            self.trace_canvas.create_line(
                padding,
                y,
                plot_right,
                y,
                fill="#020617",
                width=7,
            )
            self.trace_canvas.create_line(
                padding,
                y,
                plot_right,
                y,
                fill=color,
                width=4,
            )
            self.trace_canvas.create_line(
                axis_x - 10,
                y,
                axis_x,
                y,
                fill=color,
                width=4,
            )
            self._boxed_text(
                label_x,
                y,
                THRESHOLD_EMOTION_LABELS[threshold],
                fill="#020617",
                background=color,
                font=("TkDefaultFont", 10, "bold"),
                bounds=(padding, plot_top, plot_right, plot_bottom),
            )

        if estimate is not None and estimate.average_db is not None:
            # The average marker represents the last completed prediction window.
            average_y = db_to_y(estimate.average_db, plot_top, plot_bottom)
            self.trace_canvas.create_line(
                padding,
                average_y,
                plot_right,
                average_y,
                fill="#020617",
                width=7,
                dash=(7, 5),
            )
            self.trace_canvas.create_line(
                padding,
                average_y,
                plot_right,
                average_y,
                fill="#fbbf24",
                width=4,
                dash=(7, 5),
            )
            self._boxed_text(
                label_x,
                average_y - 14,
                f"avg {estimate.average_db:.1f}",
                fill="#020617",
                background="#fbbf24",
                font=("TkDefaultFont", 10, "bold"),
                bounds=(padding, plot_top, plot_right, plot_bottom),
            )

        if estimate is not None and estimate.volume_db is not None:
            # The live marker follows the most recent microphone block.
            marker_y = db_to_y(estimate.volume_db, plot_top, plot_bottom)
            self.trace_canvas.create_line(
                padding,
                marker_y,
                plot_right,
                marker_y,
                fill="#020617",
                width=8,
            )
            self.trace_canvas.create_line(
                padding,
                marker_y,
                plot_right,
                marker_y,
                fill="#2dd4bf",
                width=4,
            )
            self._boxed_text(
                label_x,
                marker_y + 14,
                f"live {estimate.volume_db:.1f}",
                fill="#020617",
                background="#2dd4bf",
                font=("TkDefaultFont", 10, "bold"),
                bounds=(padding, plot_top, plot_right, plot_bottom),
            )

        if width >= MIN_READABLE_WIDTH:
            self.trace_canvas.create_text(
                padding,
                height - 6,
                text="dBFS scale: louder sound is higher on the graph",
                anchor="sw",
                fill="#cdd5df",
                font=("TkDefaultFont", 9),
                width=max(plot_right - padding - TEXT_SAFE_MARGIN, 120),
            )

    def _boxed_text(
        self,
        x: float,
        y: float,
        text: str,
        fill: str,
        background: str,
        font: tuple[str, int, str] = ("TkDefaultFont", 9, "bold"),
        bounds: tuple[float, float, float, float] | None = None,
        anchor: str = "w",
        pad_x: int = 5,
        pad_y: int = 3,
    ) -> None:
        """Draw readable text with a solid background, optionally clamped in bounds."""
        item = self.trace_canvas.create_text(
            x,
            y,
            text=text,
            anchor=anchor,
            fill=fill,
            font=font,
        )
        left, top, right, bottom = self.trace_canvas.bbox(item)
        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
            # Move the text only enough to keep its padded rectangle visible.
            dx = clamp_delta(left - pad_x, right + pad_x, min_x, max_x)
            dy = clamp_delta(top - pad_y, bottom + pad_y, min_y, max_y)
            if dx or dy:
                self.trace_canvas.move(item, dx, dy)
                left, top, right, bottom = self.trace_canvas.bbox(item)
        box = self.trace_canvas.create_rectangle(
            left - pad_x,
            top - pad_y,
            right + pad_x,
            bottom + pad_y,
            fill=background,
            outline="",
        )
        # The rectangle is created after the text, so raise the text back above it.
        self.trace_canvas.tag_raise(item, box)

    def close(self) -> None:
        """Close audio resources before destroying the window."""
        self.stop_audio()
        self.master.destroy()


def db_to_unit(decibel: float) -> float:
    """Convert a dBFS value into a 0-to-1 position within the graph range."""
    clipped = min(max(float(decibel), DECIBEL_MIN), DECIBEL_MAX)
    return (clipped - DECIBEL_MIN) / (DECIBEL_MAX - DECIBEL_MIN)


def db_to_y(decibel: float, plot_top: int, plot_bottom: int) -> float:
    """Convert a dBFS value into a Tkinter canvas y coordinate."""
    return plot_bottom - db_to_unit(decibel) * (plot_bottom - plot_top)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Limit ``value`` so it stays between ``minimum`` and ``maximum``."""
    return min(max(value, minimum), maximum)


def clamp_delta(left: float, right: float, minimum: float, maximum: float) -> float:
    """Return how far an interval must move to fit inside another interval."""
    if right - left > maximum - minimum:
        return minimum - left
    if left < minimum:
        return minimum - left
    if right > maximum:
        return maximum - right
    return 0.0


def sensitivity_boost_percent(boost_db: int) -> int:
    """Convert the analyzer's dB boost value into a slider percent label."""
    boost_range = SENSITIVITY_BOOST_MAX_DB - SENSITIVITY_BOOST_MIN_DB
    if boost_range <= 0:
        return 100
    fraction = (boost_db - SENSITIVITY_BOOST_MIN_DB) / boost_range
    return round(clamp(fraction, 0, 1) * 100)


def default_input_device(sd) -> int | None:
    """Return sounddevice's default input id in a normalized form."""
    devices = sd.default.device
    if isinstance(devices, (list, tuple)):
        default_input = devices[0] if len(devices) > 0 else None
    else:
        default_input = devices
    return normalize_device_id(default_input)


def normalize_device_id(device_id) -> int | None:
    """Convert a possible device id into a non-negative integer or ``None``."""
    if device_id is None:
        return None
    try:
        value = int(device_id)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def sample_rate_candidates(*device_rates: float) -> list[int]:
    """Build a unique list of sample rates to try for the selected microphone."""
    candidates: list[int] = []
    for rate in (*device_rates, 48_000, 44_100, 32_000, DEFAULT_SAMPLE_RATE, 8_000):
        try:
            rounded_rate = int(round(float(rate)))
        except (TypeError, ValueError):
            continue
        if rounded_rate > 0 and rounded_rate not in candidates:
            candidates.append(rounded_rate)
    return candidates


def open_started_input_stream(sd, input_device: int, input_sample_rate: float, callback):
    """Try common sample rates until an input stream opens and starts."""
    last_error: Exception | None = None
    for sample_rate in sample_rate_candidates(input_sample_rate):
        stream = None
        try:
            stream = sd.InputStream(
                device=input_device,
                samplerate=sample_rate,
                blocksize=BLOCK_SIZE,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            stream.start()
            return stream, sample_rate, None
        except Exception as exc:
            last_error = exc
            close_stream(stream)
    return None, DEFAULT_SAMPLE_RATE, last_error or "No compatible input sample rate"


def close_stream(stream) -> None:
    """Stop and close an audio stream, ignoring cleanup errors."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass


def load_sounddevice():
    """Import sounddevice lazily so the app can explain missing audio setup."""
    try:
        import sounddevice as sd

        return sd, None
    except ImportError:
        return (
            None,
            "Install sounddevice with:\n"
            "python3 -m pip install --user --break-system-packages sounddevice",
        )
    except OSError as exc:
        return (
            None,
            f"{exc}\n\nOn Ubuntu/Debian, install PortAudio with:\n"
            "sudo apt install libportaudio2 portaudio19-dev",
        )


def format_value(
    value: float | None, empty: str, suffix: str, decimals: int
) -> str:
    """Format a possible numeric value for a Tkinter label."""
    if value is None:
        return empty
    return f"{value:.{decimals}f}{suffix}"


def format_category(category: str) -> str:
    """Convert an internal category key into the label shown in the GUI."""
    if category == "unknown":
        return "--"
    return category.title()
