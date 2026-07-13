"""Tkinter interface for the Arduino heart-rate monitor."""

import tkinter as tk
from dataclasses import fields
from tkinter import messagebox, ttk

import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bpm_core import (
    HeartRateProcessor,
    SerialSampleReader,
    SignalSettings,
    available_serial_ports,
)


UPDATE_INTERVAL_MS = 50
DEFAULT_BAUD_RATE = 115200


class HeartRateMonitorApp:
    """Build and coordinate the GUI without implementing signal processing."""

    SETTING_LABELS = {
        "sample_window": "Graph sample window",
        "baseline_window": "Baseline window",
        "smoothing_window": "Smoothing window",
        "min_beat_interval_ms": "Minimum beat interval (ms)",
        "max_beat_interval_ms": "Maximum beat interval (ms)",
        "threshold_offset": "Minimum threshold offset",
        "noise_threshold_multiplier": "Noise threshold multiplier",
        "bpm_average_count": "BPM average count",
        "stale_bpm_ms": "BPM timeout (ms)",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.processor = HeartRateProcessor()
        self.reader = SerialSampleReader()
        self.setting_vars = {}
        self._build_window()
        self._build_header()
        self._build_connection_controls()
        self._build_main_content()
        self.refresh_ports()
        self.root.after(UPDATE_INTERVAL_MS, self.update_interface)

    def _build_window(self) -> None:
        self.root.title("Arduino Heart Rate Monitor")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 700)
        try:
            self.root.attributes("-zoomed", True)
        except tk.TclError:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=(20, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Heart Rate Monitor", font=("Arial", 28, "bold")).pack()

    def _build_connection_controls(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Serial connection", padding=10)
        frame.pack(fill="x", padx=20, pady=(0, 8))

        ttk.Label(frame, text="Port").pack(side="left")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            frame, textvariable=self.port_var, width=25, state="readonly"
        )
        self.port_combo.pack(side="left", padx=(6, 8))
        ttk.Button(frame, text="Refresh", command=self.refresh_ports).pack(side="left")

        ttk.Label(frame, text="Baud").pack(side="left", padx=(18, 0))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD_RATE))
        ttk.Entry(frame, textvariable=self.baud_var, width=10).pack(side="left", padx=6)
        self.connect_button = ttk.Button(frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side="left", padx=6)
        self.connection_label = ttk.Label(frame, text="Disconnected")
        self.connection_label.pack(side="left", padx=12)

    def _build_main_content(self) -> None:
        content = ttk.Frame(self.root, padding=(20, 0, 20, 15))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        self._build_readout(content)
        self._build_chart(content)
        self._build_settings(content)

    def _build_readout(self, parent) -> None:
        frame = ttk.Frame(parent, padding=(5, 5, 5, 10))
        frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.bpm_label = ttk.Label(frame, text="-- BPM", font=("Arial", 42, "bold"))
        self.bpm_label.pack(side="left")
        self.status_label = ttk.Label(frame, text="Select a port and connect", font=("Arial", 14))
        self.status_label.pack(side="left", padx=30)
        self.health_label = ttk.Label(
            frame,
            text="Adult resting reference: Low <60 | Typical 60–100 | High >100 BPM",
            font=("Arial", 11),
        )
        self.health_label.pack(side="left", padx=10)
        self.signal_label = ttk.Label(frame, text="Signal: 0")
        self.signal_label.pack(side="right", padx=12)
        self.threshold_label = ttk.Label(frame, text="Threshold: 0.0")
        self.threshold_label.pack(side="right", padx=12)

    def _build_chart(self, parent) -> None:
        chart_frame = ttk.Frame(parent)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 15))
        figure = Figure(figsize=(10, 5), dpi=100)
        self.axis = figure.add_subplot(111)
        (self.signal_line,) = self.axis.plot([], [], linewidth=2.0, label="Pulse signal")
        (self.threshold_line,) = self.axis.plot(
            [], [], linestyle="--", linewidth=1.5, label="Detection threshold"
        )
        self.axis.set_title("Live Pulse Signal", fontweight="bold")
        self.axis.set_xlabel("Time (seconds)")
        self.axis.set_ylabel("Sensor value")
        self.axis.grid(True, alpha=0.4)
        self.axis.legend(loc="upper right")
        figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_settings(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Signal processing", padding=12)
        frame.grid(row=1, column=1, sticky="nsew")
        current = self.processor.settings
        for row, setting_field in enumerate(fields(SignalSettings)):
            name = setting_field.name
            ttk.Label(frame, text=self.SETTING_LABELS[name]).grid(
                row=row, column=0, sticky="w", pady=5
            )
            variable = tk.StringVar(value=str(getattr(current, name)))
            self.setting_vars[name] = variable
            ttk.Entry(frame, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="e", padx=(12, 0), pady=5
            )
        ttk.Button(frame, text="Apply settings", command=self.apply_settings).grid(
            row=len(self.setting_vars), column=0, columnspan=2, sticky="ew", pady=(15, 5)
        )
        ttk.Button(frame, text="Restore defaults", command=self.restore_defaults).grid(
            row=len(self.setting_vars) + 1, column=0, columnspan=2, sticky="ew"
        )

    def refresh_ports(self) -> None:
        selected = self.port_var.get()
        ports = available_serial_ports()
        self.port_combo["values"] = ports
        if selected in ports:
            self.port_var.set(selected)
        elif ports:
            self.port_var.set(ports[0])
        else:
            self.port_var.set("")
            self.connection_label.config(text="No serial ports found")

    def toggle_connection(self) -> None:
        if self.reader.is_open:
            self.reader.disconnect()
            self.processor.reset()
            self.connect_button.config(text="Connect")
            self.connection_label.config(text="Disconnected")
            self.port_combo.config(state="readonly")
            return
        try:
            baud_rate = int(self.baud_var.get())
            self.reader.connect(self.port_var.get(), baud_rate)
        except (ValueError, serial.SerialException) as error:
            messagebox.showerror("Serial connection error", str(error), parent=self.root)
            return
        self.processor.reset()
        self.connect_button.config(text="Disconnect")
        self.connection_label.config(text=f"Connected to {self.reader.port}")
        self.port_combo.config(state="disabled")

    def apply_settings(self) -> None:
        try:
            values = {}
            for setting_field in fields(SignalSettings):
                text = self.setting_vars[setting_field.name].get().strip()
                converter = (
                    float
                    if setting_field.name
                    in {"threshold_offset", "noise_threshold_multiplier"}
                    else int
                )
                values[setting_field.name] = converter(text)
            self.processor.apply_settings(SignalSettings(**values))
        except ValueError as error:
            messagebox.showerror("Invalid setting", str(error), parent=self.root)
            return
        self.status_label.config(text="Signal-processing settings applied")

    def restore_defaults(self) -> None:
        defaults = SignalSettings()
        for name, variable in self.setting_vars.items():
            variable.set(str(getattr(defaults, name)))
        self.processor.apply_settings(defaults)
        self.status_label.config(text="Default settings restored")

    def update_interface(self) -> None:
        if self.reader.is_open:
            try:
                for timestamp_ms, sensor_value in self.reader.read_samples():
                    self.processor.process_sample(timestamp_ms, sensor_value)
            except serial.SerialException as error:
                self.reader.disconnect()
                self.connect_button.config(text="Connect")
                self.port_combo.config(state="readonly")
                self.connection_label.config(text=f"Connection lost: {error}")

        state = self.processor
        self.signal_label.config(text=f"Signal: {state.current_signal}")
        self.threshold_label.config(text=f"Threshold: {state.current_threshold:.1f}")
        self.bpm_label.config(text=f"{state.current_bpm or '--'} BPM")
        if state.current_bpm:
            if state.current_bpm < 60:
                resting_band = "Low resting range (<60 BPM)"
                color = "#b36b00"
            elif state.current_bpm <= 100:
                resting_band = "Typical adult resting range (60–100 BPM)"
                color = "#16803a"
            else:
                resting_band = "High resting range (>100 BPM)"
                color = "#b3261e"
            self.health_label.config(text=resting_band, foreground=color)
        else:
            self.health_label.config(
                text="Adult resting reference: Low <60 | Typical 60–100 | High >100 BPM",
                foreground="",
            )
        if self.reader.is_open:
            self.status_label.config(text=state.status_message)
        self._update_chart()
        self.root.after(UPDATE_INTERVAL_MS, self.update_interface)

    def _update_chart(self) -> None:
        if not self.processor.signal_values:
            return
        x_data = list(self.processor.time_values)
        y_data = list(self.processor.signal_values)
        self.signal_line.set_data(x_data, y_data)
        self.threshold_line.set_data(
            x_data, [self.processor.current_threshold] * len(x_data)
        )
        x_start = x_data[0]
        self.axis.set_xlim(x_start, max(x_data[-1], x_start + 1))
        minimum, maximum = min(y_data), max(y_data)
        margin = max(20, (maximum - minimum) * 0.25)
        self.axis.set_ylim(minimum - margin, maximum + margin)
        self.canvas.draw_idle()

    def close(self) -> None:
        self.reader.disconnect()
        self.root.destroy()


def run() -> None:
    root = tk.Tk()
    HeartRateMonitorApp(root)
    root.mainloop()
