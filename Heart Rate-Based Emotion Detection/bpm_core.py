"""Heart-rate signal processing and serial communication.

This module deliberately contains no Tkinter or other GUI code so it can be
reused by command-line programs and unit tests.
"""

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Optional

import serial
from serial.tools import list_ports


@dataclass(frozen=True)
class SignalSettings:
    """Tunable settings used by :class:`HeartRateProcessor`."""

    sample_window: int = 500
    baseline_window: int = 200
    smoothing_window: int = 5
    min_beat_interval_ms: int = 300
    max_beat_interval_ms: int = 2000
    threshold_offset: float = 10.0
    noise_threshold_multiplier: float = 3.0
    bpm_average_count: int = 5
    stale_bpm_ms: int = 4000

    def validate(self) -> None:
        if self.sample_window < 2:
            raise ValueError("Sample window must be at least 2.")
        if self.baseline_window < 2:
            raise ValueError("Baseline window must be at least 2.")
        if self.smoothing_window < 1:
            raise ValueError("Smoothing window must be at least 1.")
        if self.min_beat_interval_ms <= 0:
            raise ValueError("Minimum beat interval must be greater than 0.")
        if self.max_beat_interval_ms <= self.min_beat_interval_ms:
            raise ValueError("Maximum beat interval must exceed the minimum.")
        if self.threshold_offset <= 0:
            raise ValueError("Threshold offset must be greater than 0.")
        if self.noise_threshold_multiplier <= 0:
            raise ValueError("Noise multiplier must be greater than 0.")
        if self.bpm_average_count < 1:
            raise ValueError("BPM average count must be at least 1.")
        if self.stale_bpm_ms <= 0:
            raise ValueError("BPM timeout must be greater than 0.")


class HeartRateProcessor:
    """Maintain pulse-signal state and estimate heart rate from samples."""

    def __init__(self, settings: Optional[SignalSettings] = None):
        self.settings = settings or SignalSettings()
        self.settings.validate()
        self.signal_values = deque(maxlen=self.settings.sample_window)
        self.time_values = deque(maxlen=self.settings.sample_window)
        self.baseline_values = deque(maxlen=self.settings.baseline_window)
        self.smoothing_values = deque(maxlen=self.settings.smoothing_window)
        self.recent_intervals = deque(maxlen=self.settings.bpm_average_count)
        self.reset()

    def reset(self) -> None:
        """Clear measurements while retaining the current settings."""

        self.signal_values.clear()
        self.time_values.clear()
        self.baseline_values.clear()
        self.smoothing_values.clear()
        self.recent_intervals.clear()
        self.last_beat_time = None
        self.last_detected_beat_time = None
        self.signal_was_high = False
        self.pulse_peak_time = None
        self.pulse_peak_value = float("-inf")
        self.current_bpm = 0
        self.current_threshold = 0.0
        self.current_baseline = 0.0
        self.current_noise = 0.0
        self.current_signal = 0
        self.current_smoothed_signal = 0.0
        self.beat_count = 0

    def apply_settings(self, settings: SignalSettings) -> None:
        """Validate and apply settings, then recalibrate the detector."""

        settings.validate()
        self.settings = settings
        self.signal_values = deque(maxlen=settings.sample_window)
        self.time_values = deque(maxlen=settings.sample_window)
        self.baseline_values = deque(maxlen=settings.baseline_window)
        self.smoothing_values = deque(maxlen=settings.smoothing_window)
        self.recent_intervals = deque(maxlen=settings.bpm_average_count)
        self.reset()

    @property
    def warmup_sample_count(self) -> int:
        """Number of smoothed samples required before peak detection starts."""

        return min(100, self.settings.baseline_window)

    @property
    def is_calibrated(self) -> bool:
        return len(self.baseline_values) >= self.warmup_sample_count

    @property
    def status_message(self) -> str:
        if not self.is_calibrated:
            return "Calibrating signal — keep your finger still"
        if self.current_bpm:
            return "Heart rate detected"
        if self.current_noise < 0.5:
            return "Pulse signal is too flat — check sensor contact"
        return "Detecting heartbeat — keep your finger still"

    def process_sample(self, timestamp_ms: int, sensor_value: int) -> bool:
        """Process one sample and return ``True`` when a valid beat is found."""

        self.current_signal = sensor_value
        self.time_values.append(timestamp_ms / 1000.0)
        self.signal_values.append(sensor_value)
        self.smoothing_values.append(sensor_value)
        self.current_smoothed_signal = sum(self.smoothing_values) / len(
            self.smoothing_values
        )
        self.baseline_values.append(self.current_smoothed_signal)

        beat_detected = False
        self._expire_old_measurement(timestamp_ms)

        if not self.is_calibrated:
            self.current_baseline = median(self.baseline_values)
            self.current_threshold = (
                self.current_baseline + self.settings.threshold_offset
            )
            return False

        # The Arduino pulse waveform is positive-going. Using its lower band
        # keeps frequent pulse peaks from pulling the baseline upward at high
        # heart rates, while a median deviation rejects occasional low spikes.
        ordered_values = sorted(self.baseline_values)
        baseline_index = round((len(ordered_values) - 1) * 0.20)
        self.current_baseline = ordered_values[baseline_index]
        lower_band = ordered_values[: max(5, len(ordered_values) // 4)]
        lower_center = median(lower_band)
        absolute_deviations = [
            abs(value - lower_center) for value in lower_band
        ]
        self.current_noise = 1.4826 * median(absolute_deviations)
        adaptive_offset = max(
            self.settings.threshold_offset,
            self.settings.noise_threshold_multiplier * self.current_noise,
        )
        self.current_threshold = self.current_baseline + adaptive_offset

        if (
            self.current_smoothed_signal > self.current_threshold
            and not self.signal_was_high
        ):
            self.signal_was_high = True
            self.pulse_peak_time = timestamp_ms
            self.pulse_peak_value = self.current_smoothed_signal

        if self.signal_was_high and self.current_smoothed_signal > self.pulse_peak_value:
            self.pulse_peak_time = timestamp_ms
            self.pulse_peak_value = self.current_smoothed_signal

        # Hysteresis lets one complete pulse finish before its peak is tested.
        reset_level = self.current_baseline + adaptive_offset * 0.5
        if self.signal_was_high and self.current_smoothed_signal < reset_level:
            self.signal_was_high = False
            if self.pulse_peak_time is not None:
                beat_detected = self._register_peak(self.pulse_peak_time)
            self.pulse_peak_time = None
            self.pulse_peak_value = float("-inf")

        return beat_detected

    def _register_peak(self, peak_time: int) -> bool:
        """Accept a completed peak only when its timing is physiologically valid."""

        if self.last_beat_time is None:
            self.last_beat_time = peak_time
            return False

        observed_interval = peak_time - self.last_beat_time
        if observed_interval < self.settings.min_beat_interval_ms:
            # Do not move last_beat_time: a noise spike must not disrupt the
            # timing between the surrounding real beats.
            return False

        interval = observed_interval
        if len(self.recent_intervals) >= 3:
            center_interval = median(self.recent_intervals)
            missed_beat_count = min(
                3, max(1, round(observed_interval / center_interval))
            )
            normalized_interval = observed_interval / missed_beat_count
            if abs(normalized_interval - center_interval) > center_interval * 0.25:
                # A peak far from the established rhythm is usually motion.
                # Leave last_beat_time on the last reliable peak so the next
                # genuine beat can still produce the correct interval.
                return False
            interval = normalized_interval

        if interval > self.settings.max_beat_interval_ms:
            # This peak begins a fresh pair; one peak alone cannot produce BPM.
            self.last_beat_time = peak_time
            self.recent_intervals.clear()
            self.current_bpm = 0
            return False

        self.last_beat_time = peak_time
        self.last_detected_beat_time = peak_time
        self.recent_intervals.append(interval)
        center_interval = median(self.recent_intervals)
        inlier_intervals = [
            value
            for value in self.recent_intervals
            if abs(value - center_interval) <= center_interval * 0.25
        ]
        # With an even number of widely separated intervals, the median can
        # lie far enough between them that neither falls inside the inlier
        # band. Use the median itself until later beats establish a cluster.
        average_interval = (
            sum(inlier_intervals) / len(inlier_intervals)
            if inlier_intervals
            else center_interval
        )
        self.current_bpm = round(60000.0 / average_interval)
        self.beat_count += 1
        return True

    def _expire_old_measurement(self, timestamp_ms: int) -> None:
        """Clear BPM after the signal has stopped producing valid beats."""

        if (
            self.last_detected_beat_time is not None
            and timestamp_ms - self.last_detected_beat_time
            > self.settings.stale_bpm_ms
        ):
            self.current_bpm = 0
            self.recent_intervals.clear()


def available_serial_ports() -> list[str]:
    """Return device names for serial ports currently visible to the OS."""

    return [port.device for port in sorted(list_ports.comports())]


class SerialSampleReader:
    """Own a non-blocking serial connection and parse sensor samples."""

    def __init__(self):
        self.connection: Optional[serial.Serial] = None
        self.port: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.connection is not None and self.connection.is_open

    def connect(self, port: str, baud_rate: int) -> None:
        if not port:
            raise ValueError("Select a serial port first.")
        if baud_rate <= 0:
            raise ValueError("Baud rate must be greater than 0.")
        self.disconnect()
        connection = serial.Serial(port=port, baudrate=baud_rate, timeout=0)
        try:
            connection.reset_input_buffer()
        except Exception:
            connection.close()
            raise
        self.connection = connection
        self.port = port

    def disconnect(self) -> None:
        if self.connection is not None and self.connection.is_open:
            self.connection.close()
        self.connection = None
        self.port = None

    def read_samples(self) -> Iterable[tuple[int, int]]:
        """Yield all complete, valid samples that are currently buffered."""

        if not self.is_open:
            return
        while self.connection.in_waiting > 0:
            line = self.connection.readline().decode("utf-8", errors="ignore").strip()
            try:
                timestamp_text, sensor_text = line.split(",", maxsplit=1)
                yield int(timestamp_text), int(sensor_text)
            except ValueError:
                # Arduino startup text and incomplete lines are harmless.
                continue
