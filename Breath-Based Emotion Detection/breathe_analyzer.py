"""Windowed decibel analysis and emotion-guessing logic.

This file contains the non-GUI part of the app. It receives small chunks of
microphone audio, measures how loud the voice-frequency range is, and converts
each completed 32-second window into a simple emotion estimate.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np


# The app waits for a full 32 seconds of sound before making each prediction.
PREDICTION_WINDOW_SECONDS = 32

# A small tolerance keeps tiny timing differences from making a nearly complete
# audio window look invalid.
WINDOW_TOLERANCE_SECONDS = 0.5

# Keep enough recent points to draw the live graph without storing audio forever.
HISTORY_SECONDS = 45

# The analyzer can work with other sample rates, but this is the fallback used
# when an audio device does not report a useful rate.
DEFAULT_SAMPLE_RATE = 16_000

# Most speech energy lives roughly between these frequencies, so the analyzer
# focuses on them instead of all background noise.
VOICE_BAND_LOW_HZ = 80
VOICE_BAND_HIGH_HZ = 4_000

# Voice-band RMS values can be very small after filtering, so this boost makes
# the displayed dBFS values easier to compare with the app's thresholds.
VOICE_BAND_EXTRA_BOOST_DB = 12

# The GUI sensitivity slider adds an extra dB boost within this range.
SENSITIVITY_BOOST_MIN_DB = 0
SENSITIVITY_BOOST_MAX_DB = 30

# dBFS values are normally <= 0. The graph clips values to this readable range.
DECIBEL_MIN = -80
DECIBEL_MAX = 0

# Each tuple means: lower bound, upper bound, text shown on graph, category key.
DECIBEL_BANDS = (
    (-80, -55, "very quiet / unclear", "unknown"),
    (-55, -42, "quiet / calm", "calm"),
    (-42, -30, "moderate / active", "focused"),
    (-30, 0, "loud / stressed", "stressed"),
)

# These are the visible cut lines between the broad decibel categories.
DECIBEL_THRESHOLDS = (-55, -42, -30)


@dataclass
class EmotionEstimate:
    """A complete prediction result that the GUI can display.

    ``volume_db`` is the current live loudness when available. ``average_db`` is
    only filled after a 32-second window has completed.
    """

    label: str
    message: str
    category: str = "unknown"
    volume_db: float | None = None
    average_db: float | None = None
    window_seconds: float = 0.0
    confidence: float = 0.0


class BreathingAnalyzer:
    """Stores microphone loudness and estimates emotion from 32-second windows."""

    def __init__(self) -> None:
        # Deques let us efficiently add new readings at the end and remove old
        # readings from the front as time moves forward.
        self.live_db_history: deque[tuple[float, float]] = deque()
        self.window_points: deque[tuple[float, float]] = deque()
        self.average_db_history: deque[tuple[float, float]] = deque()
        self.window_start_time: float | None = None
        self.last_completed_estimate: EmotionEstimate | None = None
        self.sensitivity_boost_db = SENSITIVITY_BOOST_MAX_DB

    def clear(self) -> None:
        """Forget collected audio points and start the next window from scratch."""
        self.live_db_history.clear()
        self.window_points.clear()
        self.average_db_history.clear()
        self.window_start_time = None
        self.last_completed_estimate = None

    def change_sensitivity_boost(self, delta_db: int) -> int:
        """Move the sensitivity boost up or down by ``delta_db``."""
        return self.set_sensitivity_boost(self.sensitivity_boost_db + delta_db)

    def set_sensitivity_boost(self, boost_db: int) -> int:
        """Set the sensitivity boost and clear old readings if it changed."""
        new_boost = min(
            max(int(boost_db), SENSITIVITY_BOOST_MIN_DB),
            SENSITIVITY_BOOST_MAX_DB,
        )
        if new_boost != self.sensitivity_boost_db:
            # Old readings were measured with a different boost, so mixing them
            # with new readings would make the graph and prediction misleading.
            self.sensitivity_boost_db = new_boost
            self.clear()
        return self.sensitivity_boost_db

    def add_audio_block(
        self,
        audio: np.ndarray,
        timestamp: float | None = None,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """Convert one microphone audio block into one voice-sensitive dBFS point."""
        # Sounddevice may give one channel or several channels. The analyzer only
        # needs one loudness value, so stereo input is averaged to mono.
        audio = to_mono_float(audio)

        # Measure only the frequency range where human speech is most likely to
        # appear, then convert the RMS value to dBFS.
        voice_db = rms_to_db(voice_band_rms(audio, sample_rate))
        now = time.monotonic() if timestamp is None else float(timestamp)
        adjusted_db = self.adjust_decibel(voice_db + VOICE_BAND_EXTRA_BOOST_DB)

        # Store the same point in two places: one history for the live graph, and
        # one list for the current 32-second prediction window.
        self.live_db_history.append((now, adjusted_db))
        if self.window_start_time is None:
            self.window_start_time = now
        self.window_points.append((now, adjusted_db))
        self._complete_ready_windows(now)

        # Remove graph points that are too old to display.
        cutoff = now - HISTORY_SECONDS
        while self.live_db_history and self.live_db_history[0][0] < cutoff:
            self.live_db_history.popleft()
        while self.average_db_history and self.average_db_history[0][0] < cutoff:
            self.average_db_history.popleft()

    def analyze(self) -> EmotionEstimate:
        """Return the best current estimate without modifying collected audio."""
        live_db = self.current_live_db()
        progress = self.current_window_progress()

        if self.last_completed_estimate is not None:
            # Keep showing the last completed prediction while the next window is
            # collecting, because a new estimate is only available every 32 sec.
            message = (
                f"{self.last_completed_estimate.message} "
                f"Next {PREDICTION_WINDOW_SECONDS}-second window is collecting."
            )
            return replace(
                self.last_completed_estimate,
                message=message,
                volume_db=live_db,
                window_seconds=progress,
            )

        if live_db is None:
            # No microphone samples have reached the analyzer yet.
            return EmotionEstimate(
                "Waiting for audio",
                "Select a microphone, press Start, and make sound near the microphone.",
            )

        return EmotionEstimate(
            "Listening",
            f"Collecting a {PREDICTION_WINDOW_SECONDS}-second decibel window.",
            volume_db=live_db,
            window_seconds=progress,
            confidence=min(progress / PREDICTION_WINDOW_SECONDS, 0.95),
        )

    def current_live_db(self) -> float | None:
        """Return the newest adjusted dBFS value, or ``None`` before audio starts."""
        if not self.live_db_history:
            return None
        return self.live_db_history[-1][1]

    def current_window_progress(self) -> float:
        """Return how many seconds of the current prediction window are filled."""
        if self.window_start_time is None or not self.live_db_history:
            return 0.0

        progress = self.live_db_history[-1][0] - self.window_start_time
        return min(max(float(progress), 0.0), PREDICTION_WINDOW_SECONDS)

    def db_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return live adjusted decibel points with absolute timestamps."""
        if not self.live_db_history:
            return np.array([]), np.array([])
        return absolute_points_to_arrays(list(self.live_db_history))

    def average_db_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return completed-window average adjusted decibel points."""
        if not self.average_db_history:
            return np.array([]), np.array([])
        return absolute_points_to_arrays(list(self.average_db_history))

    def adjust_decibel(self, decibel: float) -> float:
        """Apply the sensitivity slider and clip the result to the graph range."""
        decibel += self.sensitivity_boost_db
        return min(max(decibel, DECIBEL_MIN), DECIBEL_MAX)

    def _complete_ready_windows(self, now: float) -> None:
        """Finish every 32-second window that has enough time behind it."""
        if self.window_start_time is None:
            return

        while now - self.window_start_time >= PREDICTION_WINDOW_SECONDS:
            window_end = self.window_start_time + PREDICTION_WINDOW_SECONDS
            # Keep only the points that actually belong to this completed window.
            points = [
                point
                for point in self.window_points
                if self.window_start_time <= point[0] <= window_end
            ]
            self.last_completed_estimate = self._estimate_completed_window(points)
            if self.last_completed_estimate.average_db is not None:
                self.average_db_history.append(
                    (window_end, self.last_completed_estimate.average_db)
                )

            # Drop completed points so the next window starts cleanly.
            while self.window_points and self.window_points[0][0] <= window_end:
                self.window_points.popleft()
            self.window_start_time = window_end

    def _estimate_completed_window(
        self, points: list[tuple[float, float]]
    ) -> EmotionEstimate:
        """Classify one completed prediction window from its decibel points."""
        if len(points) < 10:
            return EmotionEstimate(
                "Sound not clear",
                "The completed 32-second window did not contain enough audio.",
                window_seconds=PREDICTION_WINDOW_SECONDS,
                confidence=0.1,
            )

        times, values = absolute_points_to_arrays(points)
        duration = float(times[-1] - times[0])
        average_db = float(values.mean())
        variability = float(values.std())

        # A window can have many points but still be too short if the stream was
        # interrupted. Duration catches that case.
        if duration < PREDICTION_WINDOW_SECONDS - WINDOW_TOLERANCE_SECONDS:
            return EmotionEstimate(
                "Sound not clear",
                "The completed 32-second window did not contain enough audio.",
                average_db=average_db,
                window_seconds=PREDICTION_WINDOW_SECONDS,
                confidence=0.1,
            )

        label, message, category = classify_decibel(average_db, windowed=True)
        return EmotionEstimate(
            label=label,
            message=message,
            category=category,
            volume_db=average_db,
            average_db=average_db,
            window_seconds=PREDICTION_WINDOW_SECONDS,
            confidence=estimate_confidence(average_db, duration, variability),
        )


def absolute_points_to_arrays(
    points: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[(time, value), ...]`` into NumPy arrays for plotting/math."""
    times, values = zip(*points)
    return np.array(times, dtype=float), np.array(values, dtype=float)


def rms_to_db(rms: float) -> float:
    """Convert a root-mean-square amplitude into dBFS."""
    return 20 * math.log10(max(float(rms), 1e-9))


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert microphone samples to clean mono floating-point samples."""
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    # Replace NaN or infinite values so a bad audio block cannot poison the math.
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


def voice_band_rms(
    audio: np.ndarray,
    sample_rate: float,
    low_hz: int = VOICE_BAND_LOW_HZ,
    high_hz: int = VOICE_BAND_HIGH_HZ,
) -> float:
    """Measure energy in the part of the signal where speech is usually strongest."""
    if audio.size == 0:
        return 0.0

    # Remove the DC offset so a shifted waveform is not mistaken for loud sound.
    samples = audio - float(audio.mean())
    sample_rate = max(float(sample_rate), 1.0)

    # Frequencies above half the sample rate cannot be represented reliably
    # (Nyquist limit), so cap the requested voice band there.
    high_hz = min(float(high_hz), sample_rate / 2)
    if samples.size < 8 or high_hz <= low_hz:
        return rms(samples)

    # The Hanning window softens the edges of the block before the FFT, which
    # reduces artificial frequency spikes caused by cutting the audio into chunks.
    window = np.hanning(samples.size)
    window_scale = max(rms(window), 1e-9)
    spectrum = np.fft.rfft(samples * window)
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    voice_bins = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(voice_bins):
        return rms(samples)

    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[voice_bins] = spectrum[voice_bins]
    voice_signal = np.fft.irfft(filtered_spectrum, n=samples.size)
    return rms(voice_signal) / window_scale


def rms(samples: np.ndarray) -> float:
    """Return the root-mean-square amplitude of the samples."""
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def classify_decibel(decibel: float, windowed: bool = False) -> tuple[str, str, str]:
    """Map a dBFS value to a broad display label and category."""
    scope = "32-second average" if windowed else "live"
    threshold_db = round(decibel, 6)

    if threshold_db < -55:
        return (
            "Sound too quiet to tell",
            f"The {scope} decibel level is very low.",
            "unknown",
        )

    if threshold_db < -42:
        return (
            "Likely calm",
            f"The {scope} decibel level is quiet.",
            "calm",
        )

    if threshold_db < -30:
        return (
            "Possibly focused or active",
            f"The {scope} decibel level is moderate.",
            "focused",
        )

    return (
        "Possibly stressed or excited",
        f"The {scope} decibel level is loud.",
        "stressed",
    )


def estimate_confidence(decibel: float, duration: float, variability: float) -> float:
    """Combine duration, loudness, and steadiness into a simple 0-to-1 score."""
    duration_score = min(duration / PREDICTION_WINDOW_SECONDS, 1)
    loudness_score = min(max((decibel + 60) / 35, 0), 1)
    stability_score = 1 - min(variability / 18, 0.55)
    return 0.40 * duration_score + 0.35 * loudness_score + 0.25 * stability_score
