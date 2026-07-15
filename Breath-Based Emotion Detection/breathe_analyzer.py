"""Live decibel analysis and signal-label estimation logic.

This file contains the non-GUI part of the app. It receives small chunks of
microphone audio, measures how loud the voice-frequency range is, and converts
each live decibel measurement into a simple signal estimate.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


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


@dataclass
class SignalEstimate:
    """A live decibel-based estimate result that the GUI can display."""

    label: str
    message: str
    category: str = "unknown"
    volume_db: float | None = None


class BreathingAnalyzer:
    """Store microphone loudness and estimate a label from the latest reading."""

    def __init__(self) -> None:
        # Deques let us efficiently add new readings at the end and remove old
        # readings from the front as time moves forward.
        self.live_db_history: deque[tuple[float, float]] = deque()
        self.sensitivity_boost_db = SENSITIVITY_BOOST_MAX_DB

    def clear(self) -> None:
        """Forget collected audio points."""
        self.live_db_history.clear()

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
            # with new readings would make the graph and estimate misleading.
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

        # Store the point for the live graph and current display.
        self.live_db_history.append((now, adjusted_db))

        # Remove graph points that are too old to display.
        cutoff = now - HISTORY_SECONDS
        while self.live_db_history and self.live_db_history[0][0] < cutoff:
            self.live_db_history.popleft()

    def analyze(self) -> SignalEstimate:
        """Return the best current estimate without modifying collected audio."""
        live_db = self.current_live_db()

        if live_db is None:
            # No microphone samples have reached the analyzer yet.
            return SignalEstimate(
                "Waiting for audio",
                "Select a microphone, press Start, and make sound near the microphone.",
            )

        label, message, category = classify_decibel(live_db)
        return SignalEstimate(label, message, category, live_db)

    def current_live_db(self) -> float | None:
        """Return the newest adjusted dBFS value, or ``None`` before audio starts."""
        if not self.live_db_history:
            return None
        return self.live_db_history[-1][1]

    def db_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return live adjusted decibel points with absolute timestamps."""
        if not self.live_db_history:
            return np.array([]), np.array([])
        return absolute_points_to_arrays(list(self.live_db_history))

    def adjust_decibel(self, decibel: float) -> float:
        """Apply the sensitivity slider and clip the result to the graph range."""
        decibel += self.sensitivity_boost_db
        return min(max(decibel, DECIBEL_MIN), DECIBEL_MAX)


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


def classify_decibel(decibel: float) -> tuple[str, str, str]:
    """Map a dBFS value to a broad signal-level label and category."""
    threshold_db = round(decibel, 6)

    if threshold_db < -55:
        return (
            "Sound too quiet to tell",
            "The live decibel level is very low.",
            "unknown",
        )

    if threshold_db < -42:
        return (
            "Low signal level",
            "The live decibel level is low.",
            "low",
        )

    if threshold_db < -30:
        return (
            "Moderate signal level",
            "The live decibel level is moderate.",
            "moderate",
        )

    return (
        "High signal level",
        "The live decibel level is loud.",
        "high",
    )
