"""Reproducible FIR speech experiment. Run --help for options."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

FS = 16000


def snr(reference, estimate):
    error = np.sum((reference - estimate) ** 2)
    return float(10 * np.log10(max(np.sum(reference ** 2), 1e-30) / max(error, 1e-30)))


def apply_fir(x, b):
    # Odd tap count: compensate the integer group delay explicitly.
    delay = (len(b) - 1) // 2
    return signal.fftconvolve(x, b, mode="full")[delay:delay + len(x)]


def spectrum(x, fs):
    window = signal.windows.hann(len(x), sym=False)
    amplitude = np.abs(np.fft.rfft(x * window)) / window.sum()
    amplitude[1:-1 if len(x) % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(len(x), 1 / fs), 20 * np.log10(np.maximum(amplitude, 1e-10))


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Real speech WAV recording")
    source.add_argument("--demo", action="store_true", help="Synthetic pipeline check; not speech results")
    parser.add_argument("--output", type=Path, default=Path("outputs/speech"))
    parser.add_argument("--snr-db", type=float, default=5)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        t = np.arange(FS * 4) / FS
        x = (0.6 + 0.3 * np.sin(2 * np.pi * 3 * t)) * (np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 1100 * t))
        label = "SYNTHETIC DEMO"
        original_fs = FS
    else:
        original_fs, x = wavfile.read(args.input)
        if x.dtype.kind == "u":
            x = (x.astype(float) - 128) / 128
        elif x.dtype.kind == "i":
            x = x.astype(float) / (2 ** (x.dtype.itemsize * 8 - 1))
        else:
            x = x.astype(float)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if not np.all(np.isfinite(x)):
            raise ValueError("Recording contains nonfinite samples")
        divisor = np.gcd(original_fs, FS)
        x = signal.resample_poly(x, FS // divisor, original_fs // divisor)
        label = "Speech recording"
    if len(x) < FS * 2:
        raise ValueError("Use at least two seconds of audio")
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-8:
        raise ValueError("Recording is silent")
    x = x / np.max(np.abs(x)) * 0.7
    t = np.arange(len(x)) / FS
    tone = np.sin(2 * np.pi * 2500 * t)
    valid = slice(256, -256)  # Largest tested filter half-length, shared across designs.

    def corrupt(db):
        gain = np.sqrt(np.mean(x[valid] ** 2) / (10 ** (db / 10) * np.mean(tone[valid] ** 2)))
        return x + gain * tone

    noisy = corrupt(args.snr_db)
    designs = []
    for order in (128, 256, 512):
        for cutoff in (1800, 2100):
            designs.append(("lowpass", order, cutoff, signal.firwin(order + 1, cutoff, fs=FS)))
        for width in (200, 400, 800):
            designs.append(("bandstop", order, width, signal.firwin(order + 1, [2500-width/2, 2500+width/2], pass_zero="bandstop", fs=FS)))
    rows = []
    for kind, order, parameter, b in designs:
        y = apply_fir(noisy, b)
        clean_filtered = apply_fir(x, b)
        _, response = signal.freqz(b, worN=[2500], fs=FS)
        rows.append(dict(filter=kind, order=order, parameter_hz=parameter,
                         input_snr_db=snr(x[valid], noisy[valid]),
                         output_snr_db=snr(x[valid], y[valid]),
                         clean_distortion_snr_db=snr(x[valid], clean_filtered[valid]),
                         tone_gain_db=float(20*np.log10(max(abs(response[0]), 1e-15)))))
    save_csv(out / "filter_comparison.csv", rows)
    best_index = int(np.argmax([r["output_snr_db"] for r in rows]))
    best = designs[best_index]
    filtered = apply_fir(noisy, best[3])
    sweep = []
    for db in (-5, 0, 5, 10, 20):
        for kind in ("lowpass", "bandstop"):
            indices = [i for i, d in enumerate(designs) if d[0] == kind]
            chosen = max(indices, key=lambda i: rows[i]["output_snr_db"])
            y = apply_fir(corrupt(db), designs[chosen][3])
            sweep.append(dict(input_snr_db=db, filter=kind, order=designs[chosen][1],
                              parameter_hz=designs[chosen][2], output_snr_db=snr(x[valid], y[valid])))
    save_csv(out / "noise_sweep.csv", sweep)
    naive = x[::2]
    safe = signal.resample_poly(x, 1, 2)
    reconstructed = signal.resample_poly(safe, 2, 1)[:len(x)]
    recordings = {"reference": (x, FS), "noisy": (noisy, FS), "filtered": (filtered, FS),
                  "downsampled_naive": (naive, FS//2), "downsampled_antialias": (safe, FS//2),
                  "reconstructed": (reconstructed, FS)}
    gain = 0.95 / max(1, max(np.max(np.abs(v[0])) for v in recordings.values()))
    for name, (audio, fs) in recordings.items():
        wavfile.write(out / (name + ".wav"), fs, (audio * gain).astype(np.float32))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for row, (name, audio) in enumerate((("Reference", x), ("Noisy", noisy), ("Filtered", filtered))):
        axes[row, 0].plot(t, audio, linewidth=0.4)
        axes[row, 0].set(title=name, xlabel="Time (s)", ylabel="Amplitude", ylim=(-max(abs(noisy)), max(abs(noisy))))
        f, db = spectrum(audio, FS)
        axes[row, 1].plot(f, db, linewidth=0.6)
        axes[row, 1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (dB re 1)", ylim=(-100, 0))
        f, tt, power = signal.spectrogram(audio, FS, window="hann", nperseg=512, noverlap=384, scaling="density")
        mesh = axes[row, 2].pcolormesh(tt, f, 10*np.log10(np.maximum(power, 1e-15)), shading="auto", vmin=-100, vmax=-30)
        axes[row, 2].set(xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(mesh, ax=list(axes[:, 2]), label="PSD (dB/Hz)")
    fig.suptitle(label + ": waveform, FFT amplitude, spectrogram")
    fig.subplots_adjust(hspace=0.4, wspace=0.35, right=0.86)
    fig.savefig(out / "signal_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for kind in ("lowpass", "bandstop"):
        i = max((i for i,d in enumerate(designs) if d[0] == kind), key=lambda i: rows[i]["output_snr_db"])
        f, h = signal.freqz(designs[i][3], worN=8192, fs=FS)
        ax.plot(f, 20*np.log10(np.maximum(abs(h), 1e-8)), label=str(designs[i][:3]))
    ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", ylim=(-100, 5), title="Best tested design in each filter family")
    ax.legend(); ax.grid()
    fig.tight_layout(); fig.savefig(out / "filter_response.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for name, audio in (("Naive speech", naive), ("Anti-aliased speech", safe)):
        f, db = spectrum(audio, FS//2)
        axes[0].plot(f, db, label=name, alpha=0.8)
    diagnostic = np.sin(2*np.pi*6000*t)
    for name, audio in (("Naive 6 kHz tone", diagnostic[::2]), ("Anti-aliased 6 kHz tone", signal.resample_poly(diagnostic, 1, 2))):
        f, db = spectrum(audio[256:-256], FS//2)
        axes[1].plot(f, db, label=name)
    for ax in axes:
        ax.set(xlabel="Frequency at 8 kHz sampling (Hz)", ylabel="Amplitude (dB re 1)", ylim=(-120, 5))
        ax.legend(); ax.grid()
    axes[1].set_title("Diagnostic: 6000 Hz folds to 2000 Hz without anti-aliasing")
    fig.tight_layout(); fig.savefig(out / "aliasing.png", dpi=160); plt.close(fig)
    config = dict(source="synthetic demo" if args.demo else str(args.input.resolve()), original_fs=int(original_fs),
                  processing_fs=FS, duration_s=len(x)/FS, noise_frequency_hz=2500,
                  requested_snr_db=args.snr_db, selected_filter=rows[best_index],
                  audio_export_gain=gain, excluded_samples_per_edge=256)
    (out / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(label, "complete:", out.resolve())
    print("Best tested design:", rows[best_index])


if __name__ == "__main__":
    main()
