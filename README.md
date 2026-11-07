# Speech interference removal using Python

Research question: How well can a narrow FIR band-stop filter suppress a
tonal interferer while preserving speech, compared with a low-pass filter?

## Run

Install Python 3.11 or later, then run in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe speech_project.py --input data/speech.wav
```

See data/README.md for recording instructions. No speech data or measured
results are included yet. To check the pipeline with an explicitly synthetic
signal instead of speech:

```powershell
.\.venv\Scripts\python.exe speech_project.py --demo --output outputs/demo
```

## Experiment

The reference recording is converted to mono, centered, and resampled to
16 kHz with anti-aliasing. A 2500 Hz sinusoid is added at a controlled input
SNR (default 5 dB). This models tonal interference, not arbitrary room noise.
Compare Hamming-window FIR low-pass and band-stop filters with orders
128, 256, and 512, and vary their cutoff/bandwidth. All orders are even,
giving an integer group delay. A second sweep varies input SNR.

Metrics use the original recording as the reference. Filtering uses linear
convolution with group-delay compensation; boundary samples are excluded
from all filter comparisons using a shared mask. Output SNR includes both
remaining interference and filter-induced speech distortion. A separate
clean-reference filtering error measures distortion. Results cannot establish
performance against noise already present in the original recording.

Downsampling from 16 to 8 kHz is compared with and without an anti-aliasing
filter. A separate 6000 Hz test tone demonstrates folding to 2000 Hz.
Speech above 4 kHz may be weak; the diagnostic tone makes aliasing explicit.
Resampling back to 16 kHz illustrates interpolation, which cannot recover
discarded or aliased information.

Outputs: CSV measurements, JSON configuration, comparison figures, and WAV
audio at key stages. Audio exports share one gain so listening comparisons
do not independently normalize away level differences. Plots and metrics
use the unscaled processing arrays. The selected filter maximizes reference
SNR on this recording; this is an exploratory comparison, not independent
validation or a universally optimal filter.

## Presentation plan (13 slides)

1. Title, full name, student ID
2. Problem and research question
3. Recording and experiment setup
4. DSP pipeline and lecture connections
5. Original signal: waveform, FFT, spectrogram
6. Controlled interference and input SNR
7. FIR designs, frequency response, and delay
8. Filter parameter comparison
9. Before/after plots and listening examples
10. Noise-level sweep and speech-distortion tradeoff
11. Downsampling and aliasing diagnostic
12. Anti-aliasing and interpolation results
13. Conclusions, limitations, and sources

Use actual results only after running on your recording. Submit slides,
recording, source, dependencies, and outputs together. Verify your assigned
presentation date separately; it was not supplied in the project files.
