"""
Real-Time Audio Denoising
Captures microphone audio, denoises it live, and plays it back.
"""

import torch
import numpy as np
import threading
import queue
import time

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("Run: pip install sounddevice")

try:
    import torchaudio.transforms as T
except ImportError:
    raise ImportError("Run: pip install torchaudio")

from model_spectrogram import SpectrogramAutoencoder


# ── Config ─────────────────────────────────────────────────────
SR          = 16000
BLOCK_SIZE  = 4096
N_FFT       = 512
HOP         = 128
N_MELS      = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load Model ─────────────────────────────────────────────────
def load_model():

    model = SpectrogramAutoencoder()

    ckpt = "models/best_model.pt"

    try:
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)

        print(f"Loaded checkpoint: {ckpt}")

    except FileNotFoundError:

        print(f"No checkpoint found at {ckpt}")
        print("Train model first.")

    model.to(DEVICE)
    model.eval()

    return model


# ── Spectrogram Helpers ────────────────────────────────────────
mel_transform = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS
)

amp2db = T.AmplitudeToDB()


def wav_to_spec(wav_tensor):

    spec = amp2db(
        mel_transform(wav_tensor.unsqueeze(0))
    )

    mn = spec.min()
    mx = spec.max()

    spec = (spec - mn) / (mx - mn + 1e-8)

    return spec, mn, mx


def spec_to_wav_approx(spec_norm, noisy_wav):

    window = torch.hann_window(N_FFT)

    stft = torch.stft(
        noisy_wav,
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(stft)
    phase = torch.angle(stft)

    pred = spec_norm.squeeze()

    if pred.dim() > 1:
        pred = pred.mean(0)

    pred = pred[:magnitude.shape[1]]

    pred = pred.unsqueeze(0).expand(
        magnitude.shape[0],
        pred.shape[0]
    )

    clean_mag = magnitude[:, :pred.shape[1]] * pred

    real = clean_mag * torch.cos(
        phase[:, :pred.shape[1]]
    )

    imag = clean_mag * torch.sin(
        phase[:, :pred.shape[1]]
    )

    clean_stft = torch.complex(real, imag)

    audio = torch.istft(
        clean_stft,
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        length=noisy_wav.shape[-1]
    )

    return audio


# ── Queues ─────────────────────────────────────────────────────
audio_queue = queue.Queue()
output_queue = queue.Queue()

running = True


# ── Audio Callback ─────────────────────────────────────────────
def audio_callback(indata, outdata, frames, time_info, status):

    audio_queue.put(indata.copy())

    try:
        out = output_queue.get_nowait()
        outdata[:] = out

    except queue.Empty:

        outdata[:] = indata


# ── Denoising Thread ───────────────────────────────────────────
def denoise_loop(model):

    global running

    print("\nReal-time denoising active.")
    print("Press Ctrl+C to stop.\n")

    while running:

        try:
            block = audio_queue.get(timeout=1.0)

        except queue.Empty:
            continue

        wav = torch.tensor(
            block[:, 0],
            dtype=torch.float32
        )

        wav = wav / (wav.abs().max() + 1e-8)

        with torch.no_grad():

            spec, mn, mx = wav_to_spec(wav)

            inp = spec.unsqueeze(0).to(DEVICE)

            out_spec = model(inp).squeeze().cpu()

            denoised = spec_to_wav_approx(
                out_spec,
                wav
            )

        denoised = denoised[:len(block)]

        out_block = np.zeros_like(block)

        out_block[:len(denoised), 0] = denoised.numpy()

        output_queue.put(out_block)


# ── Main ───────────────────────────────────────────────────────
def run_realtime():

    global running

    model = load_model()

    print(f"Device      : {DEVICE}")
    print(f"Sample Rate : {SR}")
    print(f"Block Size  : {BLOCK_SIZE}")

    worker = threading.Thread(
        target=denoise_loop,
        args=(model,),
        daemon=True
    )

    worker.start()

    try:

        with sd.Stream(
            samplerate=SR,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            channels=1,
            callback=audio_callback,
        ):

            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:

        running = False

        print("\nStopped realtime denoising.")


if __name__ == "__main__":
    run_realtime()
