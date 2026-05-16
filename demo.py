"""
Audio Noise Reduction — Demo with Audio Playback
================================================
Steps this script performs:
  1. Downloads LibriSpeech test-clean if not present
  2. Trains the Spectrogram Autoencoder
  3. Takes a real speech sample
  4. Saves & plays NOISY audio
  5. Denoises it
  6. Saves & plays CLEAN (denoised) audio
  7. Shows spectrogram comparison image
  8. Prints SNR improvement score

Run:
    python demo.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader

from model_spectrogram import SpectrogramAutoencoder
from evaluate import snr_improvement
from train import NoisyAudioDataset, download_librispeech, CFG


# ── Playback ───────────────────────────────────────────────────
try:
    import sounddevice as sd
    PLAYBACK_OK = True
except ImportError:
    PLAYBACK_OK = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ── Spectrogram Helpers ────────────────────────────────────────
def wav_to_spec(wav, cfg=CFG):

    mel = T.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
    )

    db = T.AmplitudeToDB()

    spec = db(
        mel(wav.unsqueeze(0))
    )

    mn = spec.min()
    mx = spec.max()

    spec = (spec - mn) / (mx - mn + 1e-8)

    return spec


# ── Audio Playback ─────────────────────────────────────────────
def play_audio(wav_tensor, sr, label):

    wav_np = wav_tensor.cpu().numpy()

    out_path = f"outputs/{label}.wav"

    sf.write(out_path, wav_np, sr)

    print(f"  Saved → {out_path}")

    if PLAYBACK_OK:

        print(f"  Playing {label} audio...")

        sd.play(wav_np, samplerate=sr)
        sd.wait()

        print("  Playback done.")

    else:

        print("  Playback skipped.")
        print(f"  Open {out_path} manually.")


# ── Better Reconstruction ──────────────────────────────────────
def spec_to_wav(denoised_spec, noisy_wav, cfg=CFG):

    window = torch.hann_window(cfg["n_fft"])

    stft = torch.stft(
        noisy_wav,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        window=window,
        return_complex=True,
    )

    magnitude = torch.abs(stft)
    phase = torch.angle(stft)

    pred = denoised_spec.squeeze()

    # Average mel bins -> time mask
    if pred.dim() > 1:
        pred = pred.mean(0)

    # Match dimensions
    pred = pred[:magnitude.shape[1]]

    pred = pred.unsqueeze(0).expand(
        magnitude.shape[0],
        pred.shape[0]
    )

    # Stronger denoising
    pred = torch.clamp(pred, 0.15, 1.0)

    # Apply mask
    clean_mag = magnitude[:, :pred.shape[1]] * pred

    # Reconstruct STFT
    real = clean_mag * torch.cos(
        phase[:, :pred.shape[1]]
    )

    imag = clean_mag * torch.sin(
        phase[:, :pred.shape[1]]
    )

    clean_stft = torch.complex(real, imag)

    # Inverse STFT
    audio = torch.istft(
        clean_stft,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        window=window,
        length=noisy_wav.shape[-1],
    )

    # Boost volume
    audio = audio * 1.8

    return audio.clamp(-1, 1)


# ── Training ───────────────────────────────────────────────────
def train_model(files, cfg):

    np.random.shuffle(files)

    n_val = max(
        1,
        int(len(files) * cfg["val_split"])
    )

    train_files = [
        str(f) for f in files[n_val:]
    ]

    loader = DataLoader(
        NoisyAudioDataset(train_files, cfg),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0
    )

    model = SpectrogramAutoencoder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"]
    )

    loss_fn = nn.MSELoss()

    print(f"\n{'='*55}")
    print("  Training Spectrogram Autoencoder")
    print(f"  Files : {len(train_files)}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*55}")

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()

        epoch_loss = 0.0

        for noisy, clean in loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            pred = model(noisy)

            loss = loss_fn(pred, clean)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        print(
            f"  Epoch {epoch:2d}/{cfg['epochs']} "
            f"| Loss: {epoch_loss/len(loader):.5f}"
        )

    torch.save(
        model.state_dict(),
        cfg["checkpoint"]
    )

    print(f"\n  Model saved → {cfg['checkpoint']}")

    return model


# ── Spectrogram Plot ───────────────────────────────────────────
def plot_spectrograms(
    noisy_spec,
    denoised_spec,
    clean_spec,
    snr_val
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    items = [
        ("Noisy Input", noisy_spec),
        (
            f"Denoised Output\nSNR: {snr_val:+.2f} dB",
            denoised_spec
        ),
        ("Clean Target", clean_spec),
    ]

    for ax, (title, s) in zip(axes, items):

        ax.imshow(
            s.numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma"
        )

        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

    plt.tight_layout()

    out = "outputs/demo_result.png"

    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Spectrogram image saved → {out}")


# ── Main ───────────────────────────────────────────────────────
def main():

    print("=" * 55)
    print("  Audio Noise Reduction System")
    print("  Spectrogram CNN Autoencoder")
    print("=" * 55)

    # Dataset
    data_dir = Path(CFG["data_dir"])

    files = list(data_dir.rglob("*.flac"))
    files += list(data_dir.rglob("*.wav"))

    if len(files) == 0:

        download_librispeech(
            str(data_dir)
        )

        files = list(data_dir.rglob("*.flac"))
        files += list(data_dir.rglob("*.wav"))

    print(f"\nFound {len(files)} audio files.")

    # Train
    model = train_model(files, CFG)

    model.eval()

    # Demo sample
    demo_file = str(files[0])

    print(f"\nDemo file: {Path(demo_file).name}")

    wav_np, sr = sf.read(
        demo_file,
        dtype="float32"
    )

    wav = torch.tensor(wav_np)

    if wav.dim() == 2:
        wav = wav.mean(1)

    if sr != CFG["sample_rate"]:

        wav = torchaudio.functional.resample(
            wav,
            sr,
            CFG["sample_rate"]
        )

    # Crop to 4 sec
    seg = int(CFG["sample_rate"] * 4.0)

    wav = wav[:seg]

    # Normalize
    wav = wav / (wav.abs().max() + 1e-8)

    # Add stronger noise
    noisy_wav = (
        wav
        + CFG["noise_factor"]
        * torch.randn_like(wav)
    )

    noisy_wav = noisy_wav.clamp(-1, 1)

    # Play noisy
    print("\n--- NOISY AUDIO ---")

    play_audio(
        noisy_wav,
        CFG["sample_rate"],
        "noisy_speech"
    )

    time.sleep(1)

    # Spectrograms
    clean_spec = wav_to_spec(wav)

    noisy_spec = wav_to_spec(noisy_wav)

    # Denoise
    with torch.no_grad():

        pred_spec = model(
            noisy_spec.unsqueeze(0).to(DEVICE)
        ).squeeze().cpu()

    denoised_wav = spec_to_wav(
        pred_spec,
        noisy_wav
    )

    # Play denoised
    print("\n--- DENOISED AUDIO ---")

    play_audio(
        denoised_wav,
        CFG["sample_rate"],
        "denoised_speech"
    )

    # Save clean
    sf.write(
        "outputs/original_clean_speech.wav",
        wav.cpu().numpy(),
        CFG["sample_rate"]
    )

    print(
        "\n  Original clean saved → "
        "outputs/original_clean_speech.wav"
    )

    # SNR
    snr_val = snr_improvement(
        clean_spec.unsqueeze(0),
        pred_spec.unsqueeze(0),
        noisy_spec.unsqueeze(0),
    )

    print(f"\nSNR Improvement: {snr_val:+.2f} dB")

    # Plot
    print("\n--- SPECTROGRAM COMPARISON ---")

    plot_spectrograms(
        noisy_spec.squeeze().cpu(),
        pred_spec.squeeze().cpu(),
        clean_spec.squeeze().cpu(),
        snr_val,
    )

    print(f"\n{'='*55}")
    print("  DEMO COMPLETE")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()