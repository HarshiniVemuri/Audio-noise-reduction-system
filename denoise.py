"""
Denoise a single audio file using the trained autoencoder.
Usage:
    python denoise.py --input noisy.wav --output clean.wav
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import argparse
import os
from pathlib import Path

from train import SpectrogramAutoencoder, CFG, snr_improvement


def load_model(checkpoint_path, device):
    model = SpectrogramAutoencoder().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def spec_to_audio(spec_db, noisy_wav, cfg):
    """
    Reconstruct audio from a denoised magnitude spectrogram using
    the noisy audio's phase (Griffin-Lim-style phase transfer).
    """
    # Convert db spec back to linear magnitude (approximate)
    spec_lin = 10 ** (spec_db / 20.0)

    # Get phase from noisy waveform via STFT
    stft = torch.stft(
        noisy_wav,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        return_complex=True,
    )
    phase = stft / (stft.abs() + 1e-8)

    # Apply denoised magnitude to noisy phase
    # (spec_lin is mel; we use it as a mask proxy on the original STFT)
    mask = torch.clamp(spec_lin, 0, 1)  # treat as Wiener-like mask
    denoised_stft = stft * mask.mean(dim=0, keepdim=True)[:stft.shape[0], :stft.shape[1]]

    audio = torch.istft(
        denoised_stft,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        length=noisy_wav.shape[-1],
    )
    return audio


def denoise_file(input_path, output_path, checkpoint, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(checkpoint, device)

    # Load audio
    wav, sr = torchaudio.load(input_path)
    if sr != cfg["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, cfg["sample_rate"])
    wav = wav.mean(0)  # mono

    mel_transform = T.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
    )
    amp2db = T.AmplitudeToDB()

    # Build spectrogram
    spec = mel_transform(wav.unsqueeze(0))
    spec_db = amp2db(spec)
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
    spec_norm = spec_norm.unsqueeze(0).to(device)  # (1, 1, mels, T)

    with torch.no_grad():
        denoised_spec = model(spec_norm).squeeze().cpu()

    # Reconstruct audio
    denoised_audio = spec_to_audio(denoised_spec, wav, cfg)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torchaudio.save(output_path, denoised_audio.unsqueeze(0), cfg["sample_rate"])
    print(f"Saved denoised audio: {output_path}")

    # SNR improvement (use spec-domain proxy)
    snr_val = snr_improvement(spec_norm.squeeze().cpu(), denoised_spec, spec_norm.squeeze().cpu())
    print(f"Spec-domain SNR improvement: {snr_val:+.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise an audio file")
    parser.add_argument("--input", required=True, help="Path to noisy input .wav")
    parser.add_argument("--output", default="outputs/denoised.wav", help="Path for denoised output")
    parser.add_argument("--checkpoint", default=CFG["checkpoint"])
    args = parser.parse_args()
    denoise_file(args.input, args.output, args.checkpoint, CFG)
