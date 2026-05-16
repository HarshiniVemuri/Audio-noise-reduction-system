"""
Audio Noise Reduction — Full Training Pipeline
Dataset : LibriSpeech test-clean (auto-downloaded)
Model   : Spectrogram-based CNN Autoencoder
Metric  : SNR Improvement

Run: python train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

from model_spectrogram import SpectrogramAutoencoder
from evaluate import snr_improvement

# ── Config ──────────────────────────────────────────────────────────────────
CFG = {
    "sample_rate" : 16000,
    "n_fft"       : 512,
    "hop_length"  : 128,
    "n_mels"      : 128,
    "segment_len" : 2.0,
    "noise_factor": 0.15,
    "batch_size"  : 8,
    "epochs"      : 50,
    "lr"          : 1e-3,
    "val_split"   : 0.1,
    "data_dir"    : "data/librispeech",
    "checkpoint"  : "models/best_model.pt",
    "results_dir" : "outputs",
}


# ── Dataset ──────────────────────────────────────────────────────────────────
class NoisyAudioDataset(Dataset):
    def __init__(self, file_list, cfg):
        self.files       = file_list
        self.cfg         = cfg
        self.seg_samples = int(cfg["sample_rate"] * cfg["segment_len"])
        self.mel         = T.MelSpectrogram(
            sample_rate = cfg["sample_rate"],
            n_fft       = cfg["n_fft"],
            hop_length  = cfg["hop_length"],
            n_mels      = cfg["n_mels"],
        )
        self.amp2db = T.AmplitudeToDB()

    def _load(self, path):
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        wav = torch.tensor(data)
        if wav.dim() == 2:
            wav = wav.mean(1)  # stereo to mono
        if sr != self.cfg["sample_rate"]:
            wav = torchaudio.functional.resample(wav, sr, self.cfg["sample_rate"])
        if wav.shape[0] < self.seg_samples:
            wav = torch.nn.functional.pad(wav, (0, self.seg_samples - wav.shape[0]))
        else:
            start = torch.randint(0, wav.shape[0] - self.seg_samples + 1, (1,)).item()
            wav   = wav[start: start + self.seg_samples]
        return wav

    def _to_spec(self, wav):
        s = self.amp2db(self.mel(wav.unsqueeze(0)))
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav       = self._load(self.files[idx])
        noisy_wav = wav + self.cfg["noise_factor"] * torch.randn_like(wav)
        return self._to_spec(noisy_wav), self._to_spec(wav)


# ── Download LibriSpeech ─────────────────────────────────────────────────────
def download_librispeech(data_dir):
    print("Downloading LibriSpeech test-clean (~346 MB)...")
    print("This will take 2-5 minutes depending on your internet.\n")
    torchaudio.datasets.LIBRISPEECH(data_dir, url="test-clean", download=True)
    print("Download complete!\n")


# ── Training loop ────────────────────────────────────────────────────────────
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Auto-download if not present
    data_dir = Path(cfg["data_dir"])
    files    = list(data_dir.rglob("*.flac")) + list(data_dir.rglob("*.wav"))
    if len(files) == 0:
        download_librispeech(str(data_dir))
        files = list(data_dir.rglob("*.flac")) + list(data_dir.rglob("*.wav"))

    print(f"Found {len(files)} audio files.")

    # Split
    np.random.shuffle(files)
    n_val        = max(1, int(len(files) * cfg["val_split"]))
    train_files  = [str(f) for f in files[n_val:]]
    val_files    = [str(f) for f in files[:n_val]]

    train_loader = DataLoader(NoisyAudioDataset(train_files, cfg),
                              batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(NoisyAudioDataset(val_files, cfg),
                              batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model     = SpectrogramAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.MSELoss()

    history  = {"train_loss": [], "val_loss": [], "snr_improvement": []}
    best_val = float("inf")

    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["checkpoint"]), exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Training Spectrogram Autoencoder")
    print(f"  Epochs: {cfg['epochs']} | Batch: {cfg['batch_size']}")
    print(f"{'='*55}")

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss, snr_scores = 0.0, []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred      = model(noisy)
                val_loss += criterion(pred, clean).item()
                snr_scores.append(snr_improvement(clean, pred, noisy))
        val_loss /= len(val_loader)
        mean_snr  = float(np.mean(snr_scores))

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["snr_improvement"].append(mean_snr)

        print(f"  Epoch {epoch:2d}/{cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"SNR: {mean_snr:+.2f} dB")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg["checkpoint"])
            print(f"    ✓ Best model saved")

    # Save curves
    _plot_history(history, cfg["results_dir"])
    with open(os.path.join(cfg["results_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved → {cfg['checkpoint']}")
    print(f"Curves saved    → {cfg['results_dir']}/training_curves.png\n")


def _plot_history(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history["snr_improvement"], color="green")
    ax2.set_title("SNR Improvement (dB)"); ax2.set_xlabel("Epoch")
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--data_dir",   type=str,   default=CFG["data_dir"])
    args = parser.parse_args()
    CFG.update(vars(args))
    train(CFG)
