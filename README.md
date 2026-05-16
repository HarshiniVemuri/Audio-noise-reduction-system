# 🎧 Audio Noise Reduction System

Clean noisy audio signals using a **Spectrogram-based CNN Autoencoder** (U-Net style).

---

## Project Structure

```
audio_noise_reduction/
├── train.py          # Model definition + full training pipeline
├── denoise.py        # Denoise a single audio file with a trained model
├── demo.py           # Quick demo — NO dataset download required
├── requirements.txt
├── data/             # Put LibriSpeech here (optional)
├── models/           # Saved checkpoints
└── outputs/          # Training curves + denoised audio
```

---

##  Quick Start 

### 1 — Install Python & dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Run the demo

```bash
python demo.py
```

This trains on **synthetic sine-wave audio** for 10 epochs and saves a
spectrogram comparison to `outputs/demo_result.png`.

---

## 🗂️ Full Training on LibriSpeech

### Download the dataset

```bash
# Option A — torchaudio (auto-download, ~6 GB)
python - <<'EOF'
import torchaudio
torchaudio.datasets.LIBRISPEECH("data/librispeech", url="train-clean-100", download=True)
EOF

# Option B — manual download from https://www.openslr.org/12
# Extract into data/librispeech/
```

### Train

```bash
python train.py \
  --data_dir data/librispeech \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.001
```

Training progress and loss curves are saved to `outputs/`.
The best checkpoint is saved to `models/best_model.pt`.

---

## 🔊 Denoise Your Own Audio

```bash
python denoise.py \
  --input  path/to/noisy.wav \
  --output outputs/clean.wav \
  --checkpoint models/best_model.pt
```

Supported input formats: `.wav`, `.flac`

---

## 🧠 Model Architecture

```
Input noisy spectrogram  (1 × 64 × T)
         │
    Encoder (4 × ConvBlock, stride=2)
    32 → 64 → 128 → 256 channels
         │
    Bottleneck (2 × Conv2d + GELU)
         │
    Decoder (4 × TransposeConvBlock + skip connections)
    256 → 128 → 64 → 32 → 16 channels
         │
    Output Conv (1 × 1) + Sigmoid
         │
 Denoised spectrogram (1 × 64 × T)
```

- **Loss**: MSE between clean and denoised log-mel spectrograms
- **Optimizer**: AdamW + Cosine Annealing LR
- **Metric**: SNR Improvement (dB)

---

## 📊 Expected Results

| Setting            | SNR Improvement |
|--------------------|----------------|
| Demo (synthetic)   | +2 – +5 dB     |
| LibriSpeech (30ep) | +5 – +10 dB    |

---

## 🖥️ Hardware

| Hardware | Speed estimate |
|----------|---------------|
| CPU only | ~5 min/epoch (LibriSpeech-100) |
| GPU (RTX 3060+) | ~30 sec/epoch |

GPU is **auto-detected** — no code changes needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No audio files found` | Check `--data_dir` points to LibriSpeech folder |
| Out of memory | Reduce `--batch_size` to 8 or 4 |
| Slow on CPU | Use `--epochs 5` for a quick test |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
