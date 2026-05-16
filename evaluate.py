"""
Evaluation utilities — SNR Improvement metric.
"""

import torch
import numpy as np


def snr(signal, reference):
    """Signal-to-Noise Ratio in dB."""
    noise = signal - reference
    return 10 * torch.log10(
        (reference ** 2).sum() / ((noise ** 2).sum() + 1e-8)
    )


def snr_improvement(clean, denoised, noisy):
    """
    SNR Improvement = SNR_after - SNR_before (dB).
    Higher is better. Positive means denoising helped.
    """
    snr_before = snr(noisy, clean)
    snr_after  = snr(denoised, clean)
    return (snr_after - snr_before).item()


def evaluate_batch(clean_batch, denoised_batch, noisy_batch):
    """Returns mean SNR improvement over a batch."""
    scores = []
    for c, d, n in zip(clean_batch, denoised_batch, noisy_batch):
        scores.append(snr_improvement(c, d, n))
    return float(np.mean(scores))
