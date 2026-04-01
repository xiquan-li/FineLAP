import torch
import torchaudio
from torchaudio.transforms import Resample
import argparse
import random
from pathlib import Path


def load_audio(audio_path, target_sr=16000):
    """Load audio and resample to target sample rate"""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0), target_sr


def insert_foreground(foreground, background, snr_db, sample_rate, insert_position=None):
    """
    Insert foreground into background at specified position with SNR control
    
    Args:
        foreground: foreground audio (1D tensor)
        background: background audio (1D tensor) - length will be preserved
        snr_db: Signal-to-Noise Ratio in dB
        sample_rate: sample rate for time calculation
        insert_position: insertion position in seconds (None for random)
    
    Returns:
        mixed: mixed audio with same length as background
    """
    bg_len = len(background)
    fg_len = len(foreground)
    
    # Crop foreground if longer than background
    if fg_len > bg_len:
        foreground = foreground[:bg_len]
        fg_len = bg_len
    
    # Calculate insertion position
    if insert_position is None:
        # Random position
        max_start = bg_len - fg_len
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
    else:
        # Specified position in seconds
        start_idx = int(insert_position * sample_rate)
        start_idx = max(0, min(start_idx, bg_len - fg_len))
    
    # Calculate SNR gain
    signal_power = torch.mean(foreground ** 2)
    noise_power = torch.mean(background[start_idx:start_idx+fg_len] ** 2)
    
    if signal_power > 0 and noise_power > 0:
        snr_linear = 10 ** (snr_db / 20.0)
        target_signal_power = noise_power * (snr_linear ** 2)
        fg_gain = torch.sqrt(target_signal_power / signal_power)
        foreground = foreground * fg_gain
    
    # Insert foreground into background
    mixed = background.clone()
    mixed[start_idx:start_idx+fg_len] += foreground
    
    # Normalize if needed
    max_val = torch.max(torch.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


def main():
    parser = argparse.ArgumentParser(description="Insert foreground audio into background with specified SNR")
    parser.add_argument("--foreground", type=str, required=True, help="Path to foreground audio file")
    parser.add_argument("--background", type=str, required=True, help="Path to background audio file")
    parser.add_argument("--snr", type=float, required=True, help="Signal-to-Noise Ratio in dB")
    parser.add_argument("--output", type=str, required=True, help="Output path for mixed audio")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--insert_position", type=float, default=None, help="Insertion position in seconds (None for random)")
    
    args = parser.parse_args()
    
    foreground, _ = load_audio(args.foreground, args.sample_rate)
    background, _ = load_audio(args.background, args.sample_rate)
    
    mixed = insert_foreground(foreground, background, args.snr, args.sample_rate, args.insert_position)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), mixed.unsqueeze(0), args.sample_rate)
    
    print(f"Mixed audio saved to: {output_path}")


if __name__ == "__main__":
    main()


"""
python data_handling/simulate_sed_data/simple_mix_audio.py \
    --foreground ../../data/FSD50K/clips/dev_energy_cropped_-20db/trimmed_audio/16404.wav \
    --background ../../../datasets/AudioCaps/test/YtjCNwdOUiGc.wav \
    --snr 10 \
    --output output/audio_mix_test.wav \
    --sample_rate 32000 \
    --insert_position 3
"""
