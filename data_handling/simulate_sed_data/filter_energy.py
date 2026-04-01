import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools


def compute_energy_envelope(audio_path, frame_length=1024, hop_length=512):
    """Compute energy envelope of audio"""
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    energy_envelope = []
    times = []
    
    for i in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[i:i + frame_length]
        energy = np.sum(frame ** 2)
        energy_envelope.append(energy)
        times.append(i / sr)
    
    return np.array(energy_envelope), np.array(times)


def detect_onset_offset(energy_envelope, times, noise_floor_db=-50.0, method='absolute', percentile_threshold=10.0, window_size=10, window_step=1):
    """
    Detect onset and offset using sliding window on log energy envelope
    
    Args:
        energy_envelope: Energy envelope
        times: Time axis
        noise_floor_db: Noise floor threshold (dB)
        method: Detection method ('absolute', 'percentile')
        percentile_threshold: Percentile threshold for percentile method
        window_size: Sliding window size (frames)
        window_step: Sliding window step (frames)
    
    Returns:
        onset_time: Onset time
        offset_time: Offset time
    """
    # Convert to log energy (dB)
    log_energy = 10 * np.log10(energy_envelope + 1e-12)
    
    # Determine noise floor based on method
    if method == 'absolute':
        threshold = noise_floor_db
    elif method == 'percentile':
        threshold = np.percentile(log_energy, percentile_threshold)
    else:
        raise ValueError("Method must be 'absolute' or 'percentile'")
    
    n_frames = len(log_energy)
    
    if n_frames == 0:
        return None, None
    
    window_size = min(window_size, n_frames)
    
    onset_idx = None
    offset_idx = None
    
    # Detect onset: find first window with energy above threshold
    for start_idx in range(0, n_frames - window_size + 1, window_step):
        end_idx = start_idx + window_size
        window_energy = log_energy[start_idx:end_idx]
        
        if np.any(window_energy > threshold):
            onset_idx = start_idx
            break
    
    if onset_idx is None:
        return None, None
    
    # Detect offset: find first window after onset where all energy is below threshold
    for start_idx in range(onset_idx + window_size, n_frames - window_size + 1, window_step):
        end_idx = start_idx + window_size
        window_energy = log_energy[start_idx:end_idx]
        
        if np.all(window_energy <= threshold):
            offset_idx = start_idx
            break
    
    # If no offset found, set to last frame
    if offset_idx is None:
        offset_idx = n_frames - 1
    
    onset_time = times[onset_idx] if onset_idx is not None else None
    offset_time = times[offset_idx] if offset_idx is not None else None
    
    return onset_time, offset_time


def process_single_audio(args):
    """Process single audio file"""
    row_data, energy_images_dir, trimmed_audio_dir, noise_floor_db, method, percentile_threshold, window_size, window_step = args
    audio_path = Path(row_data['audio_path'])
    
    try:
        if not audio_path.exists():
            return {
                'idx': row_data['idx'],
                'success': False,
                'error': f"Audio file not found: {audio_path}",
                'onset_time': 0.0,
                'offset_time': 0.0,
                'energy_image_path': '',
                'trimmed_audio_path': ''
            }
        
        # Compute energy envelope
        energy_envelope, times = compute_energy_envelope(audio_path)
        
        # Detect onset and offset
        onset_time, offset_time = detect_onset_offset(energy_envelope, times, noise_floor_db, method, percentile_threshold, window_size, window_step)
        
        # Generate visualization
        image_filename = f"{audio_path.stem}_energy.png"
        image_path = energy_images_dir / image_filename
        plot_energy_envelope(energy_envelope, times, onset_time, offset_time, image_path)
        
        # Trim audio and save
        trimmed_audio_path = ''
        if onset_time is not None and offset_time is not None:
            trimmed_filename = audio_path.name
            trimmed_output_path = trimmed_audio_dir / trimmed_filename
            if trim_audio(audio_path, onset_time, offset_time, trimmed_output_path):
                trimmed_audio_path = str(trimmed_output_path)
        
        return {
            'idx': row_data['idx'],
            'success': True,
            'error': None,
            'onset_time': onset_time,
            'offset_time': offset_time,
            'energy_image_path': str(image_path),
            'trimmed_audio_path': trimmed_audio_path
        }
        
    except Exception as e:
        return {
            'idx': row_data['idx'],
            'success': False,
            'error': f"Error processing {audio_path}: {e}",
            'onset_time': 0.0,
            'offset_time': 0.0,
            'energy_image_path': '',
            'trimmed_audio_path': ''
        }


def trim_audio(audio_path, onset_time, offset_time, output_path):
    """Trim audio based on onset and offset times"""
    try:
        if onset_time is None or offset_time is None:
            return False
        
        audio, sr = sf.read(audio_path)
        is_stereo = len(audio.shape) > 1
        
        # Convert time to sample indices
        onset_sample = int(onset_time * sr)
        offset_sample = int(offset_time * sr)
        
        # Ensure indices are valid
        onset_sample = max(0, min(onset_sample, len(audio)))
        offset_sample = max(onset_sample, min(offset_sample, len(audio)))
        
        if offset_sample <= onset_sample:
            return False
        
        # Trim audio
        if is_stereo:
            trimmed_audio = audio[onset_sample:offset_sample, :]
        else:
            trimmed_audio = audio[onset_sample:offset_sample]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, trimmed_audio, sr)
        
        return True
    except Exception as e:
        print(f"Error trimming audio {audio_path}: {e}")
        return False


def plot_energy_envelope(energy_envelope, times, onset_time, offset_time, output_path):
    """Plot log energy envelope with onset/offset markers"""
    plt.figure(figsize=(12, 6))
    
    log_energy = 10 * np.log10(energy_envelope + 1e-12)
    
    plt.plot(times, log_energy, 'b-', linewidth=1, label='Log Energy Envelope (dB)')
    
    if onset_time is not None:
        plt.axvline(x=onset_time, color='r', linestyle='--', linewidth=2, label=f'Onset: {onset_time:.3f}s')
    if offset_time is not None:
        plt.axvline(x=offset_time, color='g', linestyle='--', linewidth=2, label=f'Offset: {offset_time:.3f}s')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.title('Log Energy Envelope with Onset/Offset Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_audio_events(csv_path, output_dir, max_data_nums=None, noise_floor_db=-50.0, method='absolute', percentile_threshold=10.0, num_processes=None, window_size=5, window_step=1):
    """Process audio event detection from CSV file (multiprocessing)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    energy_images_dir = output_path / "energy_images"
    energy_images_dir.mkdir(exist_ok=True)
    trimmed_audio_dir = output_path / "trimmed_audio"
    trimmed_audio_dir.mkdir(exist_ok=True)
    
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Limit data size (random sampling)
    if max_data_nums != -1 and max_data_nums < len(df):
        df = df.sample(n=max_data_nums, random_state=42).reset_index(drop=True)
        print(f"Randomly selected {max_data_nums} files for processing")
    
    print(f"Processing {len(df)} audio files...")
    
    # Add new columns
    df['onset_time'] = 0.0
    df['offset_time'] = 0.0
    df['energy_image_path'] = ''
    df['trimmed_audio_path'] = ''
    
    if num_processes is None:
        num_processes = min(cpu_count(), len(df))
    
    print(f"Using {num_processes} processes")
    
    # Prepare arguments for multiprocessing
    process_args = []
    for idx, row in df.iterrows():
        row_data = row.to_dict()
        row_data['idx'] = idx
        process_args.append((row_data, energy_images_dir, trimmed_audio_dir, noise_floor_db, method, percentile_threshold, window_size, window_step))
    
    processed_count = 0
    error_count = 0
    
    with Pool(processes=num_processes) as pool:
        results = []
        with tqdm(total=len(process_args), desc="Processing audio files") as pbar:
            for result in pool.imap(process_single_audio, process_args):
                results.append(result)
                pbar.update(1)
    
    # Process results
    for result in results:
        idx = result['idx']
        df.at[idx, 'onset_time'] = result['onset_time']
        df.at[idx, 'offset_time'] = result['offset_time']
        df.at[idx, 'energy_image_path'] = result['energy_image_path']
        df.at[idx, 'trimmed_audio_path'] = result['trimmed_audio_path']
        
        if result['success']:
            processed_count += 1
        else:
            error_count += 1
            print(result['error'])
    
    # Save updated CSV
    output_csv_path = output_path / "collection_with_events.csv"
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Files processed successfully: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Energy images directory: {energy_images_dir}")
    print(f"Trimmed audio directory: {trimmed_audio_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect audio events using sound energy envelope (multiprocessing)"
    )
    parser.add_argument(
        '--csv_path',
        required=True,
        help='Path to CSV file containing audio metadata'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--max_data_nums',
        type=int,
        default=None,
        help='Max data number to process (default: None)'
    )
    parser.add_argument(
        '--noise_floor_db',
        type=float,
        default=-50.0,
        help='Noise floor threshold in dB (default: -50.0)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='absolute',
        choices=['absolute', 'percentile'],
        help='Detection method: absolute threshold or percentile (default: absolute)'
    )
    parser.add_argument(
        '--percentile_threshold',
        type=float,
        default=10.0,
        help='Percentile threshold for noise floor detection (default: 10.0)'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=None,
        help='Number of processes to use (default: CPU count)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=5,
        help='Sliding window size in frames (default: 5)'
    )
    parser.add_argument(
        '--window_step',
        type=int,
        default=1,
        help='Sliding window step size in frames (default: 1)'
    )
    
    args = parser.parse_args()
    
    process_audio_events(args.csv_path, args.output_dir, args.max_data_nums, args.noise_floor_db, args.method, args.percentile_threshold, args.num_processes, args.window_size, args.window_step)


'''
Usage examples:

# Absolute threshold method (-20dB)
python data_handling/simulate_sed_data/filter_energy.py \
    --csv_path data/FSD50K/dev_meta.csv   \
    --output_dir ../../data/FSD50K/clips/dev_energy_cropped_-20db \
    --max_data_nums 10 \
    --noise_floor_db -20.0 \
    --method absolute \
    --window_size 20 \
    --num_processes 16

# Percentile method (5th percentile as noise floor)
python data_handling/simulate_sed_data/filter_energy.py \
    --csv_path data/FSD50K/collection_merged.csv \
    --output_dir data/FSD50K/energy_events \
    --max_data_nums 100 \
    --method percentile \
    --percentile_threshold 5.0 \
    --num_processes 8

# Percentile method (20th percentile as noise floor)
python data_handling/simulate_sed_data/filter_energy.py \
    --csv_path data/FSD50K/collection_merged_shorter_2s.csv \
    --output_dir data/FSD50K/energy_events_shorter_2s_percentile_10 \
    --max_data_nums 100 \
    --method percentile \
    --percentile_threshold 10.0 \
    --num_processes 8
'''

if __name__ == "__main__":
    main()
