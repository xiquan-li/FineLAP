import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import random
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm


def extract_single_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    def select_random_label(labels_str):
        if pd.isna(labels_str):
            return labels_str
        labels_list = str(labels_str).split(',')
        return random.choice(labels_list).strip()
    df['labels'] = df['labels'].apply(select_random_label)
    return df


def generate_single_mixture(args: Tuple) -> Dict:
    df_fg, df_bg, mix_id, output_dir = args
    
    bg_idx = random.randint(0, len(df_bg) - 1)
    background_event = df_bg.iloc[bg_idx].to_dict()
    
    n_events = random.randint(1, 5)
    selected_indices = random.sample(range(len(df_fg)), n_events)
    selected_events = [df_fg.iloc[idx].to_dict() for idx in selected_indices]
    
    total_duration = 10.0
    placed_events = []
    
    for event in selected_events:
        audio_duration = event['audio_duration']
        k_repeats = random.randint(1, 3) if audio_duration < 3.0 else 1
        event_segments = []
        
        for repeat_idx in range(k_repeats):
            max_start_time = total_duration - audio_duration
            if max_start_time > 0:
                max_attempts = 100
                attempt = 0
                placed = False
                
                while not placed and attempt < max_attempts:
                    start_time = random.uniform(0, max_start_time)
                    end_time = start_time + audio_duration
                    
                    overlaps = False
                    for existing_start, existing_end in event_segments:
                        if (start_time < existing_end and end_time > existing_start):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        new_event = event.copy()
                        new_event['start_time'] = start_time
                        new_event['end_time'] = end_time
                        new_event['snr_db'] = random.uniform(12, 20)
                        placed_events.append(new_event)
                        event_segments.append((start_time, end_time))
                        placed = True
                    attempt += 1
    
    # use_background = random.random() < 0.7
    use_background = True
    
    bg_label = ''
    if use_background:
        bg_label = background_event.get('labels', '')
        if pd.notna(bg_label):
            bg_label_str = str(bg_label)
            if ',' in bg_label_str:
                bg_label = random.choice(bg_label_str.split(',')).strip()
            else:
                bg_label = bg_label_str.strip()
    
    label_groups = {}
    for event in placed_events:
        label = event['labels']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(event)
    
    phrases = []
    # if use_background and bg_label:
    #     phrases.append({
    #         "phrase": bg_label,
    #         "segments": [[0.0, 10.0]],
    #         "original_clip_id": background_event.get('fname', ''),
    #     })
    
    for label, events in label_groups.items():
        segments = [[event['start_time'], event['end_time']] for event in events]
        phrases.append({
            "phrase": label,
            "segments": segments,
            "original_clip_id": events[0]['fname'],
        })
    
    audio_path = os.path.join(output_dir, "audio", f"{mix_id}.wav")
    timeline_path = os.path.join(output_dir, "timeline", f"{mix_id}_timeline.png")
    
    sample_rate = 22050
    total_samples = int(10.0 * sample_rate)
    
    try:
        if use_background:
            bg_audio, sr = librosa.load(background_event['audio_path'], sr=sample_rate)
            if len(bg_audio) > total_samples:
                bg_audio = bg_audio[:total_samples]
            elif len(bg_audio) < total_samples:
                bg_audio = np.pad(bg_audio, (0, total_samples - len(bg_audio)), mode='constant')
            bg_rms = np.sqrt(np.mean(bg_audio ** 2))

        else:
            bg_audio = np.random.normal(0, 0.01, total_samples).astype(np.float32)
            bg_rms = 0.01  # 让噪声不会过响

        mixed_audio = bg_audio.copy()
        
        for event in placed_events:
            try:
                fg_audio, sr = librosa.load(event['audio_path'], sr=sample_rate)
                fg_rms = np.sqrt(np.mean(fg_audio ** 2))
                
                if fg_rms > 0 and bg_rms > 0:
                    snr_db = event['snr_db']
                    snr_linear = 10 ** (snr_db / 20.0)
                    target_fg_rms = bg_rms * snr_linear
                    gain = target_fg_rms / fg_rms
                    scaled_audio = fg_audio * gain
                else:
                    scaled_audio = fg_audio
                
                start_sample = int(event['start_time'] * sample_rate)
                end_sample = int(event['end_time'] * sample_rate)
                audio_length = min(len(scaled_audio), end_sample - start_sample)
                end_sample = start_sample + audio_length
                mixed_audio[start_sample:end_sample] += scaled_audio[:audio_length]
            except Exception as e:
                continue
        
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val * 0.95
        
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        sf.write(audio_path, mixed_audio, sample_rate)
        
        try:
            phrases_list = phrases
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(-0.5, len(phrases_list) - 0.5)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Sound Events', fontsize=12)
            ax.set_title(f'Timeline Visualization - {mix_id}', fontsize=14)
            
            for i, phrase in enumerate(phrases_list):
                label = phrase['phrase']
                segments = phrase['segments']
                ax.text(-0.5, i, label, ha='right', va='center', fontsize=10)
                
                for segment in segments:
                    start_time, end_time = segment
                    width = end_time - start_time
                    rect = patches.Rectangle(
                        (start_time, i - 0.4), width, 0.8,
                        linewidth=1, edgecolor='darkblue', facecolor='lightblue', alpha=0.7
                    )
                    ax.add_patch(rect)
            
            ax.set_yticks(range(len(phrases_list)))
            ax.set_yticklabels([phrase['phrase'] for phrase in phrases_list])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            templates = [
                "A soundscape featuring {}.",
                "This audio contains the sounds of {}.",
                "Listen to the mix of {}.",
                "An audio recording with {}.",
                "The sound of {} can be heard.",
                "This clip includes {}.",
                "A mixture of sound events: {}.",
                "Audio featuring {}.",
                "Sounds present in this recording: {}.",
                "This audio sample contains {}."
            ]
            
            all_labels = [label.lower() for label in label_groups.keys()]
            labels_str = ", ".join(all_labels)
            template = random.choice(templates)
            caption = template.format(labels_str)
            
            metadata = {
                "audio_id": mix_id,
                "audio_path": audio_path,
                "timeline_path": timeline_path,
                "caption": caption,
                "phrases": phrases
            }
            return metadata
        except Exception as e:
            return None
    except Exception as e:
        return None


def simulate_sed_data(fg_csv_path: str, bg_csv_path: str, output_dir: str, num_mixtures: int = 10, num_processes: int = None):
    print(f"Starting generation of {num_mixtures} SED mixed audios...")
    
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "timeline"), exist_ok=True)
    
    df_fg = extract_single_events(fg_csv_path)
    df_bg = pd.read_csv(bg_csv_path)
    
    if num_processes is None:
        num_processes = min(cpu_count(), num_mixtures)
    
    mix_ids = [f"mix_{i+1:04d}" for i in range(num_mixtures)]
    args_list = [(df_fg, df_bg, mix_id, output_dir) for mix_id in mix_ids]

    print(f"Using {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(generate_single_mixture, args_list),
                total=len(args_list),
                desc="Generating mixtures"
            )
        )
    
    all_metadata = [metadata for metadata in results if metadata is not None]
 
    if all_metadata:
        metadata_path = os.path.join(output_dir, "sed_metadata.jsonl")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for metadata in all_metadata:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    print(f"\nCompleted! Generated {len(all_metadata)} mixed audios")
    if all_metadata:
        print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate SED dataset generation')
    parser.add_argument('--fg_csv_path', 
                       default="/inspire/hdd/project/embodied-multimodality/public/xqli/ALE/F-CLAP/data/FSD50K/dev_energy_filtered_1s-7.5s.csv",
                       help='Foreground audio CSV file path')
    parser.add_argument('--bg_csv_path', 
                       default="/inspire/hdd/project/embodied-multimodality/public/xqli/ALE/F-CLAP/data/FSD50K/dev_energy_filtered_10s-30s.csv",
                       help='Background audio CSV file path')
    parser.add_argument('--output_dir', 
                       default="/inspire/hdd/project/embodied-multimodality/public/xqli/ALE/F-CLAP/data_handling/simulate_sed_data/output",
                       help='Output directory')
    parser.add_argument('--num_mixtures', type=int, default=10,
                       help='Number of mixed audios to generate')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of processes to use (default: CPU count)')
    
    args = parser.parse_args()
    
    simulate_sed_data(args.fg_csv_path, args.bg_csv_path, args.output_dir, args.num_mixtures, args.num_processes)


'''
Usage example:
python data_handling/simulate_sed_data/simulate_sed_data.py \
    --fg_csv_path data/FSD50K/dev_energy_filtered_1s-7.5s.csv \
    --bg_csv_path ../../data/Adobe_Sound_Effect/Ambience_cut10s.csv \
    --output_dir ../../data/FSD50K/clips/dev_simulated_sed_data \
    --num_mixtures 100 \
    --num_processes 10
'''
