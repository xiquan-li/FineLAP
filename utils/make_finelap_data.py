## Make jsonl data for FineLAP training

import json
import os
import pandas as pd
from collections import defaultdict
from audioset_sl_labels import mids_to_labels

DATASET_NAME = "DESED_Strong_fromAS"

if DATASET_NAME == "AudioSet_SL":
    AUDIO_DIR = "/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/audio/AudioSet_SL_flac"
    META_FILE_SEGMENT = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/audioset_train_strong.tsv'
    META_FILE_CAPTION = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/caption_tsv/AudioSet_SL.tsv'
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioSet_SL/finelap_metadata.jsonl'
elif DATASET_NAME == "DESED_Strong":
    AUDIO_DIR = "/apdcephfs_zwfy2/share_303944931/xiquanli/data/DESED/strong_label_real"
    META_FILE_SEGMENT = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/DESED/desed_strong_label.tsv'
    META_FILE_CAPTION = ''
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/DESED/Meta_FineLAP/desed_strong_wocaption.jsonl'
elif DATASET_NAME == "DESED_Strong_fromAS": 
    AUDIO_DIR = "/apdcephfs_zwfy2/share_303944931/xiquanli/data/DESED/DESED_from_StrongAS/data"
    META_FILE_SEGMENT = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/DESED/desed_in_audioset_sl.tsv'
    META_FILE_CAPTION = ''
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/DESED/Meta_FineLAP/desed_in_audioset_sl_wocaption.jsonl'

if __name__ == "__main__":
    print("Loading strong labels...")
    strong_df = pd.read_csv(META_FILE_SEGMENT, sep='\t')
    
    print("Loading captions...")
    if META_FILE_CAPTION != '': 
        caption_df = pd.read_csv(META_FILE_CAPTION, sep='\t')
        caption_dict = dict(zip(caption_df['id'], caption_df['caption']))
    else:
        caption_dict = {}
    
    # Group strong labels by segment_id and merge same phrases
    print("Processing segments...")
    segment_data = defaultdict(lambda: defaultdict(list))

    if DATASET_NAME == "AudioSet_SL":
        for _, row in strong_df.iterrows():
            segment_id = row['segment_id']
            phrase = mids_to_labels[row['label']]
            segment_data[segment_id][phrase].append([row['start_time_seconds'], row['end_time_seconds']]) # {segment: {phrase: [[start, end], [start, end], ...]}}
        
    elif DATASET_NAME == "DESED_Strong":
        for _, row in strong_df.iterrows():
            segment_id = row['filename'][:-4]  # without .wav suffix
            phrase = row['event_label']
            segment_data[segment_id][phrase].append([row['onset'], row['offset']]) # {segment: {phrase: [[start, end], [start, end], ...]}}
        
    elif DATASET_NAME == "DESED_Strong_fromAS":
        for _, row in strong_df.iterrows():
            split = row['filename'].split('/')[0]
            if split == 'eval': 
                print(f'Skipping row {row["filename"]} because it is in eval split')
                continue
            segment_id = row['filename'].split('/')[-1][:-4]  # without .wav suffix
            phrase = row['event_label']
            segment_data[segment_id][phrase].append([row['onset'], row['offset']]) # {segment: {phrase: [[start, end], [start, end], ...]}}
        
    # Generate metadata
    print("Generating metadata...")
    metadata = []
    processed_count = 0
    
    for segment_id, phrase_segments in segment_data.items():
        # Get corresponding audio file ID (remove segment suffix)
        if DATASET_NAME == "AudioSet_SL":
            audio_id = "Y" + "_".join(segment_id.split("_")[:-1])
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.flac")

        elif DATASET_NAME == "DESED_Strong":
            audio_id = segment_id
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
        
        elif DATASET_NAME == "DESED_Strong_fromAS":
            audio_id = segment_id
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
            if not os.path.exists(audio_path):
                print(f'Skipping row {row["filename"]} because it is in eval split')
                continue
        
        if not os.path.exists(audio_path):
            continue
        caption = caption_dict.get(audio_id, "")
        
        phrases = []
        for phrase, segments in phrase_segments.items():
            phrases.append({
                "phrase": phrase,
                "segments": segments  # segments is already a list of [start, end] pairs
            })
        
        # Create metadata entry
        metadata_entry = {
            "audio_id": segment_id,
            "audio_path": audio_path,
            "caption": caption,
            "phrases": phrases
        }
        
        metadata.append(metadata_entry)
        processed_count += 1
        
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} segments...")
    
    # Write to JSONL file
    print(f"Writing {len(metadata)} entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Done! Generated {len(metadata)} metadata entries.")
    print(f"Sample entry: {json.dumps(metadata[100], indent=2)}")