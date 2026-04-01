import pandas as pd
import os
import torchaudio
from tqdm import tqdm

INPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/audioset_strong_eval.tsv'

AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/audio/AudioSet_SL_flac/'
OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioSet_SL/AudioSet_SL_Eval.tsv'
DURATION_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioSet_SL/AudioSet_SL_Eval_Duration.tsv'

from audioset_sl_labels import mids, labels

if __name__ == "__main__":
    mid_to_label = dict(zip(mids, labels))

    df = pd.read_csv(INPUT_FILE, sep='\t')
    df['filename'] = 'Y' + df['filename'].str.split('_').str[:-1].str.join('_') + '.flac'
    df['event_label'] = df['event_label'].map(mid_to_label)
    df['audio_path'] = AUDIO_DIR + df['filename']
    df = df[df['audio_path'].apply(os.path.exists)]

    unique_files = df[['filename', 'audio_path']].drop_duplicates()
    durations_dict = {}

    # sanity check
    for _, row in tqdm(unique_files.iterrows(), total=len(unique_files), desc="Checking unique audio files"):
        audio_path = row['audio_path']
        filename = row['filename']
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            durations_dict[filename] = duration
        except:
            durations_dict[filename] = None

    # metadata filtering
    df = df[df['filename'].isin([f for f, d in durations_dict.items() if d is not None])]
    df = df[['filename', 'onset', 'offset', 'event_label']]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, sep='\t', index=False)

    # make duration file
    duration_df = df[['filename']].drop_duplicates().copy()
    duration_df['duration'] = duration_df['filename'].map(durations_dict)
    duration_df.to_csv(DURATION_FILE, sep='\t', index=False)