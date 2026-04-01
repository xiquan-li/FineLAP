# FineLAP 训练和推理制作AudioCaption Data的Meta

import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATASET_NAME = "VGGSound"

if DATASET_NAME == "AudioCaps_test":
    INPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/json_files/test.json"
    OUTPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/Meta_FineLAP/test.jsonl"
    AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/AudioCaps/audio/test'

elif DATASET_NAME == "AudioCaps_val":
    INPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/json_files/val.json"
    OUTPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/Meta_FineLAP/val.jsonl"
    AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/AudioCaps/audio/val'

elif DATASET_NAME == "AudioCaps_train":
    INPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/json_files/train.json"
    OUTPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/Meta_FineLAP/train.jsonl"
    AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/AudioCaps/audio/train'

elif DATASET_NAME == "Clotho_train":
    INPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/data/Clotho/clotho_captions_development.csv"
    OUTPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/Clotho/Meta_FineLAP/train.jsonl"
    AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/Clotho/development'

elif DATASET_NAME == "Clotho_test":
    INPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/data/Clotho/clotho_captions_evaluation.csv"
    OUTPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/Clotho/Meta_FineLAP/test.jsonl"
    AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/Clotho/evaluation'

elif DATASET_NAME == "Clotho_val":
    INPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/data/Clotho/clotho_captions_validation.csv"
    OUTPUT_FILE = "/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/Clotho/Meta_FineLAP/val.jsonl"
    AUDIO_DIR = '/apdcephfs_gy4/share_302507476/xiquanli/data/Clotho/validation'

elif DATASET_NAME == "FreeSound":
    INPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/caption_tsv/FreeSound.tsv'
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/WavCaps/Meta_FineLAP/FreeSound.jsonl'
    AUDIO_DIR = '/apdcephfs_gy2/share_302507476/0_public_datasets/video2audio/hq_pure_audio_public_dataset/audio_file_root/WavCaps/data/waveforms_split/FreeSound_flac'

elif DATASET_NAME == "BBC_Sound_Effects":
    INPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/caption_tsv/BBC_Sound_Effects.tsv'
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/WavCaps/Meta_FineLAP/BBC_Sound_Effects.jsonl'
    AUDIO_DIR = '/apdcephfs_gy2/share_302507476/0_public_datasets/video2audio/hq_pure_audio_public_dataset/audio_file_root/WavCaps/data/waveforms_split/BBC_Sound_Effects_flac'

elif DATASET_NAME == "SoundBible":
    INPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/data/WavCaps/caption_tsv/SoundBible_flac.tsv'
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/WavCaps/Meta_FineLAP/SoundBible.jsonl'
    AUDIO_DIR = '/apdcephfs_gy2/share_302507476/0_public_datasets/video2audio/hq_pure_audio_public_dataset/audio_file_root/WavCaps/data/waveforms_split/SoundBible_flac'

elif DATASET_NAME == "AudioSet":
    INPUT_FILE = '/apdcephfs_zwfy2/share_303944931/xiquanli/data/AudioSet/caption_tsv/unbal_train_caption.tsv'
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioSet/unbal_train.jsonl'
    AUDIO_DIR = '/apdcephfs_zwfy2/share_303944931/xiquanli/data/AudioSet/audio/unbal_train'

elif DATASET_NAME == "VGGSound":
    INPUT_FILE = '/apdcephfs_zwfy2/share_303944931/xiquanli/data/AudioSet/AudioSetCaps/Dataset/VGGSound_AudioSetCaps_caption.csv'
    OUTPUT_FILE = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/VGGSound/train.jsonl'
    AUDIO_DIR = '/apdcephfs_zwfy2/share_303944931/xiquanli/data/VGGSound/audio'


AUDIOCAPS_TEST = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/AudioCaps/Meta_FineLAP/test_zw.jsonl'
CLOTHO_TEST = '/apdcephfs_gy4/share_302507476/xiquanli/ALE/F-CLAP/data/Clotho/Meta_FineLAP/test_zw.jsonl'

AudioCaps_test_id = [json.loads(line)['audio_id'].split('.')[0] for line in open(AUDIOCAPS_TEST, 'r')]
Clotho_test_id = [json.loads(line)['audio_id'].split('.')[0] for line in open(CLOTHO_TEST, 'r')]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def process_item(item):
    """处理单个数据项"""
    new_d = {}
    
    ## ************************Step1: Filter according to audio id ************************
    if 'AudioSet' in DATASET_NAME:
        if 'Y' + item['id'] in AudioCaps_test_id: 
            print(f"Audio {item['id']} is in AudioCaps_test")
            return None
    
    ## ************************Step2: Add audio_id and audio_path ************************
    if 'AudioCaps' in DATASET_NAME:
        new_d['audio_id'] = item['id']
        new_d['audio_path'] = os.path.join(AUDIO_DIR, item['id'])
    elif 'Clotho' in DATASET_NAME:
        new_d['audio_id'] = item['file_name']
        new_d['audio_path'] = os.path.join(AUDIO_DIR, item['file_name'])
    elif DATASET_NAME == "FreeSound" or DATASET_NAME == "BBC_Sound_Effects" or DATASET_NAME == "SoundBible" or DATASET_NAME == "AudioSet":
        audio_id = str(item['id'])
        new_d['audio_id'] = audio_id
        audio_path = os.path.join(AUDIO_DIR, audio_id + '.flac')
        if not os.path.exists(audio_path):
            audio_path = os.path.join(AUDIO_DIR, audio_id + '_0.flac')  # for first clips
        new_d['audio_path'] = audio_path
    elif DATASET_NAME == "VGGSound":
        new_d['audio_id'] = item['id']
        new_d['audio_path'] = os.path.join(AUDIO_DIR, item['id'] + '.wav')
    
    if not os.path.exists(new_d['audio_path']):
        return None
        
    ## ************************Step3: Add caption ************************
    if DATASET_NAME == "AudioCaps_val" or DATASET_NAME == "AudioCaps_test" or DATASET_NAME == "Clotho_val" or DATASET_NAME == "Clotho_test" or DATASET_NAME == "Clotho_train":
        new_d['caption'] = [item['caption_1'], item['caption_2'], item['caption_3'], item['caption_4'], item['caption_5']]
    elif DATASET_NAME == 'AudioCaps_train' or DATASET_NAME == "FreeSound" or DATASET_NAME == "BBC_Sound_Effects" or DATASET_NAME == "SoundBible" or DATASET_NAME == "AudioSet":
        new_d['caption'] = item['caption']
    elif DATASET_NAME == "VGGSound":
        new_d['caption'] = item['caption'].strip()
    
    return new_d

if __name__ == "__main__":
    if DATASET_NAME == "AudioCaps_test" or DATASET_NAME == "AudioCaps_val" or DATASET_NAME == "AudioCaps_train":
        with open(INPUT_FILE, "r") as f:
            data = json.load(f)['data']

    elif DATASET_NAME == "Clotho_train" or DATASET_NAME == "Clotho_val" or DATASET_NAME == "Clotho_test":
        data = pd.read_csv(INPUT_FILE)
        data = data.to_dict(orient='records')

    elif DATASET_NAME == "FreeSound" or DATASET_NAME == "BBC_Sound_Effects" or DATASET_NAME == "SoundBible":
        data = pd.read_csv(INPUT_FILE, sep="\t")
        data = data.to_dict(orient='records')

    elif DATASET_NAME == "AudioSet":
        data = pd.read_csv(INPUT_FILE, sep='\t')
        data = data.to_dict(orient='records')

    elif DATASET_NAME == "VGGSound":
        data = pd.read_csv(INPUT_FILE, sep=',')
        data = data.to_dict(orient='records')

    else: 
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    # 使用多进程处理
    num_workers = min(cpu_count(), 16)  # 最多使用16个进程
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_item, data),
            total=len(data),
            desc="Processing data {}".format(DATASET_NAME)
        ))
    
    # 过滤掉None结果（文件不存在的项）
    new_data = [r for r in results if r is not None]

    with open(OUTPUT_FILE, "w") as f:
        for new_d in new_data:
            json.dump(new_d, f)
            f.write("\n")
        
    print(f"Caption data saved to {OUTPUT_FILE}")
    print(f"Total {len(new_data)} data saved")