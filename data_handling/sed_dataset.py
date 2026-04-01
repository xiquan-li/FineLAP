import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os.path
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
from data_handling.data_utils import pad_sequence
from collections import defaultdict
import ruamel.yaml as yaml
from models.feature_extractor import EAT_preprocess



def csv_to_multilabel(csv_path: str, class_list: list, 
                      time_resolution=0.01, total_duration=10.0):
    
    df = pd.read_csv(csv_path, sep='\t')
    num_classes = len(class_list)
    num_steps = int(total_duration / time_resolution)
    audio_to_labels = defaultdict(
        lambda: np.zeros((num_steps, num_classes), dtype=np.float32)
    )  
    
    class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
    
    for _, row in df.iterrows():
        audio_file = row['segment_id']
        cls = row['label']
        onset = row['start_time_seconds']
        offset = row['end_time_seconds']
        
        if cls not in class_to_idx:
            continue  
        
        start_step = int(onset / time_resolution)
        end_step = min(int(offset / time_resolution), num_steps)
        
        audio_to_labels[audio_file][start_step:end_step, class_to_idx[cls]] = 1.0
    
    return audio_to_labels, class_to_idx


class AudiosetStrongDataset(Dataset):

    def __init__(self,
                 audio_dir: str,
                 metadata_file: str,
                 classes: list, 
                 sample_rate: int = 32000,
                 max_length: int = 10,
                 dataset_name: str = "AudioSet_Strong",
                 time_resolution: float = 1/3.2, 
                 **kwargs):  
        
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_file, sep='\t')
        self.metadata_file = metadata_file
        self.classes = classes
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.dataset_name = dataset_name
        audio_ids = []
        self.audio_ids = []
        self.wav_paths = []

        for _, row in self.df.iterrows(): 
            audio_ids.append(row["segment_id"])
        audio_ids = list(set(audio_ids)) 

        for audio_id in audio_ids: 
            wav_id = "Y" + "_".join(audio_id.split("_")[:-1])
            wav_path = os.path.join(self.audio_dir, f"{wav_id}.flac")
            if os.path.exists(wav_path): 
                self.audio_ids.append(audio_id)
                self.wav_paths.append(wav_path)

        self.audio_to_labels, self.class_to_idx = \
            csv_to_multilabel(
                metadata_file, classes, time_resolution, max_length)

    def __len__(self):
        return len(self.audio_ids)
    
    def __getitem__(self, index):
        audio_id = self.audio_ids[index]
        wav_path = self.wav_paths[index]
        labels = self.audio_to_labels[audio_id] # (T, num_classes)

        try: 
            wav_info = torchaudio.info(wav_path)
            waveform, sr = torchaudio.load(
                wav_path, num_frames=self.max_length*wav_info.sample_rate
            )   # (1, wav_length)
        except (FileNotFoundError, RuntimeError) as e: 
            waveform = torch.zeros((1, 32000))
            sr = 16000  
            # print(f"Encountered error when loading {wav_path}")

        if waveform.shape[-1] < 0.1*self.sample_rate: 
            waveform = torch.zeros(self.max_length*self.sample_rate)
        else: 
            waveform = waveform[0]
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)  # 32k for HTSAT
            waveform = resampler(waveform)  

        output = {
            "audio_id": audio_id, 
            "waveform": waveform, 
            "labels": labels
        }
        return output


    def collate_fn(self, data_batch): 
        # TODO for the mask calculation
        waveforms = [data["waveform"] for data in data_batch]  # (btz, t*sr)
        labels = [data["labels"] for data in data_batch]  # (btz, T, num_classes)
        audio_ids = [data["audio_id"] for data in data_batch]
        waveforms, _ = pad_sequence(waveforms)
        labels, labels_length = pad_sequence(labels)
        labels_length = np.array(labels_length)

        output = {
            "waveform": waveforms, 
            "label": labels, 
            "label_len": labels_length,
            "audio_ids": audio_ids
        }
        return output


class SoundEventDataset():

    def __init__(self,
                 audio_dir,
                 metadata_file,
                 audio_duration_file, 
                 sample_rate: int = 32000,
                 max_length: int = 10,
                 dataset_name: str = "DESED",
                 return_type: str = "raw",  # raw or mel
                 **kwargs):  
        
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_file, sep='\t')
        self.metadata_file = metadata_file
        self.classes = [x for x in sorted([i for i in self.df['event_label'].unique() if type(i) == str])]  # sound event names
        self.class_to_idx = {}  # sound event to id map
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.return_type = return_type
        self.audio_duration_file = audio_duration_file
        self.audio_ids = []
        for _, row in self.df.iterrows(): 
            self.audio_ids.append(row["filename"])
        self.audio_ids = list(set(self.audio_ids)) 

    def __len__(self):
        return len(self.audio_ids)
    
    def __getitem__(self, index):
        audio_id = self.audio_ids[index]
        wav_path = os.path.join(self.audio_dir, audio_id)
        try: 
            wav_info = torchaudio.info(wav_path)
            waveform, sr = torchaudio.load(
                wav_path, num_frames=self.max_length*wav_info.sample_rate
            )   #[1, wav_length]
        except (FileNotFoundError, RuntimeError) as e: 
            waveform = torch.zeros((1, 32000))
            sr = 16000  
            print(f"{wav_path} not found")

        if waveform.shape[-1] < 0.1*self.sample_rate: 
            waveform = torch.zeros(self.max_length*self.sample_rate)
        else: 
            waveform = waveform[0]
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)  # 32k for HTSAT
            waveform = resampler(waveform)  

        if self.return_type == "mel":
            audio = EAT_preprocess(waveform=waveform)
        elif self.return_type == "raw":
            audio = waveform
        
        return {
            "audio_id": audio_id,
            "audio": audio
        }

    def collate_fn(self, data_batch):
        """Collate function for batch processing"""
        audio_ids = []
        audios = []
        
        for data in data_batch:
            audio_ids.append(data["audio_id"])
            audios.append(data["audio"])
        
        # Pad waveforms or concatenate mel spectrograms
        if self.return_type == "raw":
            padded_audios, audio_lengths = pad_sequence(audios)
        elif self.return_type == "mel":
            padded_audios = torch.cat(audios, dim=0)
        else:
            raise ValueError(f"Invalid return type: {self.return_type}")
        
        return {
            "audio_id": audio_ids,
            "audios": padded_audios
        }



if __name__ == "__main__": 
    config = "/home/xiquan/F-CLAP/config/lsed_audioset_strong.yaml"

    with open(config, "r") as f:
        try: 
            config = yaml.safe_load(f)
        except AttributeError as e:  
            from ruamel.yaml import YAML   # issue of version mismatch 
            y = YAML(typ='safe', pure=True)
            config = y.load(f)  

    toy_dataset = AudiosetStrongDataset(
        classes=mids, 
        **config["data_args"]["basic_data_args"], 
        **config["data_args"]["train_data_args"]
    )
    dataloader = DataLoader(dataset=toy_dataset,
                            collate_fn=toy_dataset.collate_fn,
                            shuffle=False,
                            batch_size=config["data_args"]["train_data_args"]["batch_size"],
                            num_workers=config["data_args"]["train_data_args"]["num_workers"])

    for batch in dataloader:
        audio, label, length, audio_ids = \
            batch["waveform"], batch["label"], batch['label_len'], batch["audio_ids"]
        print(audio.shape)
        break