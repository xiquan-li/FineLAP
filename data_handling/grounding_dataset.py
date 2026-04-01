import pickle
import math
import json
import h5py
import numpy as np
import torch
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
import os.path
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
from models.feature_extractor import EAT_preprocess
from data_handling.data_utils import pad_sequence


class AudioGroundingDataset(Dataset):

    def __init__(self,
                 audio_dir,
                 metadata_file,
                 sample_rate: int = 32000,
                 max_length: int = 10,
                 time_resolution: float = 10/32,
                 dataset_name: str = "AudioGrounding",
                 return_type: str = "raw",  # raw or mel
                 **kwargs):  
        
        self.audio_dir = audio_dir
        self.data = json.load(open(metadata_file))
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.time_resolution = time_resolution
        self.return_type = return_type
        self.pad_keys = ['label', 'audio']
        self.ignore_collect_keys = ['phrase_segments', 'phrase']
        self.dataset_name = dataset_name
        if kwargs.get("audio_duration_file", None) is not None: 
            self.audio_duration_file = kwargs["audio_duration_file"]
            
        self.generate_index()
        
    def generate_index(self):
        self.idxs = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, phrase_item in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))

    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        caption = audio_item.get("tokens", "")
        audio_id = audio_item["audio_id"]
        phrase_item = audio_item["phrases"][phrase_idx]

        ## load audio
        wav_path = os.path.join(self.audio_dir, audio_id)
        try: 
            wav_info = torchaudio.info(wav_path)
            waveform, sr = torchaudio.load(
                wav_path, num_frames=self.max_length*wav_info.sample_rate
            )   #[1, wav_length]
        except (FileNotFoundError, RuntimeError) as e: 
            waveform = torch.zeros((1, 32000))
            phrase_item["phrase"] = ""
            sr = 16000  
            print(f"{wav_path} not found")

        if waveform.shape[-1] < 0.1*self.sample_rate: 
            phrase_item["phrase"] = ""
            waveform = torch.zeros(self.max_length*self.sample_rate)
        else: 
            waveform = waveform[0]
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)  # 32k for HTSAT
            waveform = resampler(waveform)  

        ## create frame-level strong level
        audio_duration = waveform.shape[-1] / self.sample_rate
        n_frame = math.floor(audio_duration / self.time_resolution) + 1
        label = np.zeros(n_frame, dtype=int)
        for start, end in phrase_item["segments"]: 
            onset = round(start / self.time_resolution)  # onset frame (int)
            offset = round(end / self.time_resolution)   # offset frame (int)
            label[onset: offset] = 1

        # Process audio based on return_type
        if self.return_type == "mel":
            audio = EAT_preprocess(waveform=waveform)
        elif self.return_type == "raw":
            audio = waveform

        phrase = phrase_item["phrase"]
        output = {
            "audio_id": audio_id,
            "audiocap_id": audio_item["audiocap_id"],
            "start_index": phrase_item.get("start_index", 0),
            "end_index": phrase_item.get("end_index", 1),
            "phrase_segments": phrase_item["segments"], 
            "audio": audio,
            "phrase": phrase,
            "caption": caption, 
            "label": label
        }
        return output

    def __len__(self):
        return len(self.idxs)

    def collate_fn(self, data_batch): 
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in self.ignore_collect_keys: 
                    continue
                if key in self.pad_keys:
                    # Special handling for audio based on return_type
                    if key == 'audio' and self.return_type == 'mel':
                        # Mel spectrograms are already trimmed/padded, just concatenate
                        output[key] = torch.cat(output[key], dim=0)
                    else:
                        # Use pad_sequence for raw audio and labels
                        padded_seq, length = pad_sequence(output[key])
                        output[key] = padded_seq
                        output[f"{key}_len"] = np.array(length)
                else:
                    data = np.array(output[key])  # collect everything into numpy.array
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")

        # Rename 'audio' to 'audios' for consistency with other datasets
        if 'audio' in output:
            output['audios'] = output.pop('audio')
        if 'audio_len' in output:
            output['audios_len'] = output.pop('audio_len')

        return output