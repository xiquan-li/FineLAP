#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
from models.feature_extractor import EAT_preprocess
from data_handling.data_utils import pad_sequence

class AudioCaptionDataset(Dataset):
    """
    Audio caption dataset for retrieval and classification
    Data format: {"audio_id": audio_id, "audio_path": audio_path, "caption": [captions] or caption}
    """

    def __init__(self,
                 metadata_file,
                 sample_rate: int = 32000,
                 max_length: int = 10,
                 dataset_name: str = "",
                 return_type: str = "raw",  # raw or mel
                 **kwargs):
        
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.return_type = return_type
        # print(f"Audio Dataset: {dataset_name}, audio_dir: {self.audio_dir}")

        # Load data and expand multi-caption entries
        self.data = []
        with open(metadata_file, 'r') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                audio_id = item["audio_id"]
                audio_path = item.get("audio_path", "")
                captions = item.get("caption", [])
                
                if isinstance(captions, str):
                    captions = [captions]
                
                for caption in captions:
                    self.data.append({
                        "audio_id": audio_id,
                        "audio_path": audio_path,
                        "caption": caption
                    })
    
    def __getitem__(self, index):
        item = self.data[index]
        audio_id = item["audio_id"]
        audio_path = item["audio_path"]
        caption = item["caption"]
        
        # Load audio
        try:
            wav_info = torchaudio.info(audio_path)
            waveform, sr = torchaudio.load(
                audio_path, num_frames=self.max_length * wav_info.sample_rate
            )
        except Exception as e:
            waveform = torch.zeros((1, 32000))
            sr = 16000
            print(f"{audio_path} not found")
        
        if waveform.shape[-1] < 0.1 * self.sample_rate:
            waveform = torch.zeros((self.max_length * self.sample_rate))
        else:
            waveform = waveform[0]
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        if self.return_type == "mel":
            audio = EAT_preprocess(waveform=waveform)
        elif self.return_type == "raw":
            audio = waveform

        return {
            "audio_id": audio_id,
            "audio": audio,
            "caption": caption
        }
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, data_batch):
        audio_ids = []
        audios = []
        captions = []
        
        for data in data_batch:
            audio_ids.append(data["audio_id"])
            audios.append(data["audio"])
            captions.append(data["caption"])
        
        # Pad waveforms
        if self.return_type == "raw":
            padded_audios, waveform_lengths = pad_sequence(audios)
        elif self.return_type == "mel":
            padded_audios = torch.cat(audios, dim=0)
        else:
            raise ValueError(f"Invalid return type: {self.return_type}")
        
        return {
            "audio_id": audio_ids,
            "audios": padded_audios,
            "caption": captions
        }