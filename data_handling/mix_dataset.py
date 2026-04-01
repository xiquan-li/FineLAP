import math
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torchaudio
from torchaudio.transforms import Resample
from utils.logger_config import main_logger
from models.feature_extractor import EAT_preprocess
from data_handling.data_utils import pad_sequence
from torchlibrosa.augmentation import SpecAugmentation

PHRASE_BANK = './data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl'


class MixDataset(Dataset):
    """
    Mixed dataset for both retrieval and grounding 
    Data format: 
    - With grounding: {"audio_id": audio_id, "audio_path": audio_path, "caption": caption, "phrases": [{"phrase": phrase, "segments": [onset, offset]}]}
    - Without grounding: {"audio_id": audio_id, "audio_path": audio_path, "caption": caption}
    """

    def __init__(self,
                 metadata_files,
                 sample_rate: int = 32000,
                 max_length: int = 10,
                 time_resolution: float = 10/32,
                 use_alternatives: bool = True,
                 max_phrases: int = 20,
                 return_type: str = "raw",  # raw or mel
                 spec_aug: bool = False, 
                 max_split_segments: int = 1, # split wav into N segments, for distillation loss 
                 max_chosen_segments: int = 1, # max number of segments to choose from each wav
                 **kwargs):  
        
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.time_resolution = time_resolution
        self.use_alternatives = use_alternatives
        self.max_phrases = max_phrases
        self.dataset_name = kwargs.get("dataset_name", "")
        self.return_type = return_type
        self.spec_aug = spec_aug
        self.N_split = max_split_segments
        self.N_chosen = max_chosen_segments
        assert self.N_chosen <= self.N_split, f"N_chosen {self.N_chosen} must be less than or equal to N_split {self.N_split}"
        if self.N_chosen != 1 and self.N_split != 1:
            assert self.return_type == "mel", f"return_type {self.return_type} must be mel when N_chosen {self.N_chosen} and N_split {self.N_split} are not 1"

        with open(PHRASE_BANK, 'r') as f:
            self.phrase_bank = [json.loads(line) for line in f]
            
        main_logger.info(f"Max phrases for sample: {max_phrases}")
        
        # cluster is the main phrase name, alternatives include the cluster itself and variants
        self.cluster2alternatives = {phrase['phrase']: phrase['alternatives'] for phrase in self.phrase_bank}
        self.alternative2cluster = {alt: phrase['phrase'] for phrase in self.phrase_bank for alt in phrase['alternatives']}
        self.all_clusters = list(self.cluster2alternatives.keys())
        self.all_phrases = list(self.alternative2cluster.keys())

        data_with_grounding = []
        data_without_grounding = []
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    phrases = item.get("phrases", [])
                    has_valid_grounding = False
                    if phrases:
                        for phrase_item in phrases:
                            if phrase_item.get("segments") and len(phrase_item["segments"]) > 0:
                                has_valid_grounding = True
                                break
                    
                    if has_valid_grounding:
                        data_with_grounding.append(item)
                    else:
                        data_without_grounding.append(item)
        
        self.data = data_with_grounding + data_without_grounding
        self.num_with_grounding = len(data_with_grounding)
        main_logger.info(f'Number of clusters: {len(self.all_clusters)}, number of all phrases: {len(self.all_phrases)}')
        main_logger.info(f"Number of data with grounding: {self.num_with_grounding}, number of data without grounding: {len(data_without_grounding)}, total number of data: {len(self.data)}")
        if self.N_chosen != 1 and self.N_split != 1: 
            main_logger.info(f"Max split segments: {self.N_split}, max chosen segments: {self.N_chosen}")

    def _sample_alternative(self, phrase):
        """Sample a random alternative for a phrase (cluster or alternative)"""
        cluster = self.alternative2cluster.get(phrase, phrase)
        alternatives = self.cluster2alternatives.get(cluster, [phrase])
        return random.choice(alternatives)
    
    def _sample_negative_phrases(self, positive_clusters):
        """Sample negative phrase clusters and their alternatives"""
        excluded_clusters = set(positive_clusters)
        available_clusters = [c for c in self.all_clusters if c not in excluded_clusters]
        
        n_positive = len(positive_clusters)
        n_negative = max(0, self.max_phrases - n_positive)
        
        if n_negative == 0 or not available_clusters:
            return []
        
        n_negative = min(n_negative, len(available_clusters))
        negative_clusters = random.sample(available_clusters, n_negative)
        
        if self.use_alternatives:
            negative_phrases = [self._sample_alternative(cluster) for cluster in negative_clusters]
        else:
            negative_phrases = negative_clusters
        
        return negative_phrases

    def __getitem__(self, index):
        audio_item = self.data[index]
        audio_id = audio_item["audio_id"]
        caption = audio_item.get("caption", "")
        if type(caption) != str and type(caption) != list:
            print(f"Invalid caption type: {type(caption)}, caption: {caption}")
            caption = ""
        if isinstance(caption, list):
            caption = random.choice(caption)  # randomly choose one caption from the list
        phrases_data = audio_item.get("phrases", [])
        audio_path = audio_item.get("audio_path", "")

        # Load audio
        try: 
            wav_info = torchaudio.info(audio_path)
            waveform, sr = torchaudio.load(
                audio_path, num_frames=self.max_length*wav_info.sample_rate  # first ten seconds
            )
        except (FileNotFoundError, RuntimeError) as e: 
            waveform = torch.zeros((1, 32000))
            sr = 32000  
            print(f"{audio_path} not found")

        if waveform.shape[-1] < 0.1*self.sample_rate:
            waveform = torch.zeros((1, self.max_length*self.sample_rate))
            print(f"Audio too short: {audio_path}")
        else: 
            waveform = waveform[0]
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)  

        processed_phrases = []
        if phrases_data:
            positive_clusters = []
            for phrase_item in phrases_data[:self.max_phrases]:
                phrase = phrase_item.get("phrase", "")
                if not phrase:
                    continue
                if phrase not in self.alternative2cluster:
                    # main_logger.warning(f"Phrase {phrase} not found in phrase bank")
                    print(f"Phrase {phrase} not found in phrase bank")
                cluster = self.alternative2cluster.get(phrase, phrase)
                positive_clusters.append(cluster)
                
                if self.use_alternatives:
                    sampled_phrase = self._sample_alternative(phrase)
                else:
                    sampled_phrase = phrase
                
                processed_phrases.append({
                    "phrase": sampled_phrase,
                    "segments": phrase_item.get("segments", []),
                    "is_positive": True
                })
            
            negative_phrases = self._sample_negative_phrases(positive_clusters)
            for neg_phrase in negative_phrases:
                processed_phrases.append({
                    "phrase": neg_phrase,
                    "segments": [],
                    "is_positive": False
                })
        if self.return_type == "mel":
            audio = EAT_preprocess(waveform=waveform)  # already padded to 1024 frames
            audio_chosen_segments = None
            dense_audio_embeds_idx = None

            if self.N_split != 1:
                T, F = audio.shape[-2], audio.shape[-1]
                T_seg = T // self.N_split

                split_segments = audio.reshape(1, self.N_split, T_seg, F).squeeze(0)  # [N_split, T_seg, F]
                split_segments = split_segments.unsqueeze(1)  # [N_split, 1, T_seg, F]

                chosen_indices = random.sample(range(self.N_split), self.N_chosen)
                audio_chosen_segments = split_segments[chosen_indices]  # [N_chosen, 1, T_seg, F]

                num_embeds = 64  # ad-hoc n_embeds for EAT
                embeds_per_segment = num_embeds // self.N_split
                dense_audio_embeds_idx = []
                for idx in chosen_indices:
                    start_embed = idx * embeds_per_segment
                    end_embed = (idx + 1) * embeds_per_segment if idx < self.N_split - 1 else num_embeds
                    segment_indices = list(range(start_embed, end_embed))
                    dense_audio_embeds_idx.append(segment_indices)
                dense_audio_embeds_idx = torch.tensor(dense_audio_embeds_idx)  # [N_chosen, embeds_per_segment]

            if self.spec_aug:
                spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                  time_stripes_num=2,
                                                  freq_drop_width=8,
                                                  freq_stripes_num=2)
                audio = spec_augmenter(audio)
        elif self.return_type == "raw":
            audio = waveform
            audio_chosen_segments = None
            dense_audio_embeds_idx = None # distillation loss not implemented for raw waveform
        else: 
            raise ValueError(f"Invalid return type: {self.return_type}")

        return {
            "audio_id": audio_id,
            "audio": audio, # waveform or mel
            "caption": caption,
            "phrases": processed_phrases,
            "audio_chosen_segments": audio_chosen_segments,  # [N_chosen, 1, T_seg, F]
            "dense_audio_embeds_idx": dense_audio_embeds_idx,
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data_batch):
        audio_ids = []
        audios = []
        captions = []
        batch_phrases = []
        batch_frame_labels = []
        has_grounding = []
        batch_audio_chosen_segments = []
        batch_dense_audio_embeds_idx = []

        n_frames = int(self.max_length / self.time_resolution)

        for data in data_batch:
            audio_ids.append(data["audio_id"])
            audios.append(data["audio"])
            captions.append(data["caption"])

            phrases = data.get("phrases", [])
            has_valid_grounding = False
            phrase_names = []
            frame_labels = []

            if phrases:
                for phrase_item in phrases:
                    phrase = phrase_item.get("phrase", "")
                    segments = phrase_item.get("segments", [])
                    is_positive = phrase_item.get("is_positive", True)

                    if not phrase:
                        continue

                    phrase_names.append(phrase)

                    label = np.zeros(n_frames, dtype=int)
                    if is_positive and segments:
                        has_valid_grounding = True
                        for start, end in segments:
                            onset = int(start / self.time_resolution)
                            offset = int(end / self.time_resolution)
                            onset = max(0, min(onset, n_frames-1))
                            offset = max(0, min(offset, n_frames))
                            if onset < offset:
                                label[onset:offset] = 1
                    frame_labels.append(label)

            if has_valid_grounding and phrase_names:
                frame_labels_array = np.array(frame_labels)
                frame_labels_tensor = torch.tensor(frame_labels_array, dtype=torch.int)
            else:
                phrase_names = None
                frame_labels_tensor = None

            batch_phrases.append(phrase_names)
            batch_frame_labels.append(frame_labels_tensor)
            has_grounding.append(has_valid_grounding)

            audio_chosen_segments = data.get("audio_chosen_segments")
            dense_audio_embeds_idx = data.get("dense_audio_embeds_idx")
            batch_audio_chosen_segments.append(audio_chosen_segments)
            batch_dense_audio_embeds_idx.append(dense_audio_embeds_idx)

        # Pad waveforms or Stack mel
        if self.return_type == "raw":
            padded_audios, waveform_lengths = pad_sequence(audios)
        elif self.return_type == "mel":
            padded_audios = torch.cat(audios, dim=0)
        else:
            raise ValueError(f"Invalid return type: {self.return_type}")

        if self.N_split != 1:
            batch_audio_chosen_segments = torch.stack(batch_audio_chosen_segments, dim=0)  # [bsz, N_chosen, 1, T_seg, F]
            batch_dense_audio_embeds_idx = torch.stack(batch_dense_audio_embeds_idx, dim=0)
        else:
            batch_audio_chosen_segments = None
            batch_dense_audio_embeds_idx = None

        return {
            "audio_id": audio_ids,
            "audios": padded_audios,
            "caption": captions,
            "phrases": batch_phrases,
            "frame_labels": batch_frame_labels,
            "has_grounding": has_grounding,
            "batch_audio_chosen_segments": batch_audio_chosen_segments,  # [bsz, N_chosen, 1, T_seg, F]
            "dense_audio_embeds_idx": batch_dense_audio_embeds_idx,  # [bsz, N_chosen, embeds_per_segment]
        }
