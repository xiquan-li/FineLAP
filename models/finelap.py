import torch
import torch.nn as nn
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
import torch.nn.functional as F
from models.losses import SigmoidLoss, GroundingLoss, InfoNCELoss
import ruamel.yaml as yaml
from utils.logger_config import main_logger
from utils.utils import get_rank

class FineLAP(nn.Module): 

    def __init__(self, config):
        super().__init__()
        if type(config) == str:   # load from config path
            with open(config, "r") as f:
                try: 
                    config = yaml.safe_load(f)
                except AttributeError as e:  
                    from ruamel.yaml import YAML   # issue of version mismatch 
                    y = YAML(typ='safe', pure=True)
                    config = y.load(f)  
            
        self.audio_encoder = AudioEncoder(config['model_args'])
        self.text_encoder = TextEncoder(config['model_args'])

        self.embed_size = config['model_args']["embed_size"] 
        self.audio_width = self.audio_encoder.audio_width
        self.text_width = self.text_encoder.text_width

        if config["model_args"]["temp_global"] != 0:
            self.temp_global = nn.Parameter(torch.ones([]) * config["model_args"]["temp_global"])
        if config["model_args"]["b_global"] != 0:
            self.b_global = nn.Parameter(torch.ones([]) * config["model_args"]["b_global"])
        if config["model_args"]["temp_local"] != 0: 
            self.temp_local = nn.Parameter(torch.ones([]) * config["model_args"]["temp_local"])
        if config["model_args"]["b_local"] != 0: 
            self.b_local = nn.Parameter(torch.ones([]) * config["model_args"]["b_local"])

        self.global_audio_proj = nn.Sequential(
            nn.Linear(self.audio_width, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size),
        )

        self.global_text_proj = nn.Sequential(
            nn.Linear(self.text_width, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size),
        )

        self.local_audio_proj_type = config["model_args"]["local_audio_proj_type"]
        if self.local_audio_proj_type == "rnn": 
            self.local_audio_proj = nn.GRU(
                input_size = self.audio_encoder.audio_width,
                hidden_size = int(self.embed_size/2), 
                num_layers = 2,
                batch_first = True, 
                bidirectional = True
            )
        elif self.local_audio_proj_type == "linear": 
            self.local_audio_proj = nn.Sequential(
                nn.Linear(self.audio_encoder.audio_width, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size)
            )
        elif self.local_audio_proj_type == "transformer":
            # Standard transformer encoder without cross-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_size,
                nhead=8,
                dim_feedforward=self.embed_size * 4,  # Standard transformer uses 4x d_model
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=2
            )
            self.local_audio_proj = nn.Sequential(
                nn.Linear(self.audio_encoder.audio_width, self.embed_size),
                transformer_encoder
            )
        elif self.local_audio_proj_type == "transformer_linearlast":
            # Standard transformer encoder without cross-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.audio_encoder.audio_width,
                nhead=8,
                dim_feedforward=self.audio_encoder.audio_width * 4,  # Standard transformer uses 4x d_model
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=2
            )
            self.local_audio_proj = nn.Sequential(
                transformer_encoder,
                nn.Linear(self.audio_encoder.audio_width, self.embed_size),
            )
        else: 
            raise ValueError(f"Invalid local audio proj type: {self.local_audio_proj_type}")
        
        self.normalize_dense_audio_embeds = config["model_args"]["normalize_dense_audio_embeds"]
        self.unify_audio_proj = config["model_args"]["unify_audio_proj"]  # use audio adapter for both global and dense audio embedding extraction
        if get_rank() == 0: 
            main_logger.info(f"Normalize dense audio embeds: {self.normalize_dense_audio_embeds}")
            main_logger.info(f"Unify audio proj: {self.unify_audio_proj}")
        
        self.loss_type = config["training"]["loss_type"]
        if "sigmoid" in self.loss_type:
            self.sigmoid_loss_fn = SigmoidLoss(rank=0, world_size=1, dist_impl='reduce')  # Will be updated in training
        elif "infonce" in self.loss_type:
            self.infonce_loss_fn = InfoNCELoss(rank=0, world_size=1, dist_impl='reduce')  # Will be updated in training
        if "grounding" in self.loss_type: 
            self.grounding_loss_fn = GroundingLoss()

    def update_loss_fn_params(self, rank=0, world_size=1, dist_impl='reduce'):
        """Update distributed training parameters for loss functions."""
        main_logger.info(f"Updating loss functions: rank={rank}, world_size={world_size}, dist_impl={dist_impl}")
        if "sigmoid" in self.loss_type: 
            self.sigmoid_loss_fn = SigmoidLoss(rank=rank, world_size=world_size, dist_impl=dist_impl)
        elif "infonce" in self.loss_type:
            self.infonce_loss_fn = InfoNCELoss(rank=rank, world_size=world_size, dist_impl=dist_impl)

    def interpolate(self, x, ratio):
        # x: (btz, seq, dim)
        (batch_size, time_steps, dim) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)  # repeat in the **third** dimension by ratio
        upsampled = upsampled.reshape(batch_size, time_steps * ratio, dim)
        return upsampled

    def get_global_text_embeds(self, text): 
        text_feats, _ = self.text_encoder(text)
        text_embeds = F.normalize(self.global_text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds
    
    def get_global_audio_embeds(self, audio):
        if self.unify_audio_proj:  # use the fine-grained audio adapter for global audio embeds extraction
            audio_feats = self.audio_encoder(audio)
            audio_embeds = self.local_audio_proj(audio_feats)
            if self.local_audio_proj_type == "rnn": 
                audio_embeds = audio_embeds[0]
            if self.audio_encoder.audio_encoder_type == "htsat":
                global_audio_embeds = F.normalize(torch.mean(audio_embeds, dim=1), dim=-1)
            elif self.audio_encoder.audio_encoder_type == "eat":
                global_audio_embeds = F.normalize(audio_embeds[:, 0, :], dim=-1)
            return global_audio_embeds
            
        else:
            audio_feats = self.audio_encoder(audio)  
            if self.audio_encoder.audio_encoder_type == "htsat":
                audio_feats = torch.mean(audio_feats, dim=1)
            elif self.audio_encoder.audio_encoder_type == "eat":
                audio_feats = audio_feats[:, 0, :]
            audio_embeds = F.normalize(self.global_audio_proj(audio_feats), dim=-1)
            return audio_embeds

    def get_dense_audio_embeds(self, audio): 
        audio_feats = self.audio_encoder(audio)
        if self.audio_encoder.audio_encoder_type == "eat":
            audio_feats = audio_feats[:, 1:, :]
        audio_embeds = self.local_audio_proj(audio_feats)
        if self.local_audio_proj_type == "rnn": 
            audio_embeds = audio_embeds[0]
        
        if self.normalize_dense_audio_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    def forward(self, 
                audio: torch.Tensor, 
                captions: list[str], 
                phrases: list[list[str]] = None, 
                frame_labels_batch: list[torch.Tensor] = None, 
                has_grounding: list[bool] = None, 
                batch_audio_chosen_segments: torch.Tensor = None,
                batch_dense_audio_embeds_idx: torch.Tensor = None):
        """
        Forward pass that computes CLIP-style and/or grounding losses based on loss_type.

        Args:
            audio: audio waveforms [B, samples]
            captions: list of caption strings [B]
            phrases: list of phrase lists [[phrase1, phrase2], [phrase3], ...] or None
            frame_labels_batch: list of tensors [N_i, T] for grounding loss
            has_grounding: list of bool indicating which samples have grounding data

        Returns:
            tuple: (clip_loss, grounding_loss)
        """
        clip_loss = torch.tensor(0.0, device=audio.device)
        grounding_loss = torch.tensor(0.0, device=audio.device)
        distillation_loss = torch.tensor(0.0, device=audio.device)

        # 0) Prepare embeddings
        global_text_embeds = self.get_global_text_embeds(captions)
        audio_feats = self.audio_encoder(audio)

        if self.unify_audio_proj:
            dense_audio_embeds = self.local_audio_proj(audio_feats)
            if self.local_audio_proj_type == "rnn":
                dense_audio_embeds = dense_audio_embeds[0]

            if self.audio_encoder.audio_encoder_type == "htsat": # for HTS-AT: pool the last layer features to get the global audio embeds
                global_audio_embeds = F.normalize(torch.mean(dense_audio_embeds, dim=1), dim=-1)
            elif self.audio_encoder.audio_encoder_type == "eat": # for EAT: use the cls token to get the global audio embeds
                dense_audio_embeds = dense_audio_embeds[:, 1:, :]
                global_audio_embeds = F.normalize(dense_audio_embeds[:, 0, :], dim=-1)

        else:
            if self.audio_encoder.audio_encoder_type == "htsat":
                global_audio_embeds = F.normalize(
                    self.global_audio_proj(torch.mean(audio_feats, dim=1)),
                    dim=-1
                )
                dense_audio_embeds = self.local_audio_proj(audio_feats)
            elif self.audio_encoder.audio_encoder_type == "eat":
                global_audio_embeds = F.normalize(
                    self.global_audio_proj(audio_feats[:, 0, :]),
                    dim=-1
                )
                dense_audio_embeds = self.local_audio_proj(audio_feats[:, 1:, :])
            if self.local_audio_proj_type == "rnn":
                dense_audio_embeds = dense_audio_embeds[0]

        if self.normalize_dense_audio_embeds:
            dense_audio_embeds = F.normalize(dense_audio_embeds, dim=-1)

        temp_global_val = getattr(self, "temp_global", None)
        b_global_val = getattr(self, "b_global", None)
        temp_local_val = getattr(self, "temp_local", None)
        b_local_val = getattr(self, "b_local", None)

        # 1) Compute CLIP-style loss if needed
        if "sigmoid" in self.loss_type or "infonce" in self.loss_type:
            if "sigmoid" in self.loss_type:
                clip_loss = self.sigmoid_loss_fn(
                    global_audio_embeds,
                    global_text_embeds,
                    temp_global_val,
                    b_global_val
                )
            elif "infonce" in self.loss_type:
                clip_loss = self.infonce_loss_fn(
                    global_audio_embeds,
                    global_text_embeds,
                    temp_global_val,
                    b_global_val
                )
            else: 
                raise ValueError(f"Invalid loss type: {self.loss_type}")

        # 2) Compute grounding loss if needed
        if "grounding" in self.loss_type and phrases is not None and frame_labels_batch is not None and has_grounding is not None:
            if any(has_grounding):
                # Get phrase embeddings
                phrase_embeds = []
                for phrase_list in phrases:
                    if phrase_list is not None and len(phrase_list) > 0:
                        embeds = self.get_global_text_embeds(phrase_list)  # [N, D]
                        phrase_embeds.append(embeds)
                    else:
                        phrase_embeds.append(None)

                grounding_loss = self.grounding_loss_fn(
                    phrase_embeds,
                    frame_labels_batch,
                    dense_audio_embeds,
                    temp_local_val,
                    b_local_val,
                    audio.device,
                    has_grounding
                )

        # 3) Compute distillation loss (deprecated)
        if "distillation" in self.loss_type:
            with torch.no_grad():
                assert batch_audio_chosen_segments is not None, "batch_audio_chosen_segments is None"
                B, N_chosen, _, T_seg, Freq = batch_audio_chosen_segments.shape
                batch_audio_chosen_segments = batch_audio_chosen_segments.reshape(B * N_chosen, 1, T_seg, Freq)
                teacher_segments_embeds = self.get_global_audio_embeds(batch_audio_chosen_segments)  # [B * N_chosen, D]
                teacher_segments_embeds = teacher_segments_embeds.reshape(B, N_chosen, -1) # [B, N_chosen, D]

            batch_dense_audio_embeds_idx = batch_dense_audio_embeds_idx.reshape(B, -1)  # [B, N_chosen * embeds_per_segment]
            idx = batch_dense_audio_embeds_idx.unsqueeze(-1).expand(-1, -1, dense_audio_embeds.size(-1))   # [B, N_chosen * embeds_per_segment, D]
            student_segments_embeds = torch.gather(dense_audio_embeds, 
                                                   dim=1, 
                                                   index=idx)  # [B, N_chosen * embeds_per_segment, D]
            student_segments_embeds = student_segments_embeds.reshape(B, N_chosen, -1, student_segments_embeds.size(-1)) # [B, N_chosen, embeds_per_segment, D]
            student_segments_embeds_pooled = torch.mean(student_segments_embeds, dim=2) # [B, N_chosen, D]
            
            distillation_loss = 1.0 - (student_segments_embeds_pooled * teacher_segments_embeds).sum(-1).mean()

        return clip_loss, grounding_loss, distillation_loss