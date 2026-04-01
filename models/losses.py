import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils import all_gather_tensor


class SigmoidLoss(nn.Module):
    """Sigmoid Loss for CLIP-style training with distributed support.
    
    This loss function implements the sigmoid-based contrastive loss used in CLIP training,
    with support for distributed training across multiple GPUs.
    """
    
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: str = 'reduce',
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl
        assert self.dist_impl in ('reduce', 'shift', 'gather'), f"Unsupported dist_impl: {dist_impl}"

    def get_logits(self, audio_features, text_features, logit_scale=None, logit_bias=None):
        """Compute similarity between audio and text features."""
        assert text_features.shape[0] == audio_features.shape[0], f"text_features.shape: {text_features.shape}, audio_features.shape: {audio_features.shape}"
        logits = audio_features @ text_features.T
        if logit_scale is not None:
            logits = logits / logit_scale
        if logit_bias is not None:
            logits = logits + logit_bias
        return logits

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False):
        """Generate ground truth labels for sigmoid loss."""
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def _loss(self, audio_features, text_features, logit_scale=None, logit_bias=None, negative_only=False):
        """Compute sigmoid loss between audio and text features."""
        logits = self.get_logits(audio_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            audio_features.device,
            audio_features.dtype,
            audio_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / audio_features.shape[0]
        return loss

    def forward(self, audio_features, text_features, logit_scale=None, logit_bias=None, output_dict=False):
        """Forward pass to compute sigmoid loss with distributed training support."""
        if self.world_size > 1 and self.dist_impl == "gather":
            gathered_audio = [torch.empty_like(audio_features) for _ in range(self.world_size)]
            gathered_text = [torch.empty_like(text_features) for _ in range(self.world_size)]
            dist.all_gather(gathered_audio, audio_features)
            dist.all_gather(gathered_text, text_features)
            gathered_audio[self.rank] = audio_features
            gathered_text[self.rank] = text_features
            # for i, t in enumerate(gathered_audio):
            #     print(f"[Rank {self.rank}] gathered_audio[{i}]: requires_grad={t.requires_grad}, grad={t.grad}")
            # for i, t in enumerate(gathered_text):
            #     print(f"[Rank {self.rank}] gathered_text[{i}]: requires_grad={t.requires_grad}, grad={t.grad}")
            audio_all = torch.cat(gathered_audio, dim=0)
            text_all = torch.cat(gathered_text, dim=0)
            loss = self._loss(audio_all, text_all, logit_scale, logit_bias)
        else:
            loss = self._loss(audio_features, text_features, logit_scale, logit_bias)
        
        return {"sigmoid_loss": loss} if output_dict else loss


class InfoNCELoss(nn.Module):
    """InfoNCE Loss for CLIP-style training with distributed support.

    This loss function implements the standard InfoNCE contrastive loss,
    which maximizes agreement between positive pairs while minimizing
    agreement between negative pairs.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: str = 'reduce',
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl
        assert self.dist_impl in ('reduce', 'shift', 'gather'), f"Unsupported dist_impl: {dist_impl}"

    def get_logits(self, audio_features, text_features, logit_scale=None, logit_bias=None):
        """Compute similarity between audio and text features."""
        assert text_features.shape[0] == audio_features.shape[0], f"text_features.shape: {text_features.shape}, audio_features.shape: {audio_features.shape}"
        logits = audio_features @ text_features.T
        if logit_scale is not None:
            logits = logits / logit_scale
        if logit_bias is not None:
            logits = logits + logit_bias
        return logits

    def _loss(self, audio_features, text_features, logit_scale=None, logit_bias=None):
        """Compute InfoNCE loss between audio and text features."""
        logits = self.get_logits(audio_features, text_features, logit_scale, logit_bias)

        # Create labels for cross-entropy loss (diagonal is positive)
        labels = torch.arange(audio_features.shape[0], device=audio_features.device)

        # Compute cross-entropy loss
        loss_audio_to_text = F.cross_entropy(logits, labels)
        loss_text_to_audio = F.cross_entropy(logits.T, labels)

        # Average the two losses
        loss = (loss_audio_to_text + loss_text_to_audio) / 2

        return loss

    def forward(self, audio_features, text_features, logit_scale=None, logit_bias=None, output_dict=False):
        """Forward pass to compute InfoNCE loss with distributed training support."""
        if self.world_size > 1 and self.dist_impl == "gather":
            gathered_audio = [torch.empty_like(audio_features) for _ in range(self.world_size)]
            gathered_text = [torch.empty_like(text_features) for _ in range(self.world_size)]
            dist.all_gather(gathered_audio, audio_features)
            dist.all_gather(gathered_text, text_features)
            gathered_audio[self.rank] = audio_features
            gathered_text[self.rank] = text_features
            audio_all = torch.cat(gathered_audio, dim=0)
            text_all = torch.cat(gathered_text, dim=0)
            loss = self._loss(audio_all, text_all, logit_scale, logit_bias)
        else:
            loss = self._loss(audio_features, text_features, logit_scale, logit_bias)

        return {"infonce_loss": loss} if output_dict else loss


class GroundingLoss(nn.Module):
    """Grounding Loss for frame-level audio-text alignment.
    
    This loss function computes binary cross-entropy loss between predicted
    frame-level audio-text similarities and ground truth frame labels.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, phrase_embeds_batch, frame_labels_batch, dense_audio_embeds, 
                temp_val=None, b_val=None, device=None, has_grounding=None, output_dict=False):
        """Compute grounding loss for samples with grounding data.
        
        Args:
            phrase_embeds_batch: list of [N_i, D] tensors or None
            frame_labels_batch: list of [N_i, T] tensors or None  
            dense_audio_embeds: [B, T, D] tensor
            temp_val: temperature value (optional)
            device: device for computation
            has_grounding: list of bool indicating which samples have grounding data
            output_dict: whether to return dict or tensor
        """
        if device is None:
            device = dense_audio_embeds.device
            
        grounding_indices = [
            i
            for i, (has_gt, phrase_embeds, labels) 
            in enumerate(zip(has_grounding, phrase_embeds_batch, frame_labels_batch)) 
            if has_gt and phrase_embeds is not None and labels is not None
        ]

        if not grounding_indices:
            loss = torch.tensor(0.0, device=device)
            return {"grounding_loss": loss} if output_dict else loss
        
        grounding_phrase_embeds = [phrase_embeds_batch[i] for i in grounding_indices] 
        grounding_frame_labels = [frame_labels_batch[i] for i in grounding_indices]
        grounding_audio_embeds = dense_audio_embeds[grounding_indices] 
        
        phrase_counts = [emb.shape[0] for emb in grounding_phrase_embeds]
        if sum(phrase_counts) == 0:
            loss = torch.tensor(0.0, device=device)
            return {"grounding_loss": loss} if output_dict else loss
            
        N = phrase_counts[0]
        assert all(c == N for c in phrase_counts), "All grounding samples must have the same number of phrases"

        text_bnD = torch.stack(grounding_phrase_embeds, dim=0)  # [B_grounding, N_i, D]
        audio_bDt = grounding_audio_embeds.transpose(-1, -2) 
        sim_bnt = torch.matmul(text_bnD, audio_bDt)  # [B_grounding, N_i, T]
        
        if temp_val is not None:
            sim_bnt = sim_bnt / temp_val
        if b_val is not None:
            sim_bnt = sim_bnt + b_val
            
        score_bnt = torch.sigmoid(sim_bnt)
        labels_bnt = torch.stack([lbl.to(device).float() for lbl in grounding_frame_labels], dim=0)  # [B_grounding, N_i, T]

        loss = F.binary_cross_entropy(score_bnt, labels_bnt, reduction="mean")
        
        return {"grounding_loss": loss} if output_dict else loss