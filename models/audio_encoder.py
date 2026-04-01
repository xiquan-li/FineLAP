import torch
import torch.nn as nn
from models.cnns import ResNet38, Cnn14
from models.htsat import HTSAT_Swin_Transformer
from dataclasses import dataclass
from torchlibrosa.augmentation import SpecAugmentation
from utils.logger_config import main_logger


@dataclass
class UserDirModule:
    user_dir: str

class AudioEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.audio_encoder_type = config['audio_encoder_args']['type']  # eat/htsat/cnn ...
        if config["audio_encoder_args"]["type"] == "htsat":
            main_logger.info("Initializing HTSAT ... ")
            self.audio_enc = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=527,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=config,
            )
            main_logger.info("Loading HTSAT parameters from AudioSet pre-trained checkpoint")
            audio_ckpt = torch.load(config['audio_encoder_args']['audio_encoder_ckpt'], map_location="cpu")["state_dict"]  # FIXME: hard code for HTSAT path here
            for key in list(audio_ckpt.keys()):
                if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                    and 'logmel_extractor' not in key):
                    v = audio_ckpt.pop(key)
                    audio_ckpt[key[10:]] = v
            self.audio_enc.load_state_dict(audio_ckpt, strict=False)
            param_names = [n for n, p in self.audio_enc.named_parameters()]
            # for n in param_names:
            #     print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
            self.audio_width = 768

        elif config['audio_encoder_args']['type'] == "eat": 
            main_logger.info("Initializing EAT ... ")
            import fairseq
            model_path = UserDirModule('./models/third_party/EAT') 
            fairseq.utils.import_user_module(model_path)
            ckpt_path = config['audio_encoder_args']['audio_encoder_ckpt']
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            EATEncoder = models[0]
            self.audio_width = 768 
            self.audio_enc = EATEncoder
    
        else:
            raise NotImplementedError('No such audio encoder network.')

    def forward(self, inputs):
        """
        :param inputs: audio raw or mel 
        :return: encoded audio embeddings
        """
        if self.audio_encoder_type == "htsat":
            audio_encoded = self.audio_enc(inputs)['fine_grained_embedding']

        elif self.audio_encoder_type == "eat":   
            audio_mel = inputs  # [b, 1, T, F]
            # spec_augmenter = SpecAugmentation(time_drop_width=64,
            #                                   time_stripes_num=2,
            #                                   freq_drop_width=8,
            #                                   freq_stripes_num=2)
            # if self.training: 
            #     audio_mel = spec_augmenter(audio_mel)
            audio_encoded = self.audio_enc.model.extract_features(audio_mel, 
                                                                  padding_mask=None, 
                                                                  mask=False, 
                                                                  remove_extra_tokens=False)['x']
            audio_cls = audio_encoded[:, 0:1, :]  
            audio_patches = audio_encoded[:, 1:, :]
            B, T, D = audio_patches.shape
            ds_factor = 8
            audio_patches_downsampled = audio_patches.reshape(
                B, T // ds_factor, ds_factor, D
            ).mean(dim=2) 
            audio_encoded = torch.cat([audio_cls, audio_patches_downsampled], dim=1)  # [B, 1+T//8, D]
            
        return audio_encoded 