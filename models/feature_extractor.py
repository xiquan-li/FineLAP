import torch
import torch.nn as nn
from torchlibrosa import LogmelFilterBank, Spectrogram
import torchaudio


def EAT_preprocess(waveform, norm_mean=-4.268, norm_std=4.569, target_length=1024): 
    waveform = waveform - waveform.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10
    ).unsqueeze(0)
    
    # Pad or truncate to target length
    n_frames = mel.shape[1]
    if n_frames < target_length:
        mel = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))(mel)
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    mel = (mel - norm_mean) / (norm_std * 2)
    mel = mel.unsqueeze(0)  # shape: [1, 1, T, F]

    return mel


class AudioFeature(nn.Module):
    # for HTSAT
    def __init__(self, audio_config):
        super().__init__()
        self.mel_trans = Spectrogram(n_fft=audio_config["n_fft"],
                                     hop_length=audio_config["hop_length"],
                                     win_length=audio_config["n_fft"],
                                     window='hann',
                                     center=True,
                                     pad_mode='reflect',
                                     freeze_parameters=True)

        self.log_trans = LogmelFilterBank(sr=audio_config["sr"],
                                          n_fft=audio_config["n_fft"],
                                          n_mels=audio_config["n_mels"],
                                          fmin=audio_config["f_min"],
                                          fmax=audio_config["f_max"],
                                          ref=1.0,
                                          amin=1e-10,
                                          top_db=None,
                                          freeze_parameters=True)

    def forward(self, input):
        # input: waveform [bs, wav_length]
        mel_feats = self.mel_trans(input)
        log_mel = self.log_trans(mel_feats)
        return log_mel