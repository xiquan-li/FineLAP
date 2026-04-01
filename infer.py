import torch
import torchaudio
from torchaudio.transforms import Resample
import argparse
import json
import ruamel.yaml as yaml
import matplotlib.pyplot as plt
from pathlib import Path
from models.finelap import FineLAP
from models.feature_extractor import EAT_preprocess
import torch.nn.functional as F
import shutil

def load_audio(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0)
    return waveform

@torch.no_grad()
def infer(model, audio_path, caption, phrases, device, output_dir):
    model.eval()

    # Create output directory based on audio filename (without extension)
    audio_stem = Path(audio_path).stem
    output_subdir = Path(output_dir) / audio_stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Copy audio file to output directory
    audio_copy_path = output_subdir / Path(audio_path).name
    shutil.copy2(audio_path, audio_copy_path)
    print(f"Audio copied to: {audio_copy_path}")

    waveform = load_audio(audio_path)
    audio = EAT_preprocess(waveform).to(device)

    global_audio_embeds = model.get_global_audio_embeds(audio)  # [1, D]
    global_text_embeds = model.get_global_text_embeds([caption])  # [1, D]
    dense_audio_embeds = model.get_dense_audio_embeds(audio)  # [1, T, D]
    dense_audio_embeds = dense_audio_embeds.squeeze(0)  # [T, D]

    phrase_embeds = model.get_global_text_embeds(phrases)  # [N, D]

    global_sim = torch.matmul(global_audio_embeds, global_text_embeds.transpose(-1, -2))  # [1, 1]
    sim_matrix = torch.matmul(phrase_embeds, dense_audio_embeds.transpose(-1, -2))  # [N, T]

    temp_global = getattr(model, "temp_global", None)
    b_global = getattr(model, "b_global", None)

    temp_local = getattr(model, "temp_local", None)
    b_local = getattr(model, "b_local", None)

    if temp_global is not None:
        global_sim = global_sim / temp_global
    if b_global is not None:
        global_sim = global_sim + b_global
    global_sim = F.sigmoid(global_sim)

    if temp_local is not None:
        sim_matrix = sim_matrix / temp_local
    if b_local is not None:
        sim_matrix = sim_matrix + b_local

    sim_matrix = F.sigmoid(sim_matrix)
    global_sim_score = global_sim.squeeze().cpu().item()
    sim_matrix_np = sim_matrix.cpu().numpy()

    print(f"Global similarity score: {global_sim_score:.4f}")
    print(f"Similarity matrix shape: {sim_matrix_np.shape} (phrases x time_frames)")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(sim_matrix_np, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Frames', fontsize=12)
    ax.set_ylabel('Phrases', fontsize=12)
    ax.set_title('Frame-level Audio-Phrase Similarity', fontsize=14)
    ax.set_yticks(range(len(phrases)))
    ax.set_yticklabels(phrases)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity Score', rotation=270, labelpad=20)

    output_path = output_subdir / f"similarity_{audio_stem}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Similarity plot saved to: {output_path}")

    result = {
        "audio_path": str(audio_path),
        "caption": caption,
        "phrases": phrases,
        "global_similarity": global_sim_score,
        "frame_similarity_shape": list(sim_matrix_np.shape),
    }

    result_path = output_subdir / f"results_{audio_stem}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {result_path}")

    return global_sim_score, sim_matrix_np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Model config file")
    parser.add_argument("--ckpt_path", required=True, type=str, help="Path to checkpoint file")
    parser.add_argument("--audio_path", required=True, type=str, help="Path to audio file")
    parser.add_argument("--caption", required=True, type=str, help="Caption string for clip-level similarity")
    parser.add_argument("--phrases", nargs="+", required=True, help="Phrase list for frame-level similarity")
    parser.add_argument("--output_dir", default="./output", type=str, help="Output directory")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except AttributeError:
            from ruamel.yaml import YAML
            y = YAML(typ='safe', pure=True)
            config = y.load(f)
    
    device = torch.device(config["training"]["device"])
    
    model = FineLAP(config)
    model = model.to(device)
    
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("Model loaded successfully")
    
    global_similarity, similarity_matrix = infer(
        model,
        args.audio_path,
        args.caption,
        args.phrases,
        device,
        args.output_dir,
    )
    print(f"Caption: {args.caption}")
    print(f"Global similarity: {global_similarity:.4f}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Max similarity: {similarity_matrix.max():.4f}, Min similarity: {similarity_matrix.min():.4f}")

if __name__ == '__main__':
    main()
