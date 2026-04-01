import torch
import argparse
from pathlib import Path


def fix_checkpoint(ckpt_path, out_ckpt_path):
    """Rename all keys containing 'clap_model' by removing 'clap_model.' prefix"""
    print(f"Processing: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Rename clap_model keys
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        new_state_dict = {}
        keys_renamed = []
        
        for key, value in state_dict.items():
            if 'audio_adapter' in key: 
                new_key = key.replace('audio_adapter', 'local_audio_proj')
                new_state_dict[new_key] = value
                keys_renamed.append((key, new_key))
            elif 'clap_model.text_proj' in key:
                new_key = key.replace('clap_model.text_proj', 'global_text_proj')
                new_state_dict[new_key] = value
                keys_renamed.append((key, new_key))
            elif 'clap_model.audio_proj' in key:
                new_key = key.replace('clap_model.audio_proj', 'global_audio_proj')
                new_state_dict[new_key] = value
                keys_renamed.append((key, new_key))
            elif 'clap_model.b' in key:
                new_key = key.replace('clap_model.b', 'b_global')
                new_state_dict[new_key] = value
                keys_renamed.append((key, new_key))
            elif 'clap_model.temp' in key:
                new_key = key.replace('clap_model.temp', 'temp_global')
                new_state_dict[new_key] = value
                keys_renamed.append((key, new_key))
            elif 'clap_model' in key:
                # Remove 'clap_model.' prefix
                new_key = key.replace('clap_model.', '')
                new_state_dict[new_key] = value
                keys_renamed.append((key, new_key))
            else:
                new_state_dict[key] = value
        
        checkpoint['model'] = new_state_dict
        print(f"Renamed {len(keys_renamed)} keys")
        for old_key, new_key in keys_renamed[:5]:  # Show first 5
            print(f"  {old_key} -> {new_key}")
        if len(keys_renamed) > 5:
            print(f"  ... and {len(keys_renamed) - 5} more")
    
    # Save fixed checkpoint
    torch.save(checkpoint, out_ckpt_path)
    # print(new_state_dict)
    print(f"Saved to: {out_ckpt_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, 
                       help="Checkpoint file")
    parser.add_argument("--out_ckpt_path", type=str, required=True, 
                       help="Output checkpoint file")
    
    args = parser.parse_args()
    Path(args.out_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    fix_checkpoint(args.ckpt_path, args.out_ckpt_path)


if __name__ == "__main__":
    main()


"""
python /inspire/hdd/project/embodied-multimodality/public/xqli/ALE/FineLAP/utils/fix_ckpt.py \
    --ckpt_path /inspire/hdd/project/embodied-multimodality/public/xqli/ALE/FineLAP/weights/finelap_AS_WACD_UrbanSED_FSD_gpus_8_bsz128_seed31415926_specaug_eat_transformerAdap_sigmoid/models/last_epoch.pt \
    --out_ckpt_path /inspire/hdd/project/embodied-multimodality/public/xqli/ALE/FineLAP/weights/finelap_fixed.pt
"""
