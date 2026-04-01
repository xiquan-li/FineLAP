# conda activate base
# Rule-based caption generation

import json
import argparse
from tqdm import tqdm

def generate_caption(phrases):
    phrase_names = [item['phrase'] for item in phrases]
    caption = ", ".join(phrase_names)
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str,
                        default='/apdcephfs_zwfy2/share_303944931/xiquanli/data/FSD50K/simulated_sed_data_1026/sed_metadata.jsonl')
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    args = parser.parse_args()
    
    output_data = []
    
    with open(args.input_jsonl, 'r') as f:
        lines = f.readlines()
        if args.max_samples is not None:
            lines = lines[:args.max_samples]
        
        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line.strip())
            caption = generate_caption(data['phrases'])
            data['caption'] = caption
            output_data.append(data)
    
    with open(args.output_jsonl, 'w') as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()


"""
python data_handling/simulate_sed_data/label_to_caption_rulebased.py \
    --input_jsonl ../../data/FSD50K/simulated_sed_data_1026/sed_metadata.jsonl \
    --output_jsonl ../../data/FSD50K/simulated_sed_data_1026/sed_metadata_wcaption_rulebased.jsonl
"""
