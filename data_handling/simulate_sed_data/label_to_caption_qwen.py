# conda activate base
# 使用Qwen3-30B-A3B-Instruct-2507模型生成caption

import json
import torch
import torch.distributed as dist
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

PROMPT = """
You are an expert in audio captioning. Your task is to generate a caption from frame-level annotations of an audio clip.

The frame-level annotations indicate which types of sounds occur during specific time segments. Based on these annotations, you should produce a fluent and natural-sounding audio caption.

Here are some examples:
Ex. 1:
Phrases: [{"phrase": "Male speech, man speaking", "segments": [[0.0, 2.247], [3.279, 4.1], [4.481, 5.643], [5.911, 10.0]]}, {"phrase": "Mechanisms", "segments": [[0.0, 10.0]]}, {"phrase": "Tap", "segments": [[2.304, 2.589], [2.784, 2.897], [3.052, 3.336]]}, {"phrase": "Breathing", "segments": [[4.132, 4.49], [5.627, 5.936]]}]
Output Caption: A man is speaking near mechanical equipment, with occasional tapping sounds and short breathing pauses.

Ex. 2:
Phrases: [{"phrase": "Generic impact sounds", "segments": [[0.964, 1.135], [3.645, 3.775], [4.311, 4.449], [5.123, 5.237], [6.504, 7.024], [7.268, 7.414], [7.512, 7.641], [8.714, 8.941], [9.128, 9.282], [9.412, 9.575], [9.745, 9.916]]}, {"phrase": "Bark", "segments": [[1.419, 1.793], [4.1, 4.311], [8.356, 8.608]]}, {"phrase": "Dog", "segments": [[2.556, 2.816]]}, {"phrase": "Laughter", "segments": [[2.979, 3.539]]}]
Output Caption: A dog barks several times while people laugh in the background, accompanied by multiple impact sounds.

Please output only the final caption and nothing else.

Now, please generate a caption from the following phrases:
Phrases: __INPUT_DATA__

Output Caption: 
"""

class SEDDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                self.data.append(json.loads(line.strip()))
                if max_samples is not None and idx + 1 >= max_samples:
                    break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

def inference_batch(model, tokenizer, batch, rank):
    texts = []
    for item in batch:
        phrases = item['phrases']
        prompt = PROMPT.replace('__INPUT_DATA__', json.dumps(phrases))
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=16384)
    
    input_lengths = model_inputs.input_ids.shape[1]
    captions = []
    for i in range(len(batch)):
        generated_seq = generated_ids[i, input_lengths:]
        caption = tokenizer.decode(generated_seq, skip_special_tokens=True)
        captions.append(caption)
    
    results = []
    for item, caption in zip(batch, captions):
        item['caption'] = caption.strip()
        results.append(item)
    
    return results

def train(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    dataset = SEDDataset(args.input_jsonl, args.max_samples)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=f"cuda:{rank}"
    )
    
    all_results = []
    iterator = tqdm(dataloader, desc=f"Rank {rank}") if rank == 0 else dataloader
    for batch_idx, batch in enumerate(iterator):
        results = inference_batch(model, tokenizer, batch, rank)
        all_results.extend(results)
    
    output_file = args.output_jsonl.replace('.jsonl', f'_{rank}.jsonl')
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    dist.destroy_process_group()

def merge_outputs(world_size, output_jsonl):
    all_data = []
    for rank in range(world_size):
        temp_file = output_jsonl.replace('.jsonl', f'_{rank}.jsonl')
        with open(temp_file, 'r') as f:
            for line in f:
                all_data.append(json.loads(line.strip()))
        os.remove(temp_file)
    
    with open(output_jsonl, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, 
                        default='/apdcephfs_zwfy2/share_303944931/xiquanli/data/FSD50K/simulated_sed_data_1026/sed_metadata.jsonl')
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None, 
                        help='Comma-separated GPU IDs (e.g., 0,1,2,3). Default uses all GPUs.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    args = parser.parse_args()
    
    if args.gpus is None:
        device_ids = list(range(torch.cuda.device_count()))
    else:
        device_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    world_size = len(device_ids)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
    
    merge_outputs(world_size, args.output_jsonl)
    print(f"Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()


"""
python data_handling/simulate_sed_data/label_to_caption_qwen.py \
    --input_jsonl ../../data/FSD50K/simulated_sed_data_1026/sed_metadata.jsonl \
    --output_jsonl ../../data/FSD50K/simulated_sed_data_1026/sed_metadata_with_captions.jsonl \
    --batch_size 4 \
    --gpus 4,5,6,7
"""
