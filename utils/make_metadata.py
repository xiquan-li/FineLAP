import json
import os
import csv
import pandas as pd
from tqdm import tqdm

input_file = '/home/xiquan/DESED_task/data/dcase/metadata/train/audioset_strong.tsv'
output_file = "/home/xiquan/F-CLAP/data/DESED/train_with_negative.json"
audio_dir = "/home/xiquan/DESED_task/data/dcase/strong_label_real"

filename_prev = ""
audiocap_id = 0
data = []
tmp_audio_item = {}

classes = ['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', 'Electric_shaver_toothbrush', 'Frying', 'Running_water', 'Speech', 'Vacuum_cleaner']


with open(input_file, 'r') as f: 
    lines = f.readlines()
    for line in tqdm(lines[1:], total=len(lines)-1):
        line = line.strip()
        filename, onset, offset, phrase = line.split('\t')  
        if not os.path.exists(os.path.join(audio_dir, filename)): 
            continue

        onset, offset = float(onset), float(offset)

        if filename_prev != filename: 
            # change of file, new audio_item,
            if filename_prev != "":
                for phrase in classes: 
                    if phrase not in tmp_phrase_list:
                        # add negative phrase sample
                        tmp_phrase_item = {"phrase": phrase, "segments": [[0, 0]]}
                        tmp_audio_item["phrases"].append(tmp_phrase_item)
                data.append(tmp_audio_item)  # save existing audio_item

            tmp_audio_item = {
                "audiocap_id": audiocap_id, "audio_id": filename, "phrases": []
            }
            audiocap_id += 1
            filename_prev = filename
            tmp_phrase_list = set()

        if phrase not in tmp_phrase_list: 
            # create new phrase item
            tmp_phrase_item = {
                "phrase": phrase,
                "segments": [[onset, offset]]
            }
            tmp_phrase_list.add(phrase)
            tmp_audio_item["phrases"].append(tmp_phrase_item)

        else:
            # add new phrase segment to existing phrase item    
            for phrase_item in tmp_audio_item["phrases"]: 
                if phrase == phrase_item["phrase"]: 
                    phrase_item["segments"].append([onset, offset])

with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Json file saved in {output_file}")