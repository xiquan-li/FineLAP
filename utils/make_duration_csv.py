import os
from pydub.utils import mediainfo
import pandas as pd
from tqdm import tqdm

audio_dir = "/home/xiquan/data/desed/DESEDpublic_eval/dataset/audio/eval/public"
output_file = "/home/xiquan/data/desed/DESEDpublic_eval/dataset/metadata/eval/duration.csv"
audio_data = []

for file_name in tqdm(os.listdir(audio_dir)):
    if file_name.endswith(".wav"): 
        file_path = os.path.join(audio_dir, file_name)

        info = mediainfo(file_path)
        duration = float(info['duration']) 

        audio_data.append({"audio_id": file_name, "duration": round(duration, 3)})

df = pd.DataFrame(audio_data)
df.to_csv(output_file, index=False, sep="\t") 

print(f"Duration file saved to {output_file}")
