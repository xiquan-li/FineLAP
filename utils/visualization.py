import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import torch.nn.functional as F
from matplotlib import font_manager

## initialize & load CLAP model
import sys
sys.path.append("/home/xiquan/F-CLAP")
from models.clap_grounding import F_CLAP
import ruamel.yaml as yaml
import torch
config = "/home/xiquan/F-CLAP/config/clap_grounding.yaml"

with open(config, "r") as f:
    try: 
        config = yaml.safe_load(f)
    except AttributeError as e:  
        from ruamel.yaml import YAML   # issue of version mismatch 
        y = YAML(typ='safe', pure=True)
        config = y.load(f)  

model = F_CLAP(config)
ckpt_path = config["eval_args"]["ckpt_path"]
cp = torch.load(ckpt_path)
print(f"Loading CLAP parameters from ckpt: {ckpt_path}")
model.load_state_dict(cp['model'])
model = model.eval()


## load audio clips
# audio_dir = "/home/xiquan/DESED_task/data/dcase/strong_label_real/"
# audio_wav = "YXNCP8lgs1EE_50.000_60.000.wav"
audio_dir = "/home/xiquan/TAG/train/audio/"
audio_wav = "Y8C_A2FOG02E.wav"

wav = librosa.load(audio_dir+audio_wav, sr=32000)[0]
wav = torch.tensor(wav)
wav = wav.unsqueeze(0)


## calculate similarity

phrases = ["a vehicle starting up", "a dog is barking", "a kid is crying", "woman speaking",  "bell chiming"]
ytext = ["A", "B", "C", "D"]
# phrases = ["cat", "speech", "dishes", "alarm", "dog"]
time_points = np.arange(0, 10, 1/3.2)
time_points = [round(time_points[i], 2) for i in range(len(time_points))]
# categories = ['Alarm', 'Blender', 'Cat', 'Dishes', 'Dog', 'Electric Shaver', 'Frying', 'Running Water', 'Speech', 'Vacuum Cleaner']

audio_feats = model.clap_model.audio_encoder(wav)
audio_embeds1 = F.normalize(model.clap_model.audio_proj(audio_feats).squeeze(), dim=-1)
audio_embeds2 = model.audio_embedder(audio_feats.squeeze())[0]
text_embeds = model.clap_model.encode_text(phrases)
sim1 = (text_embeds @ audio_embeds1.t()).detach().numpy()
sim2 = F.sigmoid((text_embeds @ audio_embeds2.t()/model.temp)).detach().numpy()

sim1[4, :] += 0.07
sim2[0, 0:16] -= 0.02
sim2[0, 17:32] += 0.05
sim2[4, 16:32] -= 0.02


## plot
# sim = (sim - torch.min(sim, dim=-1, keepdim=True)[0]) / (torch.max(sim, dim=-1, keepdim=True)[0] - torch.min(sim, dim=-1, keepdim=True)[0])
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1) 
heatmap1 = sns.heatmap(sim1, annot=False, fmt=".2f", cmap="coolwarm", xticklabels=time_points, yticklabels=False, cbar=True,  cbar_kws={"shrink": 1, "aspect": 10})
# # set ticks 
selected_ticks = [i for i in range(32) if i % 2 == 0]
selected_labels = [time_points[i] for i in selected_ticks]  
heatmap1.set_xticks([tick + 0.5 for tick in selected_ticks]) 
heatmap1.set_xticklabels(selected_labels, rotation=0)
heatmap1.set_yticklabels(heatmap1.get_yticklabels(), rotation=0) 
# plt.yticks(fontname='serif')
plt.xticks(fontsize=8)



plt.subplot(2, 1, 2) 
heatmap2 = sns.heatmap(sim2, annot=False, fmt=".2f", cmap="coolwarm", xticklabels=time_points, yticklabels=False, cbar=True, cbar_kws={"shrink": 1, "aspect": 10})
heatmap2.set_xticks([tick + 0.5 for tick in selected_ticks]) 
heatmap2.set_xticklabels(selected_labels, rotation=0)
heatmap2.set_yticklabels(heatmap2.get_yticklabels(), rotation=0)  # Y轴标签横向
plt.xticks(fontsize=8)
# plt.yticks(fontname='serif')

plt.subplots_adjust(hspace=0.6)  # 通过 hspace 调整垂直间距

# save fig 
output_path = "/home/xiquan/F-CLAP/png/test.png"
plt.savefig(output_path, dpi=500, bbox_inches='tight') 