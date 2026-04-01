import torch
from transformers import AutoModel

audio_path = ['resources/1.wav', 'resources/2.wav']  # (B,)
caption = ["A woman speaks, dishes clanking, food frying, and music plays", 'A power tool is heard with male speech.']  # (B,)
phrases = ['Speech', 'Dog', 'Cat', 'Frying', 'Dishes', 'Music', 'Vacuum', 'Type', 'Power tool']  # (N,)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(device)
model.eval()

with torch.no_grad():
    global_text_embeds = model.get_global_text_embeds(caption)  # (B, d)
    print(global_text_embeds.shape)

    global_audio_embeds = model.get_global_audio_embeds(audio_path)   # (B, d)
    print(global_audio_embeds.shape)

    dense_audio_embeds = model.get_dense_audio_embeds(audio_path)  # (B, T, d)
    print(dense_audio_embeds.shape)

    clip_scores = model.get_clip_level_score(audio_path, caption)  # (B, B)
    print(clip_scores.shape)

    frame_scores = model.get_frame_level_score(audio_path, phrases)  # (B, N, T)
    print(frame_scores.shape)
    
    ## Only supprt single audio
    model.plot_frame_level_score(audio_path[1], phrases, output_path="output/output_plot.png")
    