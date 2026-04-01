# FineLAP Training & Fine-tuning

Before training, make sure that all files from [here](https://huggingface.co/AndreasXi/FineLAP_Pytorch) have been downloaded to `./weights/`. 

## Environmental Setup
```bash
conda create -n finelap python=3.9 

git clone https://github.com/facebookresearch/fairseq.git
pip install "pip<24.1" -U; cd fairseq; pip install -e ./

pip install -r requirements_train.txt
```

## Data Setup
To train FineLAP, we format the data in a JSONL structure as follows:

```json
{
  "audio_id": "Ycq6bqC_AsO4.flac",
  "audio_path": "path/to/audio.wav",
  "caption": "Birds are chirping with background noise.",
  "phrases": [
    {
      "phrase": "Background noise",
      "segments": [
        [0.498, 10.0]
      ]
    },
    {
      "phrase": "Bird vocalization, bird call, bird song",
      "segments": [
        [0.629, 4.114],
        [4.313, 10.0]
      ]
    }
  ]
}
```

Each entry contains:

- audio_id: Unique identifier of the audio sample.
- audio_path: Path to the audio file.
- caption: A clip-level description of the audio content.
- phrases (optional): A list of sound events, where each includes:
  - phrase: Textual phrase of the event
  - segments: Time intervals (in seconds) indicating when the event occurs

For data without frame-level annotations, the `phrases` field can be omitted. The dataset will automatically detect this and skip the frame-level loss for such samples.
An example training metadata file with 10 samples is provided at `data/training_metadata_example.jsonl`.

The current training pipeline uses the phrase bank `data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl`.

Once the dataset metadata JSONL is ready, include it in the `train_data_args.metadata_files` list defined in `config/data_config/data_eat.yaml` or `config/data_config/data_htsat.yaml`.

## Start Training
Run
```bash
bash scripts/train.sh
```
to start training. This will use the config `config/finelap_eat_config.yaml`. The output will be saved in `exps/${exp_name}`. 

## Fine-tuning From a FineLAP Checkpoint
The training code now supports loading an existing FineLAP checkpoint before training starts. This is useful when you want to finetune from a previously trained model such as `weights/finelap_fixed.pt`.

In `config/finelap_eat_config.yaml`, set:

```yaml
model_args:
  ckpt_path: './weights/finelap_fixed.pt'
```

If `ckpt_path` is an empty string:

```yaml
model_args:
  ckpt_path: ''
```

then no FineLAP checkpoint will be loaded, and training will start from the encoder initialization defined by `audio_encoder_ckpt` and `text_encoder_ckpt`.

This finetuning path loads model weights only. It does not restore the optimizer state or resume the previous epoch count.
