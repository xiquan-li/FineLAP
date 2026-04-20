# FineLAP Evaluation

## Environmental Setup
Please follow the same environment setup described in `resources/training.md`.

## Evaluation Config
The evaluation dataset config is defined in `config/data_config/data_eat_eval.yaml`.

Its structure is:

```yaml
data_args:
  basic_data_args:
    sample_rate: 16000
    max_length: 10
    time_resolution: 0.15625
    return_type: "mel"

  test_data_args_retrieval_audiocaps:
    metadata_file: ./data/test_metadata_audiocaps_example.jsonl
    dataset_name: "AudioCaps_test"
    task: "retrieval"

  test_data_args_grounding:
    audio_dir: /path/to/audio
    metadata_file: /path/to/test.json
    audio_duration_file: /path/to/test_duration.tsv
    dataset_name: "AudioGrounding"
    task: "grounding"
```

The shared fields in `basic_data_args` are applied to all evaluation datasets:

- `sample_rate`: target sampling rate after resampling
- `max_length`: maximum audio length in seconds
- `time_resolution`: frame resolution used by grounding / SED style evaluation
- `return_type`: `raw` for HTS-AT or `mel` for EAT

Each dataset-specific config usually contains:

- `metadata_file`
- `dataset_name`
- `task`
- `audio_dir` for grounding / SED
- `audio_duration_file` for grounding / SED

## Retrieval Data Setup
Retrieval datasets use `AudioCaptionDataset`.

The metadata format is JSONL. Each line should contain:

```json
{
  "audio_id": "sample.wav",
  "audio_path": "/path/to/audio/sample.wav",
  "caption": [
    "caption 1",
    "caption 2",
    "caption 3",
    "caption 4",
    "caption 5"
  ]
}
```

Each entry contains:

- `audio_id`: unique identifier of the audio sample
- `audio_path`: path to the audio file
- `caption`: a list of captions for the audio

Important: the current retrieval metric implementation in `utils/utils.py` assumes **exactly 5 captions per audio**. For retrieval evaluation, you should keep this format.

Example metadata files:

- AudioCaps: `data/test_metadata_audiocaps_example.jsonl`
- Clotho: `data/test_metadata_clotho_example.jsonl`

Once the JSONL metadata is ready, set it in the corresponding retrieval config block in `config/data_config/data_eat_eval.yaml`.

Example:

```yaml
test_data_args_retrieval_audiocaps:
  metadata_file: ./data/test_metadata_audiocaps_example.jsonl
  dataset_name: "AudioCaps_test"
  task: "retrieval"
```

## Classification Data Setup
Classification datasets also use `AudioCaptionDataset`, but here `caption` is the class label rather than a free-form sentence.

The metadata format is JSONL:

```json
{
  "audio_id": "sample.wav",
  "audio_path": "/path/to/audio/sample.wav",
  "caption": "dog_bark"
}
```

Each entry contains:

- `audio_id`: unique identifier of the audio sample
- `audio_path`: path to the audio file
- `caption`: class label string

The current classification code uses the template:

```text
The sound of {}
```

so labels such as `dog_bark` will be converted to `The sound of dog bark` internally.

Example metadata files:

- ESC-50: `data/test_metadata_esc50_example.jsonl`
- UrbanSound8K: `data/test_metadata_urbansound8k_example.jsonl`
- VGGSound: `data/test_metadata_vggsound_example.jsonl`

Example config:

```yaml
test_data_args_esc50:
  dataset_name: "ESC50"
  metadata_file: ./data/test_metadata_esc50_example.jsonl
  task: "classification"
```

## SED Data Setup
SED datasets use `SoundEventDataset`.

You need:

- an audio directory
- an event metadata TSV file
- an audio duration TSV file

The event metadata format is:

```tsv
filename	onset	offset	event_label
clip_001.wav	0.30	1.10	dog_bark
clip_001.wav	2.50	3.00	car_horn
clip_002.wav	4.00	6.00	siren
```

Each row contains:

- `filename`: audio file name
- `onset`: event start time in seconds
- `offset`: event end time in seconds
- `event_label`: sound event category

The duration metadata format is:

```tsv
filename	duration
clip_001.wav	10.0
clip_002.wav	10.0
```

Each row contains:

- `filename`: audio file name
- `duration`: total audio duration in seconds

Example metadata files:

- DESED events: `data/test_metadata_desed_example.tsv`
- DESED durations: `data/test_metadata_desed_duration_example.tsv`
- AudioSet Strong events: `data/test_metadata_audioset_strong_example.tsv`
- AudioSet Strong durations: `data/test_metadata_audioset_strong_duration_example.tsv`
- UrbanSED events: `data/test_metadata_urbansed_example.tsv`
- UrbanSED durations: `data/test_metadata_urbansed_duration_example.tsv`

Example config:

```yaml
test_data_args_sed:
  audio_dir: /path/to/your/sed_audio
  metadata_file: ./data/test_metadata_desed_example.tsv
  audio_duration_file: ./data/test_metadata_desed_duration_example.tsv
  dataset_name: "DESED"
  task: "sed"
```

## Grounding Data Setup
Grounding datasets use `AudioGroundingDataset`.

The metadata format is a single JSON file containing a list of audio samples:

```json
[
  {
    "audiocap_id": 1,
    "audio_id": "clip_001.wav",
    "tokens": "a dog barks and a car honks",
    "phrases": [
      {
        "phrase": "a dog barks",
        "start_index": 0,
        "end_index": 3,
        "segments": [[0.4, 1.2], [2.0, 2.3]]
      },
      {
        "phrase": "a car honks",
        "start_index": 4,
        "end_index": 7,
        "segments": [[5.1, 5.5]]
      }
    ]
  }
]
```

Each audio item contains:

- `audiocap_id`: unique sample identifier
- `audio_id`: audio file name
- `tokens`: full caption
- `phrases`: phrase-level grounding annotations

Each phrase item contains:

- `phrase`: phrase text
- `start_index`: phrase start token index
- `end_index`: phrase end token index
- `segments`: one or more `[start, end]` time intervals in seconds

Grounding evaluation also requires a duration TSV file:

```tsv
filename	duration
clip_001.wav	10.0
```

Example metadata files:

- TAG / grounding metadata: `data/test_metadata_tag_example.json`
- TAG / grounding durations: `data/test_metadata_tag_duration_example.tsv`

Example config:

```yaml
test_data_args_grounding:
  audio_dir: /path/to/your/grounding_audio
  metadata_file: ./data/test_metadata_tag_example.json
  audio_duration_file: ./data/test_metadata_tag_duration_example.tsv
  dataset_name: "AudioGrounding"
  task: "grounding"
```

## Start Evaluation
Run:

```bash
python evaluate.py \
  --exp_dir exps/eval_run \
  --config config/finelap_eat_config.yaml \
  --data_config config/data_config/data_eat_eval.yaml \
  --ckpt_path /path/to/your/checkpoint.pt \
  --task all
```

Supported tasks are:

- `grounding`
- `sed`
- `retrieval`
- `classification`
- `all`

For example, to evaluate only SED:

```bash
python evaluate.py \
  --exp_dir exps/eval_sed \
  --config config/finelap_eat_config.yaml \
  --data_config config/data_config/data_eat_eval.yaml \
  --ckpt_path /path/to/your/checkpoint.pt \
  --task sed
```

Evaluation logs will be saved under `<exp_dir>/evaluation_logs/`.
