import warnings
warnings.filterwarnings("ignore")
import time
from pprint import PrettyPrinter
import torch
import argparse
import ruamel.yaml as yaml
from tqdm import tqdm
from utils.logger_config import setup_logger
from data_handling.grounding_dataset import AudioGroundingDataset
from data_handling.sed_dataset import SoundEventDataset
from data_handling.caption_dataset import AudioCaptionDataset
from torch.utils.data import DataLoader
from utils.utils import (
    init_distributed_mode,
    is_main_process,
    AverageMeter, t2a, a2t, log_results
)
import pandas as pd
import numpy as np
import sed_scores_eval
import torch.nn.functional as F
from utils.eval_util import calculate_psds_from_score, median_filter, find_contiguous_regions, connect_clusters, predictions_to_time, generate_ground_truth_df, compute_th_auc
from utils.eval_utils_sed import post_process, compute_psds_from_scores, compute_collar_f1, compute_seg_f1
import math
from pathlib import Path
import wandb


@torch.no_grad()
def validate_sed(model, dataloader, device, only_psds1=False, variance_penalty=True):
    model.eval()
    
    # Get base model (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model

    ground_truth = sed_scores_eval.io.read_ground_truth_events(dataloader.dataset.metadata_file)
    audio_durations = sed_scores_eval.io.read_audio_durations(dataloader.dataset.audio_duration_file)   
    ground_truth = {
        audio_id: gt for audio_id, gt in ground_truth.items()
        if len(gt) > 0
    }
    audio_durations = {
        audio_id: audio_durations[audio_id]
        for audio_id in ground_truth.keys()
    }

    classes = dataloader.dataset.classes
    scores_raw_dic, scores_postprocessed_dic = {}, {}
    temp_local_val = getattr(base_model, "temp_local", None)
    b_local_val = getattr(base_model, "b_local", None)
    text_embeds = base_model.get_global_text_embeds(classes)  # [N_cls, D]
    
    if is_main_process():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    
    for batch_idx, batch in pbar:
        audio = batch["audios"].to(device)
        audio_ids = batch["audio_id"]
        batch_size = audio.size(0)
        dense_audio_embeds = base_model.get_dense_audio_embeds(audio)  # [B, T, D]
        T, D = dense_audio_embeds.shape[1], dense_audio_embeds.shape[2]
        audio_embeds_flat = dense_audio_embeds.reshape(batch_size * T, D)  # [B*T, D]
        sim_cls2frame_flat = torch.matmul(text_embeds, audio_embeds_flat.transpose(-1, -2))  # [N_cls, D] @ [B*T, D].T -> [N_cls, B*T]
        
        if temp_local_val is not None:
            sim_cls2frame_flat = sim_cls2frame_flat / temp_local_val
        if b_local_val is not None:
            sim_cls2frame_flat = sim_cls2frame_flat + b_local_val
            
        sim_cls2frame_batch = sim_cls2frame_flat.reshape(len(classes), batch_size, T)  # [N_cls, B, T]
        score_cls2frame_batch = F.sigmoid(sim_cls2frame_batch)  # [N_cls, B, T]
        
        score_frame2cls_batch = score_cls2frame_batch.transpose(0, 1).transpose(1, 2)  # [B, T, N_cls]
        
        for i in range(batch_size):
            score_frame2cls = score_frame2cls_batch[i]  # [T, N_cls]
            audio_id = audio_ids[i]
            scores_raw, scores_postprocessed = post_process(score_frame2cls, dataloader)
            if '.wav' in audio_id:
                scores_raw_dic[audio_id.split(".wav")[0]] = scores_raw
                scores_postprocessed_dic[audio_id.split(".wav")[0]] = scores_postprocessed
            elif '.flac' in audio_id:
                scores_raw_dic[audio_id.split(".flac")[0]] = scores_raw
                scores_postprocessed_dic[audio_id.split(".flac")[0]] = scores_postprocessed
            else:
                scores_raw_dic[audio_id] = scores_raw
                scores_postprocessed_dic[audio_id] = scores_postprocessed

    pop_lst = []  # pop keys not in ground truth csv
    for k in scores_postprocessed_dic.keys():
        if k not in ground_truth.keys():
            pop_lst.append(k)
    for k in pop_lst:       
        scores_postprocessed_dic.pop(k)

    if is_main_process():
        print("Computing PSDS1...")
    start_time = time.time()
    if variance_penalty: 
        alpha_st = 1
    else:
        alpha_st = 0

    psds1 = compute_psds_from_scores(
        scores_postprocessed_dic,
        ground_truth,
        audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0,
        alpha_st=alpha_st,
    )
    psds1_time = time.time() - start_time
    if is_main_process():
        print(f"PSDS1 computation completed in {psds1_time:.2f} seconds")
    
    if not only_psds1:
        if is_main_process():
            print("Computing PSDS2...")
        start_time = time.time()
        psds2 = compute_psds_from_scores(
            scores_postprocessed_dic,
            ground_truth,
            audio_durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )
        psds2_time = time.time() - start_time
        if is_main_process():
            print(f"PSDS2 computation completed in {psds2_time:.2f} seconds")
    
        if is_main_process():
            print("Computing F1 scores...")
        start_time = time.time()
        eb_f1 = compute_collar_f1(scores_postprocessed_dic, ground_truth)
        seg_f1 = compute_seg_f1(scores_postprocessed_dic, ground_truth, audio_durations)
        f1_time = time.time() - start_time
        if is_main_process():
            print(f"F1 scores computation completed in {f1_time:.2f} seconds")
    else: 
        psds2 = None
        eb_f1 = None
        seg_f1 = None

    return psds1, psds2, eb_f1, seg_f1


@torch.no_grad()
def validate_grounding(model, dataloader, device, save_dir):
    model.eval()
    
    base_model = model.module if hasattr(model, 'module') else model
    
    gt_dict = {}
    fname_to_aid = {}
    score_buffer = {}
    event_classes = ["fake_event"]
    audio_duration_file = dataloader.dataset.audio_duration_file

    time_resolution = dataloader.dataset.time_resolution
    n_thresholds = 50
    n_connect = math.ceil(0.5 / time_resolution)
    thresholds = np.arange(
        1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    psds_buffer = {th: [] for th in thresholds}


    temp_local_val = getattr(base_model, "temp_local", None)
    b_local_val = getattr(base_model, "b_local", None)
    
    # Only show tqdm progress bar on main process
    if is_main_process():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    
    for batch_idx, batch in pbar:
        audio = batch["audios"].to(device)
        phrase = batch["phrase"]
        audiocap_id, start_index, phrase_segments, audio_id = batch["audiocap_id"], batch["start_index"], batch["phrase_segments"], batch["audio_id"]
        phrase = list(phrase)
        
        # Use new frame-level similarity computation
        dense_audio_embeds = base_model.get_dense_audio_embeds(audio)  # [B, T, D]
        text_embeds = base_model.get_global_text_embeds(phrase)        # [B, D]
        
        # Compute frame-level similarities
        sim_cls2frame = torch.matmul(text_embeds.unsqueeze(1), dense_audio_embeds.transpose(-1, -2)).squeeze(1)  # [B, T]
        
        if temp_local_val is not None:
            sim_cls2frame = sim_cls2frame / temp_local_val
        if b_local_val is not None:
            sim_cls2frame = sim_cls2frame + b_local_val

        score_cls2frame = F.sigmoid(sim_cls2frame)

        for idx in range(audio.size(0)):  # idx in batch_size
            ## create ground truth dict
            tmp_audiocap_id = audiocap_id[idx]
            tmp_start_index = start_index[idx]
            fname = f"{tmp_audiocap_id}_{tmp_start_index}"
            gt_dict[fname] = []  # get the t_start & t_end of the current phrase
            for t_start, t_end in phrase_segments[idx]:
                t_start, t_end = t_start, t_end
                if t_start == 0 and t_end == 0:
                    continue
                gt_dict[fname].append((
                    t_start,
                    t_end,
                    "fake_event"
                ))
            fname_to_aid[fname] = audio_id[idx]

            ## evaluate the scores obtained by model
            scores_arr = score_cls2frame[idx]
            if scores_arr.ndim == 1:
                scores_arr = scores_arr.unsqueeze(-1)
            scores_arr = scores_arr.cpu().numpy()  # (1, t/t_sr)
            timestamps = np.arange(score_cls2frame.shape[1] + 1) * time_resolution  # (1, t/t_sr)
            score_buffer[fname] = sed_scores_eval.utils.create_score_dataframe(
                scores_arr, timestamps=timestamps,  
                event_classes=event_classes)
            # th-auc
            for th in thresholds:
                filtered_prob = median_filter(
                    score_cls2frame[idx].unsqueeze(0).cpu(),
                    window_size=3,
                    threshold=th
                )[0]   # [T, btz]

                change_indices = find_contiguous_regions(
                    connect_clusters(
                        filtered_prob,
                        n_connect
                    )
                )             
                for row in change_indices:
                    psds_buffer[th].append({
                        "filename": fname,
                        "event_label": "fake_event",
                        "onset": row[0],
                        "offset": row[1]
                    })
    # psds
    psds1, psds2, psds2021 = calculate_psds_from_score(score_buffer, gt_dict, audio_duration_file, fname_to_aid, save_dir)

    # th-auc
    for th in thresholds:
        if len(psds_buffer[th]) > 0:
            pred_df = pd.DataFrame(psds_buffer[th])
        else:
            pred_df = pd.DataFrame({
                "filename": [],
                "event_label": [],
                "onset": [],
                "offset": []
            })
        pred_df = predictions_to_time(pred_df, ratio=time_resolution)
        psds_buffer[th] = pred_df
        ground_truth = generate_ground_truth_df(dataloader)

    min_th, max_th = 0,1
    th_auc = compute_th_auc(
            psds_buffer,
            ground_truth.drop(["event_label", "audio_id"], axis=1),
            dtc_threshold=0.5,
            gtc_threshold=0.5,
            min_threshold=min_th,
            max_threshold=max_th,
            save_dir=save_dir
        )
    return psds1, psds2, psds2021, th_auc
    

@torch.no_grad()
def validate_retrieval(model, dataloader, device, logger, use_wandb):
    model.eval()
    
    # Get base model (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model
    
    audio_embeds_all, text_embeds_all = [], []
    
    # Only show tqdm progress bar on main process
    if is_main_process():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    
    for batch_idx, batch in pbar:
        audio = batch["audios"].to(device)
        captions = batch["caption"]
        
        # Get global audio and text embeddings
        audio_embeds = base_model.get_global_audio_embeds(audio)  # [B, D]
        text_embeds = base_model.get_global_text_embeds(captions)  # [B, D]
            
        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)

    # logging
    dataset_name = dataloader.dataset.dataset_name
    if is_main_process():
        logger.info(f"Retrieval results for dataset: {dataset_name}")
        logger.info("Text-to-Audio: r1: {:.2f}, r5: {:.2f}, r10: {:.2f}, r50: {:.2f}, medr: {:.1f}, meanr: {:.1f}, mAP10: {:.3f}".format(
                r1, r5, r10, r50, medr, meanr, mAP10))
        logger.info("Audio-to-Text: r1: {:.2f}, r5: {:.2f}, r10: {:.2f}, r50: {:.2f}, medr: {:.1f}, meanr: {:.1f}, mAP10: {:.3f}".format(
                r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a))
    if use_wandb and is_main_process(): 
        wandb.log({
            f"{dataset_name}/t2a_r1": r1,
            f"{dataset_name}/t2a_r5": r5,
            f"{dataset_name}/t2a_r10": r10,
            f"{dataset_name}/t2a_mAP10": mAP10,
            f"{dataset_name}/a2t_r1": r1_a,
            f"{dataset_name}/a2t_r5": r5_a,
            f"{dataset_name}/a2t_r10": r10_a,
            f"{dataset_name}/a2t_mAP10": mAP10_a,
    })
    

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}

@torch.no_grad()
def validate_classification(model, dataloader, device, logger, use_wandb, template="The sound of {}"):
    model.eval()
    
    base_model = model.module if hasattr(model, 'module') else model
    
    all_audio_embeds = []
    all_audio_ids = []
    all_true_labels = []
    
    if is_main_process():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    
    for batch_idx, batch in pbar:
        audio = batch["audios"].to(device)
        audio_ids = batch["audio_id"]
        captions = batch["caption"]
        
        audio_embeds = base_model.get_global_audio_embeds(audio)
        
        all_audio_embeds.append(audio_embeds.cpu())
        all_audio_ids.extend(audio_ids)
        all_true_labels.extend(captions)
    
    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)
    
    unique_labels = sorted(list(set(all_true_labels)))
    label_texts = [template.format(label.replace("_", " ")) for label in unique_labels]
    
    label_embeds = base_model.get_global_text_embeds(label_texts)
    label_embeds = label_embeds.cpu()
    
    similarities = torch.matmul(all_audio_embeds, label_embeds.transpose(-1, -2))
    
    pred_indices = torch.argmax(similarities, dim=1)
    pred_labels = [unique_labels[idx] for idx in pred_indices.tolist()]
    
    correct = sum([pred == true for pred, true in zip(pred_labels, all_true_labels)])
    accuracy = correct / len(all_true_labels) * 100.0
    
    dataset_name = dataloader.dataset.dataset_name
    if is_main_process():
        logger.info(f"Classification results for dataset: {dataset_name}")
        logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{len(all_true_labels)})")
        logger.info(f"Number of classes: {len(unique_labels)}")
    
    if use_wandb and is_main_process():
        wandb.log({
            f"{dataset_name}/classification_acc": accuracy,
        })
    
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="test", type=str, help="Experiment directory")
    parser.add_argument("--config", required=True, type=str, help="Model config file")
    parser.add_argument("--data_config", required=True, type=str, help="Evaluation dataset config file")
    parser.add_argument("--ckpt_path", default="test", type=str, help="Path to checkpoint file")
    parser.add_argument("--task", default="all", type=str, choices=["grounding", "sed", "retrieval", "classification", "all"], 
                        help="Evaluation task")
    
    args = parser.parse_args()

    # Load model config
    with open(args.config, "r") as f:
        try: 
            config = yaml.safe_load(f)
        except AttributeError as e:  
            from ruamel.yaml import YAML   # issue of version mismatch 
            y = YAML(typ='safe', pure=True)
            config = y.load(f)  
    
    # Load eval config for dataset configuration
    with open(args.data_config, "r") as f:
        try: 
            data_config = yaml.safe_load(f)
        except AttributeError as e:  
            from ruamel.yaml import YAML   # issue of version mismatch 
            y = YAML(typ='safe', pure=True)
            data_config = y.load(f)  

    # Initialize distributed mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["training"]["device"])

    # Setup logging
    log_output_dir = Path(args.exp_dir, "evaluation_logs")
    log_output_dir.mkdir(parents=True, exist_ok=True)
    main_logger = setup_logger(args.exp_dir, "evaluation", log_output_dir.joinpath('evaluation_log.txt'))
    printer = PrettyPrinter()
    
    if is_main_process():
        main_logger.info('Evaluation configuration:\n'
                         f'{printer.pformat(config)}'
                         f'{printer.pformat(data_config)}')

    # Initialize model
    from models.finelap import FineLAP
    model = FineLAP(config)
    model = model.to(device)

    # Load checkpoint
    if is_main_process():
        main_logger.info(f"Loading checkpoint from: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    
    renamed_state_dict = {}
    for k, v in checkpoint["model"].items():
        if "audio_embedder" in k:
            new_key = k.replace("audio_embedder", "audio_adapter")
            renamed_state_dict[new_key] = v
            if is_main_process():
                main_logger.info(f"Renamed: {k} -> {new_key}")
        else:
            renamed_state_dict[k] = v
    
    model.load_state_dict(renamed_state_dict)
    if is_main_process():
        main_logger.info(f"Checkpoint loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}")
    model.eval()

    # Prepare datasets using data_config
    eval_data_args = data_config["data_args"]
    
    if args.task in ["grounding", "all"]:
        test_dataloader_grounding = None
        if "test_data_args_grounding" in eval_data_args:
            test_dataset_grounding = AudioGroundingDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_grounding"])
            test_dataloader_grounding = DataLoader(dataset=test_dataset_grounding,
                                                collate_fn=test_dataset_grounding.collate_fn,
                                                shuffle=False,
                                                batch_size=16,
                                                num_workers=16,
                                                drop_last=False)
    
    if args.task in ["sed", "all"]:
        test_dataloader_desed = None
        if "test_data_args_sed" in eval_data_args:
            test_dataset_desed = SoundEventDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_sed"])
            test_dataloader_desed = DataLoader(dataset=test_dataset_desed, 
                                            batch_size=16, 
                                            num_workers=4, 
                                            shuffle=False, 
                                            collate_fn=test_dataset_desed.collate_fn, 
                                            drop_last=False)

        test_dataloader_audioset_strong = None
        if "test_data_args_audioset_strong" in eval_data_args:
            test_dataset_audioset_strong = SoundEventDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_audioset_strong"])
            test_dataloader_audioset_strong = DataLoader(dataset=test_dataset_audioset_strong, 
                                                        batch_size=16, 
                                                        num_workers=4, 
                                                        shuffle=False, 
                                                        collate_fn=test_dataset_audioset_strong.collate_fn, 
                                                        drop_last=False)

        test_dataloader_urbansed = None
        if "test_data_args_urbansed" in eval_data_args:
            test_dataset_urbansed = SoundEventDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_urbansed"])
            test_dataloader_urbansed = DataLoader(dataset=test_dataset_urbansed, 
                                                    batch_size=16, 
                                                    num_workers=4, 
                                                    shuffle=False, 
                                                    collate_fn=test_dataset_urbansed.collate_fn, 
                                                    drop_last=False)
    
    if args.task in ["retrieval", "all"]:
        test_dataloader_retrieval_audiocaps = None
        test_dataloader_retrieval_clotho = None
        
        if "test_data_args_retrieval_audiocaps" in eval_data_args:
            test_dataset_retrieval_audiocaps = AudioCaptionDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_retrieval_audiocaps"])
            test_dataloader_retrieval_audiocaps = DataLoader(dataset=test_dataset_retrieval_audiocaps, 
                                                batch_size=16,
                                                num_workers=16, 
                                                shuffle=False, 
                                                collate_fn=test_dataset_retrieval_audiocaps.collate_fn, 
                                                drop_last=False)
        
        if "test_data_args_retrieval_clotho" in eval_data_args:
            test_dataset_retrieval_clotho = AudioCaptionDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_retrieval_clotho"])
            test_dataloader_retrieval_clotho = DataLoader(dataset=test_dataset_retrieval_clotho, 
                                                batch_size=16,
                                                num_workers=16, 
                                                shuffle=False, 
                                                collate_fn=test_dataset_retrieval_clotho.collate_fn, 
                                                drop_last=False)
    
    if args.task in ["classification", "all"]:
        test_dataloader_esc50 = None
        if "test_data_args_esc50" in eval_data_args:
            test_dataset_esc50 = AudioCaptionDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_esc50"])
            test_dataloader_esc50 = DataLoader(dataset=test_dataset_esc50, 
                                                    batch_size=16,
                                                    num_workers=16, 
                                                    shuffle=False, 
                                                    collate_fn=test_dataset_esc50.collate_fn, 
                                                    drop_last=False)
        
        test_dataloader_urbansound8k = None
        if "test_data_args_urbansound8k" in eval_data_args:
            test_dataset_urbansound8k = AudioCaptionDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_urbansound8k"])
            test_dataloader_urbansound8k = DataLoader(dataset=test_dataset_urbansound8k, 
                                                    batch_size=8,
                                                    num_workers=16, 
                                                    shuffle=False, 
                                                    collate_fn=test_dataset_urbansound8k.collate_fn, 
                                                    drop_last=False)
        
        test_dataloader_vggsound = None
        if "test_data_args_vggsound" in eval_data_args:
            test_dataset_vggsound = AudioCaptionDataset(**eval_data_args["basic_data_args"], **eval_data_args["test_data_args_vggsound"])
            test_dataloader_vggsound = DataLoader(dataset=test_dataset_vggsound, 
                                                    batch_size=16,
                                                    num_workers=16, 
                                                    shuffle=False, 
                                                    collate_fn=test_dataset_vggsound.collate_fn, 
                                                    drop_last=False)

    if is_main_process():
        main_logger.info("Starting evaluation...")

    # Evaluation based on task
    if args.task in ["grounding", "all"]:
        if test_dataloader_grounding is not None:
            if is_main_process():
                main_logger.info('Evaluating Audio Grounding...')
            save_dir = log_output_dir / "psds_evaluation"
            save_dir.mkdir(parents=True, exist_ok=True)
            psds1, psds2, psds2021, th_auc = validate_grounding(model, test_dataloader_grounding, device, save_dir)
            if is_main_process():
                main_logger.info("AudioGrounding Results: psds1: {:.4f}, psds2: {:.4f}, psds2021: {:.4f}, th_auc: {:.4f}".format(psds1, psds2, psds2021, th_auc))
        
    if args.task in ["sed", "all"]:
        if test_dataloader_urbansed is not None or test_dataloader_desed is not None or test_dataloader_audioset_strong is not None:
            if is_main_process():
                main_logger.info('Evaluating Sound Event Detection...')

        if test_dataloader_desed is not None:
            psds1, psds2, eb_f1, seg_f1 = validate_sed(model, test_dataloader_desed, device, only_psds1=False)
            if is_main_process():
                main_logger.info("DESED Results: psds1: {:.4f}, psds2: {:.4f}".format(psds1, psds2))
        
        if test_dataloader_audioset_strong is not None:
            psds1, psds2, eb_f1, seg_f1 = validate_sed(model, test_dataloader_audioset_strong, device, only_psds1=True, variance_penalty=False)
            if is_main_process():
                main_logger.info("AudioSet Strong Eval Results: psds1: {:.4f}".format(psds1))
                
        if test_dataloader_urbansed is not None:
            psds1, psds2, eb_f1, seg_f1 = validate_sed(model, test_dataloader_urbansed, device, only_psds1=False)
            if is_main_process():
                main_logger.info("UrbanSED Results: psds1: {:.4f}, psds2: {:.4f}".format(psds1, psds2))

    if args.task in ["retrieval", "all"]:
        if is_main_process():
            main_logger.info('Evaluating Audio Retrieval...')
        
        if test_dataloader_retrieval_audiocaps is not None:
            retrieval_results = validate_retrieval(model, test_dataloader_retrieval_audiocaps, device, main_logger, False)
            
            if is_main_process():
                # Extract text-to-audio results
                t2a_r1, t2a_r5, t2a_r10, t2a_r50, t2a_medr, t2a_meanr, t2a_mAP10 = retrieval_results["t2a"]
                main_logger.info("AudioCaps - Text-to-Audio: r1: {:.2f}, r5: {:.2f}, r10: {:.2f}, r50: {:.2f}, medr: {:.1f}, meanr: {:.1f}, mAP10: {:.3f}".format(
                    t2a_r1, t2a_r5, t2a_r10, t2a_r50, t2a_medr, t2a_meanr, t2a_mAP10))
                
                # Extract audio-to-text results
                a2t_r1, a2t_r5, a2t_r10, a2t_r50, a2t_medr, a2t_meanr, a2t_mAP10 = retrieval_results["a2t"]
                main_logger.info("AudioCaps - Audio-to-Text: r1: {:.2f}, r5: {:.2f}, r10: {:.2f}, r50: {:.2f}, medr: {:.1f}, meanr: {:.1f}, mAP10: {:.3f}".format(
                    a2t_r1, a2t_r5, a2t_r10, a2t_r50, a2t_medr, a2t_meanr, a2t_mAP10))
        
        if test_dataloader_retrieval_clotho is not None:
            retrieval_results = validate_retrieval(model, test_dataloader_retrieval_clotho, device, main_logger, False)
            
            if is_main_process():
                # Extract text-to-audio results
                t2a_r1, t2a_r5, t2a_r10, t2a_r50, t2a_medr, t2a_meanr, t2a_mAP10 = retrieval_results["t2a"]
                main_logger.info("Clotho - Text-to-Audio: r1: {:.2f}, r5: {:.2f}, r10: {:.2f}, r50: {:.2f}, medr: {:.1f}, meanr: {:.1f}, mAP10: {:.3f}".format(
                    t2a_r1, t2a_r5, t2a_r10, t2a_r50, t2a_medr, t2a_meanr, t2a_mAP10))
                
                # Extract audio-to-text results
                a2t_r1, a2t_r5, a2t_r10, a2t_r50, a2t_medr, a2t_meanr, a2t_mAP10 = retrieval_results["a2t"]
                main_logger.info("Clotho - Audio-to-Text: r1: {:.2f}, r5: {:.2f}, r10: {:.2f}, r50: {:.2f}, medr: {:.1f}, meanr: {:.1f}, mAP10: {:.3f}".format(
                    a2t_r1, a2t_r5, a2t_r10, a2t_r50, a2t_medr, a2t_meanr, a2t_mAP10))
                
    if args.task in ['classification', 'all']: 
        if test_dataloader_esc50 is not None or test_dataloader_urbansound8k is not None or test_dataloader_vggsound is not None:
            if is_main_process():
                main_logger.info('Evaluating Audio Classification...')
        
        # template = eval_data_args.get("classification_template", "The sound of {}")
        template = "The sound of {}"
        # template = "{}"
        main_logger.info(f"Using template: {template}")

        if test_dataloader_urbansound8k is not None:
            accuracy = validate_classification(model, test_dataloader_urbansound8k, device, main_logger, False, template=template)
            if is_main_process():
                main_logger.info(f"Classification accuracy on UrbanSound8K: {accuracy:.2f}%")

        if test_dataloader_vggsound is not None:
            accuracy = validate_classification(model, test_dataloader_vggsound, device, main_logger, False, template=template)
            if is_main_process():
                main_logger.info(f"Classification accuracy on VGGSound: {accuracy:.2f}%")
        
        if test_dataloader_esc50 is not None:
            accuracy = validate_classification(model, test_dataloader_esc50, device, main_logger, False, template=template)
            if is_main_process():
                main_logger.info(f"Classification accuracy on ESC-50: {accuracy:.2f}%")
        
        
        if test_dataloader_esc50 is not None or test_dataloader_urbansound8k is not None or test_dataloader_vggsound is not None:
            if is_main_process():
                main_logger.info("Classification evaluation completed.")
            

    if is_main_process():
        main_logger.info("Evaluation completed.")


if __name__ == '__main__':
    main()
