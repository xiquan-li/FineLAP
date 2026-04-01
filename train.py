import sys
import os
import warnings
warnings.filterwarnings("ignore")
import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
import ruamel.yaml as yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.logger_config import setup_logger
from data_handling.grounding_dataset import AudioGroundingDataset
from data_handling.sed_dataset import SoundEventDataset
from data_handling.mix_dataset import MixDataset
from data_handling.caption_dataset import AudioCaptionDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from utils.optim_utils import get_optimizer, cosine_lr
from utils.utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_seed,
    AverageMeter,
    move_to_device,
)
from pathlib import Path
import sys
import numpy as np
import torch.nn.functional as F
from evaluate import validate_grounding, validate_sed, validate_retrieval, validate_classification


def train(model, dataloader, optimizer, scheduler, device, epoch, use_wandb, alpha, beta, gamma, dist_impl="reduce"):
    model.train()

    epoch_loss = AverageMeter()
    epoch_clip_loss = AverageMeter()
    epoch_grounding_loss = AverageMeter()
    epoch_distillation_loss = AverageMeter()
    start_time = time.time()

    rank = get_rank()
    world_size = get_world_size()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    model_without_ddp.update_loss_fn_params(rank=rank, world_size=world_size, dist_impl=dist_impl)

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    if is_main_process():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), colour='green')
    else:
        pbar = enumerate(dataloader)
    
    for batch_id, batch in pbar:
        audio = move_to_device(batch["audios"], device)
        captions = batch["caption"]
        phrases_batch = batch["phrases"]   # list of list[str]
        frame_labels_batch = batch["frame_labels"]  # list of tensors [N_i, T]
        has_grounding = batch["has_grounding"]  # list of bool
        batch_audio_chosen_segments = move_to_device(batch['batch_audio_chosen_segments'], device)
        batch_dense_audio_embeds_idx = move_to_device(batch['dense_audio_embeds_idx'], device)

        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        if use_wandb and is_main_process():
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        clip_loss, grounding_loss, distillation_loss = model(
            audio, 
            captions, 
            phrases_batch, 
            frame_labels_batch, 
            has_grounding,
            batch_audio_chosen_segments,
            batch_dense_audio_embeds_idx
        )

        total_loss = alpha * clip_loss + beta * grounding_loss + gamma * distillation_loss
        total_loss.backward()
        optimizer.step()

        clip_loss_val = clip_loss.detach().cpu().item()
        grounding_loss_val = grounding_loss.detach().cpu().item()
        total_loss_val = total_loss.detach().cpu().item()
        distillation_loss_val = distillation_loss.detach().cpu().item()
        epoch_loss.update(total_loss_val)
        epoch_clip_loss.update(clip_loss_val)
        epoch_grounding_loss.update(grounding_loss_val)
        epoch_distillation_loss.update(distillation_loss_val)

        if is_main_process():
            pbar.set_postfix({
                'total': f'{total_loss_val:.4f}',
                'clip': f'{clip_loss_val:.4f}',
                'ground': f'{grounding_loss_val:.4f}',
                'distill': f'{distillation_loss_val:.4f}',
            })
        
        if use_wandb and is_main_process():
            wandb.log({
                "loss/clip": clip_loss_val,
                "loss/grounding": grounding_loss_val,
                "loss/distillation": distillation_loss_val,
                "loss/total": total_loss_val,
            })

    elapsed_time = time.time() - start_time

    if use_wandb and is_main_process():
        wandb.log({
            "loss": epoch_loss.avg,
            "loss/epoch_clip": epoch_clip_loss.avg,
            "loss/epoch_grounding": epoch_grounding_loss.avg,
            "loss/epoch_distillation": epoch_distillation_loss.avg,
            "epoch": epoch
        })

    return {
        "loss": epoch_loss.avg,
        "clip_loss": epoch_clip_loss.avg,
        "grounding_loss": epoch_grounding_loss.avg,
        "distillation_loss": epoch_distillation_loss.avg,
        "time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/clap_grounding.yaml", type=str,
                        help="Model config files")
    parser.add_argument("--data_config", default=None, type=str,
                        help="Evaluation dataset config file (optional, for test evaluation)")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="Name of this experiment.")
    parser.add_argument("-d", "--exp_dir", default="/data/xiquan.li/exps/fclap/train_grounding", type=str,
                        help="Dir to put the experiment")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed")
    parser.add_argument("-b", "--blacklist", default='blacklist_exclude_all_ac.json', type=str,
                        help="Blacklist file.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of epochs for training")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether or not use wandb")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Alpha for training")
    parser.add_argument("--beta", type=float, default=5,
                        help="Beta for training")

    args = parser.parse_args()

    ## ============================== Config ==============================
    exp_name, exp_dir = args.exp_name, args.exp_dir
    config = OmegaConf.load(args.config)
    data_config = OmegaConf.load(args.data_config)
    config = OmegaConf.merge(config, data_config)

    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict["optim_args"]["lr"] = args.lr
    config_dict["training"]["seed"] = args.seed
    config_dict["training"]["epochs"] = args.epochs
    config_dict["training"]["alpha"] = args.alpha
    config_dict["training"]["beta"] = args.beta
    config_dict["data_args"]["train_data_args"]["batch_size"] = args.batch_size
    
    config = config_dict

    init_distributed_mode(config["dist_args"])
    if is_dist_avail_and_initialized():
        device = torch.device(f"cuda:{config['dist_args']['gpu']}")
    else:
        device = torch.device(config["training"]["device"])

    seed = config["training"]["seed"] + get_rank()
    setup_seed(seed)
    if args.use_wandb and is_main_process(): 
        wandb.init(
            project="FineLAP",
            name=exp_name,
            config=config
        )

    ## ============================== Logging ==============================
    log_output_dir = Path(exp_dir, exp_name, 'logging')
    model_output_dir = Path(exp_dir, exp_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    main_logger = setup_logger(exp_dir, exp_name, log_output_dir.joinpath('training_log.txt'))
    printer = PrettyPrinter()
    config_output_dir = Path(exp_dir, exp_name)
    config_save_path = config_output_dir / 'config.yaml'
    yaml_writer = yaml.YAML(typ='unsafe', pure=True)
    yaml_writer.default_flow_style = False
    if is_main_process():
        main_logger.info(f'Config: {config}')
    if is_main_process():
        with open(config_save_path, 'w') as f:
            yaml_writer.dump(config, f)
        main_logger.info(f'Config saved to: {config_save_path}')
    
    ## ============================== Model ==============================
    from models.finelap import FineLAP
    model = FineLAP(config)

    finelap_ckpt_path = config["model_args"].get("ckpt_path", "")
    if finelap_ckpt_path:
        if is_main_process():
            main_logger.info(f"Loading FineLAP checkpoint from: {finelap_ckpt_path}")
        checkpoint = torch.load(finelap_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        if is_main_process():
            main_logger.info("FineLAP checkpoint loaded successfully.")

    model = model.to(device)
    if args.use_wandb and is_main_process(): 
        wandb.watch(model)

    if config["training"]["freeze_text_encoder"]:
        if is_main_process():
            main_logger.info("Freezing text encoders ... ")
        for name, param in model.text_encoder.text_encoder.named_parameters():
            param.requires_grad = False
    if config["training"]["freeze_audio_encoder"]:
        if is_main_process():
            main_logger.info("Freezing audio encoder")
        for name, param in model.audio_encoder.audio_enc.named_parameters():
            param.requires_grad = False
        model.audio_encoder.audio_enc.eval()
    if config["training"]["freeze_text_proj"]: 
        if is_main_process():
            main_logger.info("Freezing text projector ... ")
        for p in model.text_proj.parameters(): 
            p.requires_grad = False

    non_trainable_params = [i for i in model.parameters() if i.requires_grad==False]
    if is_main_process():
        main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])/1e6:.2f}M')
        main_logger.info(f'Total numer of learnable parameters: {sum([i.numel() for i in model.parameters() if i.requires_grad==True])/1e6:.2f}M')
        main_logger.info(f'Total number of non-trainable parameters: {sum([i.numel()/1e6 for i in non_trainable_params]):.2f}M')
        main_logger.info(f'Non-trainable parameter names:')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                main_logger.info(f'  {name}: {param.shape} ({param.numel()/1e6:.2f}M parameters)')
        main_logger.info('Model Parameter breakdown: ')
        audio_encoder_params = sum(p.numel() for p in model.audio_encoder.parameters())
        audio_encoder_trainable = sum(p.numel() for p in model.audio_encoder.parameters() if p.requires_grad)
        main_logger.info(f'- Audio Encoder: {audio_encoder_params/1e6:.2f}M total, {audio_encoder_trainable/1e6:.2f}M trainable')

        audio_proj_params = sum(p.numel() for p in model.global_audio_proj.parameters())
        audio_proj_trainable = sum(p.numel() for p in model.global_audio_proj.parameters() if p.requires_grad)
        main_logger.info(f'- Audio Global Projector: {audio_proj_params/1e6:.2f}M total, {audio_proj_trainable/1e6:.2f}M trainable')
        
        audio_adapter_params = sum(p.numel() for p in model.local_audio_proj.parameters())
        audio_adapter_trainable = sum(p.numel() for p in model.local_audio_proj.parameters() if p.requires_grad)
        main_logger.info(f'- Audio Local Projector: {audio_adapter_params/1e6:.2f}M total, {audio_adapter_trainable/1e6:.2f}M trainable')
    
        text_encoder_params = sum(p.numel() for p in model.text_encoder.parameters())
        text_encoder_trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
        main_logger.info(f'- Text Encoder: {text_encoder_params/1e6:.2f}M total, {text_encoder_trainable/1e6:.2f}M trainable')
        
        text_proj_params = sum(p.numel() for p in model.global_text_proj.parameters())
        text_proj_trainable = sum(p.numel() for p in model.global_text_proj.parameters() if p.requires_grad)
        main_logger.info(f'- Text Projector: {text_proj_params/1e6:.2f}M total, {text_proj_trainable/1e6:.2f}M trainable')
        
    ## ============================== Training data ==============================
    train_dataset_mixed = MixDataset(metadata_files=config["data_args"]["train_data_args"]["metadata_files"], 
                                    spec_aug=config["data_args"]["train_data_args"]["spec_aug"],
                                    max_phrases=config["data_args"]["train_data_args"].get("max_phrases", 20), 
                                    max_split_segments=config["data_args"]["train_data_args"]["max_split_segments"],
                                    max_chosen_segments=config["data_args"]["train_data_args"]["max_chosen_segments"],
                                    **config["data_args"]["basic_data_args"])
    train_dataset = ConcatDataset([train_dataset_mixed])
    
    if is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        dataloader = DataLoader(dataset=train_dataset,
                                collate_fn=train_dataset_mixed.collate_fn,
                                sampler=train_sampler,
                                batch_size=config["data_args"]["train_data_args"]["batch_size"],
                                num_workers=config["data_args"]["train_data_args"]["num_workers"])
    else:
        dataloader = DataLoader(dataset=train_dataset,
                                collate_fn=train_dataset_mixed.collate_fn,
                                shuffle=True,
                                batch_size=config["data_args"]["train_data_args"]["batch_size"],
                                num_workers=config["data_args"]["train_data_args"]["num_workers"])
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_steps"],
                          steps=len(dataloader) * config["training"]["epochs"])
    start_epoch = 1
    max_epoch = config["training"]["epochs"]
    if is_main_process():
        # main_logger.info('Training setting:\n'
        #                  f'{printer.pformat(config)}')
        main_logger.info(f'Size of training set: {len(dataloader.dataset)}, number of batches: {len(dataloader)}')

    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config["dist_args"].get("gpu", 0)],
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    ## ============================== Validation data ==============================
    val_dataloader_retrieval_audiocaps = None
    if "val_data_args_retrieval_audiocaps" in config["data_args"]:
        val_dataset_retrieval_audiocaps = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_retrieval_audiocaps"])
        val_dataloader_retrieval_audiocaps = DataLoader(dataset=val_dataset_retrieval_audiocaps, 
                                                        batch_size=16, 
                                                        num_workers=16, 
                                                        shuffle=False, 
                                                        collate_fn=val_dataset_retrieval_audiocaps.collate_fn, 
                                                        drop_last=False)
    
    val_dataloader_retrieval_clotho = None
    if "val_data_args_retrieval_clotho" in config["data_args"]:
        val_dataset_retrieval_clotho = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_retrieval_clotho"])
        val_dataloader_retrieval_clotho = DataLoader(dataset=val_dataset_retrieval_clotho, 
                                                     batch_size=16, 
                                                     num_workers=16, 
                                                     shuffle=False, 
                                                     collate_fn=val_dataset_retrieval_clotho.collate_fn, 
                                                     drop_last=False)
    
    val_dataloader_grounding = None
    if "val_data_args_grounding" in config["data_args"]:
        val_dataset_grounding = AudioGroundingDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_grounding"])
        val_dataloader_grounding = DataLoader(dataset=val_dataset_grounding,
                                            collate_fn=val_dataset_grounding.collate_fn,
                                            shuffle=True,
                                            batch_size=16,
                                            num_workers=16,
                                            drop_last=False)
    
    val_dataloader_sed = None
    if "val_data_args_sed" in config["data_args"]:
        val_dataset_sed = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_sed"])
        val_dataloader_sed = DataLoader(dataset=val_dataset_sed, 
                                        batch_size=16, 
                                        num_workers=16, 
                                        shuffle=False, 
                                        collate_fn=val_dataset_sed.collate_fn, 
                                        drop_last=False)
    
    val_dataloader_audioset_strong = None
    if "val_data_args_audioset_strong" in config["data_args"]:
        val_dataset_audioset_strong = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_audioset_strong"])
        val_dataloader_audioset_strong = DataLoader(dataset=val_dataset_audioset_strong, 
                                                    batch_size=16, 
                                                    num_workers=16, 
                                                    shuffle=False, 
                                                    collate_fn=val_dataset_audioset_strong.collate_fn, 
                                                    drop_last=False)
    
    val_dataloader_simulated_sed = None
    if "val_data_args_simulated_sed" in config["data_args"]:
        val_dataset_simulated_sed = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_simulated_sed"])
        val_dataloader_simulated_sed = DataLoader(dataset=val_dataset_simulated_sed, 
                                                batch_size=16, 
                                                num_workers=16, 
                                                shuffle=False, 
                                                collate_fn=val_dataset_simulated_sed.collate_fn, 
                                                drop_last=False)
    
    val_dataloader_urbansed = None
    if "val_data_args_urbansed" in config["data_args"]:
        val_dataset_urbansed = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["val_data_args_urbansed"])
        val_dataloader_urbansed = DataLoader(dataset=val_dataset_urbansed, 
                                            batch_size=16, 
                                            num_workers=16, 
                                            shuffle=False, 
                                            collate_fn=val_dataset_urbansed.collate_fn, 
                                            drop_last=False)

    ## ============================== Evaluation data ==============================
    test_dataloader_retrieval_audiocaps = None
    if "test_data_args_retrieval_audiocaps" in config["data_args"]:
        test_dataset_retrieval_audiocaps = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_retrieval_audiocaps"])
        test_dataloader_retrieval_audiocaps = DataLoader(dataset=test_dataset_retrieval_audiocaps, 
                                                        batch_size=16, 
                                                        num_workers=16, 
                                                        shuffle=False, 
                                                        collate_fn=test_dataset_retrieval_audiocaps.collate_fn, 
                                                        drop_last=False)
    
    test_dataloader_retrieval_clotho = None
    if "test_data_args_retrieval_clotho" in config["data_args"]:
        test_dataset_retrieval_clotho = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_retrieval_clotho"])
        test_dataloader_retrieval_clotho = DataLoader(dataset=test_dataset_retrieval_clotho, 
                                                     batch_size=16, 
                                                     num_workers=16, 
                                                     shuffle=False, 
                                                     collate_fn=test_dataset_retrieval_clotho.collate_fn, 
                                                     drop_last=False)
    
    test_dataloader_grounding = None
    if "test_data_args_grounding" in config["data_args"]:
        test_dataset_grounding = AudioGroundingDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_grounding"])
        test_dataloader_grounding = DataLoader(dataset=test_dataset_grounding,
                                               collate_fn=test_dataset_grounding.collate_fn,
                                               shuffle=False,
                                               batch_size=16,
                                               num_workers=16,
                                               drop_last=False)
    
    test_dataloader_sed = None
    if "test_data_args_sed" in config["data_args"]:
        test_dataset_sed = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_sed"])
        test_dataloader_sed = DataLoader(dataset=test_dataset_sed, 
                                         batch_size=16, 
                                         num_workers=16, 
                                         shuffle=False, 
                                         collate_fn=test_dataset_sed.collate_fn, 
                                         drop_last=False)
    
    test_dataloader_audioset_strong = None
    if "test_data_args_audioset_strong" in config["data_args"]:
        test_dataset_audioset_strong = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_audioset_strong"])
        test_dataloader_audioset_strong = DataLoader(dataset=test_dataset_audioset_strong, 
                                                     batch_size=16, 
                                                     num_workers=16, 
                                                     shuffle=False, 
                                                     collate_fn=test_dataset_audioset_strong.collate_fn, 
                                                     drop_last=False)
    
    test_dataloader_urbansed = None
    if "test_data_args_urbansed" in config["data_args"]:
        test_dataset_urbansed = SoundEventDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_urbansed"])
        test_dataloader_urbansed = DataLoader(dataset=test_dataset_urbansed, 
                                             batch_size=16, 
                                             num_workers=16, 
                                             shuffle=False, 
                                             collate_fn=test_dataset_urbansed.collate_fn, 
                                             drop_last=False)
    
    test_dataloader_esc50 = None
    if "test_data_args_esc50" in config["data_args"]:
        test_dataset_esc50 = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_esc50"])
        test_dataloader_esc50 = DataLoader(dataset=test_dataset_esc50, 
                                            batch_size=16,
                                            num_workers=16, 
                                            shuffle=False, 
                                            collate_fn=test_dataset_esc50.collate_fn, 
                                            drop_last=False)
    
    test_dataloader_urbansound8k = None
    if "test_data_args_urbansound8k" in config["data_args"]:
        test_dataset_urbansound8k = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_urbansound8k"])
        test_dataloader_urbansound8k = DataLoader(dataset=test_dataset_urbansound8k, 
                                                    batch_size=16,
                                                    num_workers=16, 
                                                    shuffle=False, 
                                                    collate_fn=test_dataset_urbansound8k.collate_fn, 
                                                    drop_last=False)
    
    test_dataloader_vggsound = None
    if "test_data_args_vggsound" in config["data_args"]:
        test_dataset_vggsound = AudioCaptionDataset(**config["data_args"]["basic_data_args"], **config["data_args"]["test_data_args_vggsound"])
        test_dataloader_vggsound = DataLoader(dataset=test_dataset_vggsound, 
                                                batch_size=16,
                                                num_workers=16, 
                                                shuffle=False, 
                                                collate_fn=test_dataset_vggsound.collate_fn, 
                                                drop_last=False)

    ## ============================== Training ==============================
    loss_stats = []
    val_psds_stats = []
    for epoch in range(start_epoch, max_epoch + 1):
        if is_main_process():
            main_logger.info(f'Training for epoch [{epoch}]')

        dist_impl = config["training"].get("dist_impl", "reduce")  # Default to 'reduce'
        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch, args.use_wandb,
                            config["training"]["alpha"], config["training"]["beta"], config["training"]["gamma"],
                            dist_impl)
        loss = train_statics["loss"]
        clip_loss = train_statics["clip_loss"]
        grounding_loss = train_statics["grounding_loss"]
        distillation_loss = train_statics["distillation_loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        if is_main_process():
            main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f} '
                             f'(clip: {clip_loss:.3f}, grounding: {grounding_loss:.3f}, distillation: {distillation_loss:.3f}), '
                             f'time: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()
        
        ## ============================== Validation ==============================
        # retrieval
        if val_dataloader_retrieval_audiocaps is not None or val_dataloader_retrieval_clotho is not None:
            if is_main_process():
                main_logger.info(f'Validation retrieval for epoch [{epoch}]')
        if val_dataloader_retrieval_audiocaps is not None:
            retrieval_results_audiocaps = validate_retrieval(model, val_dataloader_retrieval_audiocaps, device, main_logger, args.use_wandb)
        if val_dataloader_retrieval_clotho is not None:
            retrieval_results_clotho = validate_retrieval(model, val_dataloader_retrieval_clotho, device, main_logger, args.use_wandb)
        # grounding
        if val_dataloader_grounding is not None:
            if is_main_process():
                main_logger.info(f'Validation grounding for epoch [{epoch}]')
            save_dir = log_output_dir / "psds_epoch_{}".format(epoch)
            psds1, psds2, psds2021, th_auc = validate_grounding(model, val_dataloader_grounding, device, save_dir)
            if is_main_process():
                main_logger.info("Dataset: {}, psds1: {}, psds2: {}, psds2021: {}, th_auc: {}".format("AudioGrounding", psds1, psds2, psds2021, th_auc))
            if args.use_wandb and is_main_process(): 
                wandb.log({
                f"AudioGrounding:val/psds1": psds1,
                f"AudioGrounding:val/psds2": psds2,
                f"AudioGrounding:val/psds2021": psds2021,
                f"AudioGrounding:val/th_auc": th_auc
            })
        # SED
        if val_dataloader_sed is not None:
            if is_main_process():
                main_logger.info(f'Validation DESED for epoch [{epoch}]')
            psds1_sed, psds2_sed, eb_f1, seg_f1 = validate_sed(model, val_dataloader_sed, device)
            if is_main_process():
                main_logger.info("Dataset: {}, psds1: {}, psds2: {}".format("DESED", psds1_sed, psds2_sed))
            if args.use_wandb and is_main_process(): 
                wandb.log({
                f"DESED:val/psds1": psds1_sed,
                f"DESED:val/psds2": psds2_sed,
            })
        if val_dataloader_simulated_sed is not None:
            if is_main_process():
                main_logger.info(f'Validation Simulated SED for epoch [{epoch}]')
            psds1, psds2, eb_f1, seg_f1 = validate_sed(model, val_dataloader_simulated_sed, device, only_psds1=True)
            if is_main_process():
                main_logger.info("Dataset: {}, psds1: {}, psds2: {}".format("Simulated SED", psds1, psds2))
            if args.use_wandb and is_main_process(): 
                wandb.log({
                f"Simulated SED:val/psds1": psds1,
                f"Simulated SED:val/psds2": psds2,
            })
        if val_dataloader_audioset_strong is not None:
            if is_main_process():
                main_logger.info(f'Validation AudioSet Strong for epoch [{epoch}]')
            psds1, psds2, eb_f1, seg_f1 = validate_sed(model, val_dataloader_audioset_strong, device, only_psds1=True, variance_penalty=False)
            if is_main_process():
                main_logger.info("Dataset: {}, psds1: {}".format("AudioSet Strong", psds1))
            if args.use_wandb and is_main_process(): 
                wandb.log({
                f"AudioSet Strong:val/psds1": psds1,
            })
        if val_dataloader_urbansed is not None:
            if is_main_process():
                main_logger.info(f'Validation UrbanSED for epoch [{epoch}]')
            psds1, psds2, eb_f1, seg_f1 = validate_sed(model, val_dataloader_urbansed, device, only_psds1=False)
            if is_main_process():
                main_logger.info("Dataset: {}, psds1: {}, psds2: {}".format("UrbanSED", psds1, psds2))
            if args.use_wandb and is_main_process(): 
                wandb.log({
                f"UrbanSED:val/psds1": psds1,
                f"UrbanSED:val/psds2": psds2,
            })

        criterion = None
        if val_dataloader_grounding is not None and val_dataloader_sed is not None:
            criterion = np.mean([psds1_sed, psds2_sed, psds2021])
            val_psds_stats.append(criterion)
        if criterion is not None and criterion >= max(val_psds_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch
            }
            main_logger.info(f"Saving model ckpt for epoch {epoch}...")
            torch.save(sav_obj, str(model_output_dir) + "/best_model.pt")
    
    if is_main_process():
        main_logger.info(f"Saving model ckpt for last epoch {epoch}...")
        sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }
        torch.save(sav_obj, str(model_output_dir) + "/last_epoch.pt")

    ## ============================== Evaluation ==============================
    if is_main_process():
        main_logger.info('========================================== Evaluation start ==========================================')

    # retrieval
    if test_dataloader_retrieval_audiocaps is not None:
        retrieval_results_audiocaps = validate_retrieval(model, test_dataloader_retrieval_audiocaps, device, main_logger, args.use_wandb)
    if test_dataloader_retrieval_clotho is not None:
        retrieval_results_clotho = validate_retrieval(model, test_dataloader_retrieval_clotho, device, main_logger, args.use_wandb)
    # grounding
    if test_dataloader_grounding is not None:
        save_dir = log_output_dir / "psds_evaluation"
        save_dir.mkdir(parents=True, exist_ok=True)
        psds1, psds2, psds2021, th_auc  = validate_grounding(model, test_dataloader_grounding, device, save_dir)
        if is_main_process():
            main_logger.info("Dataset: {}, psds1: {}, psds2: {}, psds2021: {}, th_auc: {}".format("AudioGrounding", psds1, psds2, psds2021, th_auc))
        if args.use_wandb and is_main_process(): 
            wandb.log({
            f"AudioGrounding:test/psds1": psds1,
            f"AudioGrounding:test/psds2": psds2,
            f"AudioGrounding:test/psds2021": psds2021,
            f"AudioGrounding:test/th_auc": th_auc
        })
    # SED   
    if test_dataloader_sed is not None:
        psds1, psds2, eb_f1, seg_f1  = validate_sed(model, test_dataloader_sed, device)
        if is_main_process():
            main_logger.info("Dataset: {}, psds1: {}, psds2: {}".format("DESED", psds1, psds2))
        if args.use_wandb and is_main_process(): 
            wandb.log({
            f"DESED:test/psds1": psds1,
            f"DESED:test/psds2": psds2,
        })
    if test_dataloader_audioset_strong is not None:
        psds1, psds2, eb_f1, seg_f1  = validate_sed(model, test_dataloader_audioset_strong, device, only_psds1=True, variance_penalty=False)
        if is_main_process():
            main_logger.info("Dataset: {}, psds1: {}".format("AudioSet Strong", psds1))
        if args.use_wandb and is_main_process(): 
            wandb.log({
            f"AudioSet Strong:test/psds1": psds1,
        })
    if test_dataloader_urbansed is not None:
        psds1, psds2, eb_f1, seg_f1 = validate_sed(model, test_dataloader_urbansed, device, only_psds1=False)
        if is_main_process():
            main_logger.info("Dataset: {}, psds1: {}, psds2: {}".format("UrbanSED", psds1, psds2))
        if args.use_wandb and is_main_process(): 
            wandb.log({
            f"UrbanSED:test/psds1": psds1,
            f"UrbanSED:test/psds2": psds2,
        })
    # classification
    if test_dataloader_esc50 is not None or test_dataloader_urbansound8k is not None or test_dataloader_vggsound is not None:
        if is_main_process():
            main_logger.info('Evaluating Audio Classification...')
        
        template = config["data_args"].get("classification_template", "The sound of {}")
        
        if test_dataloader_vggsound is not None:
            accuracy = validate_classification(model, test_dataloader_vggsound, device, main_logger, args.use_wandb, template=template)
            if is_main_process():
                main_logger.info(f"Classification accuracy on VGGSound: {accuracy:.2f}%")
        
        if test_dataloader_esc50 is not None:
            accuracy = validate_classification(model, test_dataloader_esc50, device, main_logger, args.use_wandb, template=template)
            if is_main_process():
                main_logger.info(f"Classification accuracy on ESC-50: {accuracy:.2f}%")
        
        if test_dataloader_urbansound8k is not None:
            accuracy = validate_classification(model, test_dataloader_urbansound8k, device, main_logger, args.use_wandb, template=template)
            if is_main_process():
                main_logger.info(f"Classification accuracy on UrbanSound8K: {accuracy:.2f}%")
        if is_main_process():
            main_logger.info("Classification evaluation completed.")

    if is_main_process():
        main_logger.info("Done.")
    if args.use_wandb and is_main_process(): 
        wandb.finish()


if __name__ == '__main__':
    main()
