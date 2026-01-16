import argparse
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

import config
from data_utils import genSpoof_list, ADD_Dataset, eval_to_score_file

from networks.w2v2_aasist import Model as w2v2_aasist
from networks.aasist import Model as aasist
from networks.beats_aasist import Model as beats_aasist
from networks.eat_lrg_aasist import Model as eat_lrg_aasist
from networks.eat_aasist import Model as eat_aasist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False 


def evaluate_accuracy(dev_loader: DataLoader, model: nn.Module, device: str) -> float:
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)
            
            val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    return val_loss


def produce_evaluation_file(dataset: Dataset, model: nn.Module, device: str, save_path: str) -> None:
    data_loader = DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=8
    )
    
    model.eval()
    model.to(device)

    with open(save_path, 'w') as fh:
        with torch.no_grad():
            for batch_x, utt_id in tqdm(data_loader, total=len(data_loader), desc="Generating Scores"):
                batch_x = batch_x.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    batch_out = model(batch_x)
                
                batch_score = batch_out[:, 1].float().cpu().numpy().ravel()
                for f, cm in zip(utt_id, batch_score):
                    fh.write(f'{f}|{cm}\n')

    print(f'Scores saved to {save_path}')


def train_epoch(
    train_loader: DataLoader, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str, 
    accumulation_steps: int
) -> float:
    running_loss = 0.0
    num_total = 0.0
    model.train()

    # Freeze EAT backbone in eval mode for stability
    if hasattr(model, 'ssl_model'):
        model.ssl_model.eval()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.cuda.amp.GradScaler()

    step_nums = 0
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Training'):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            batch_loss = batch_loss / accumulation_steps

        running_loss += (batch_loss.item() * batch_size)

        scaler.scale(batch_loss).backward()

        if (step_nums + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        step_nums += 1

    running_loss /= num_total
    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EAT-AASIST Training")

    # Dataset Paths
    parser.add_argument('--train_meta_json', type=str, default='dev_track1_train.json')
    parser.add_argument('--dev_meta_json', type=str, default='dev_track1_valid.json')
    parser.add_argument('--test_meta_json', type=str, default='test_track1.json')
    parser.add_argument('--protocols_path', type=str, default='./')


    # Hardware & Optimization
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--diff_lr', action='store_true', default=False)

    # Hyperparams
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--augment', action='store_true', default=False, help='Apply robust augmentation')

    # Model
    parser.add_argument('--model', type=str, default='eat_lrg_aasist')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--eval_output', type=str, default=None)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--exp_id', type=str, default='001')

    print('====== Begin ======')
    args = parser.parse_args()

    # --- Directory Setup ---
    output_folder = os.path.join(args.protocols_path, f'exps/exp_{args.exp_id}')
    os.makedirs(f'{output_folder}/ckpts', exist_ok=True)

    # --- Strict Seeding ---
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # --- Model Tagging ---
    model_tag = '{}_{}_ep{}_bs{}_acc{}_lr{}'.format(
        args.model, args.loss, args.num_epochs,
        args.batch_size, args.accumulation_steps, args.lr
    )
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    
    model_save_path = os.path.join(f'{output_folder}/ckpts', model_tag)
    if args.eval_output is None:
        args.eval_output = os.path.join(model_save_path, 'eval_scores.txt')
    os.makedirs(model_save_path, exist_ok=True)

    device = args.device
    print(f'Device: {device} | Batch Size: {args.batch_size} | Workers: {args.num_workers}')

    # --- Model Initialization ---
    if args.model == 'eat_lrg_aasist':
        model = eat_lrg_aasist(args, device)
    elif args.model == 'eat_aasist':
        model = eat_aasist(args, device)
    elif args.model == 'w2v2_aasist':
        model = w2v2_aasist(args, device)
    elif args.model == 'aasist':
        model_params = {
            "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        model = aasist(model_params)
    elif args.model == 'beats_aasist':
        model = beats_aasist(args, device)
    else:
        sys.exit(f'Model {args.model} not found')

    model.to(device)
    nb_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'Trainable parameters: {nb_params}')

    # --- Optimizer Configuration ---
    if hasattr(model, 'ssl_model') and args.diff_lr:
        print(f"Configuring Differential Learning Rates for {args.model}...")
        
        weight_params = []
        weight_ids = []
        if hasattr(model.ssl_model, 'layer_weights'):
            weight_params = [model.ssl_model.layer_weights]
            weight_ids = [id(p) for p in weight_params]
        
        ssl_params = [p for p in model.ssl_model.parameters() if id(p) not in weight_ids]
        ssl_ids = [id(p) for p in ssl_params]
        
        base_params = [p for p in model.parameters() if id(p) not in ssl_ids and id(p) not in weight_ids]

        param_groups = [
            {'params': ssl_params, 'lr': 5e-6},
            {'params': base_params, 'lr': args.lr}
        ]
        if weight_params:
            param_groups.append({'params': weight_params, 'lr': 1e-3})
            
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    else:
        print("Using Standard Adam Optimizer...")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model loaded : {args.model_path}')

    # --- Evaluation Mode ---
    if args.eval:
        print('====== Evaluation ======')
        d_label, file_eval = genSpoof_list(
            dir_meta=f'{config.metadata_json_file}/{args.test_meta_json}', 
            is_train=True, is_eval=False
        )
        eval_set = ADD_Dataset(args, list_IDs=file_eval, labels=d_label, is_eval=True, algo=None)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        eer = eval_to_score_file(args.eval_output, f'{config.metadata_json_file}/{args.test_meta_json}')
        
        eer_path = os.path.join(os.path.dirname(args.eval_output), "eer_test.txt")
        with open(eer_path, "w") as f:
            f.write(f"EER: {eer * 100:.6f}%\n")
        print(f"EER saved to {eer_path}")
        sys.exit()

    # --- Training Setup ---
    d_label_trn, file_train = genSpoof_list(
        dir_meta=f'{config.metadata_json_file}/{args.train_meta_json}', 
        is_train=True, is_eval=False
    )
    print(f'Train data: {len(file_train)} trials')

    # train_set = ADD_Dataset(args, list_IDs=file_train, labels=d_label_trn, algo=None)
    train_set = ADD_Dataset(args, list_IDs=file_train, labels=d_label_trn, algo=None, augment=args.augment)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    d_label_dev, file_dev = genSpoof_list(
        dir_meta=f'{config.metadata_json_file}/{args.dev_meta_json}', 
        is_train=False, is_eval=False
    )
    print(f'Validation data: {len(file_dev)} trials')

    # dev_set = ADD_Dataset(args, list_IDs=file_dev, labels=d_label_dev, algo=None)
    dev_set = ADD_Dataset(args, list_IDs=file_dev, labels=d_label_dev, algo=None, augment=False)
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    writer = SummaryWriter(f'logs/exp_{args.exp_id}')
    
    print('====== Train ======')
    val_not_decrease_epochs = 0
    min_val_loss = 1e3

    for epoch in range(args.num_epochs):
        if val_not_decrease_epochs == args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        running_loss = train_epoch(train_loader, model, optimizer, device, args.accumulation_steps)
        val_loss = evaluate_accuracy(dev_loader, model, device)

        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('train_loss', running_loss, epoch)

        print(f'\nEpoch {epoch}: train_loss:{running_loss:.4f} - val_loss:{val_loss:.4f} ')

        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}.pth'))

        if val_loss < min_val_loss:
            val_not_decrease_epochs = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print("  [Usefulness] New best model saved.")
        else:
            val_not_decrease_epochs += 1

    print("Training finished.")