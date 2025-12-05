import argparse
import os
import random
import sys
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

# --- Local Imports ---
import config
from data_utils import genSpoof_list, ADD_Dataset, eval_to_score_file
from rawboost_gpu import RawBoostGPU

# --- Network Imports ---
from networks.w2v2_aasist import Model as w2v2_aasist
from networks.aasist import Model as aasist
from networks.beats_aasist import Model as beats_aasist
from networks.eat_lrg_aasist import Model as eat_lrg_aasist
from networks.eat_aasist import Model as eat_aasist


# Allow TF32 on Ampere+ GPUs (Significant speedup for FP32 ops)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable CUDNN autotuner to find best algorithms for your hardware
torch.backends.cudnn.benchmark = True


def evaluate_accuracy(
    dev_loader: DataLoader, 
    model: nn.Module, 
    device: str
) -> float:
    """
    Evaluates the model on the validation set using Cross Entropy Loss.
    """
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    
    # Weighted Cross Entropy for class imbalance
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            # Non-blocking transfer is crucial for speed
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

            # Use mixed precision (bfloat16) for evaluation
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)
            
            val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    return val_loss


def produce_evaluation_file(
    dataset: Dataset, 
    model: nn.Module, 
    device: str, 
    save_path: str
) -> None:
    """
    Generates score file for the evaluation dataset.
    """
    data_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        drop_last=False, 
        num_workers=8
    )
    
    model.eval()
    model.to(device)

    with open(save_path, 'w') as fh:
        with torch.no_grad():
            for batch_x, utt_id in tqdm(data_loader, total=len(data_loader), desc="Generating Scores"):
                batch_x = batch_x.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    batch_out = model(batch_x)
                
                # Extract scores for the 'spoof' class (index 1)
                batch_score = batch_out[:, 1].float().cpu().numpy().ravel()
                
                for f, cm in zip(utt_id, batch_score):
                    fh.write(f'{f}|{cm}\n')

    print(f'Scores saved to {save_path}')


def train_epoch(
    train_loader: DataLoader, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str, 
    accumulation_steps: int, 
    rawboost_fn: Optional[Callable] = None
) -> float:
    """
    Runs one training epoch with Gradient Accumulation and Mixed Precision.
    """
    running_loss = 0.0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Gradient Scaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    step_nums = 0
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Training'):
        batch_size = batch_x.size(0)
        num_total += batch_size

        # 1. Non-blocking Async Transfer (CPU -> GPU)
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

        # 2. Apply GPU Augmentation (Vectorized)
        if rawboost_fn is not None:
            with torch.no_grad():  # No gradients needed for augmentation logic
                batch_x = rawboost_fn(batch_x)

        # 3. Forward Pass with Automatic Mixed Precision (BFloat16)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            batch_loss = batch_loss / accumulation_steps

        running_loss += (batch_loss.item() * batch_size)

        # 4. Backward Pass
        scaler.scale(batch_loss).backward()

        # 5. Optimizer Step
        if (step_nums + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        step_nums += 1

    running_loss /= num_total
    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EAT-AASIST High Performance Training")

    # Dataset Paths
    parser.add_argument('--train_meta_json', type=str, default='tta/dev/train.json')
    parser.add_argument('--dev_meta_json', type=str, default='tta/dev/valid.json')
    parser.add_argument('--test_meta_json', type=str, default='tta/test/test_01.json')
    parser.add_argument('--protocols_path', type=str, default='./')

    # Hardware & Optimization
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of data loader workers")
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # Training Hyperparams
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--algo', type=int, default=5, help='RawBoost Algo (default 5)')
    parser.add_argument('--seed', type=int, default=1234)

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
    if not os.path.exists(f'{output_folder}/ckpts'):
        os.makedirs(f'{output_folder}/ckpts')

    # --- Seeding ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # False is faster, True is reproducible (but slower on some kernels)
    torch.backends.cudnn.deterministic = False

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
        
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    device = args.device
    print(f'Device: {device} | Batch Size: {args.batch_size} | Workers: {args.num_workers}')

    # --- Model Initialization ---
    if args.model == 'eat_lrg_aasist':
        print("Initializing EAT-Large AASIST...")
        model = eat_lrg_aasist(args, device)
    elif args.model == 'eat_aasist':
        print("Initializing BASE EAT-AASIST...")
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
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    print(f'Trainable parameters: {nb_params}')

    # --- Augmentation Initialization ---
    rawboost = None
    if not args.eval and args.algo:
        print(f"Initializing GPU-Accelerated RawBoost (Algo {args.algo})...")
        rawboost = RawBoostGPU(algo=args.algo, device=device).to(device)

    # --- Optimizer Configuration ---
    if args.model == 'eat_lrg_aasist':
        print("Configuring Differential Learning Rates for EAT-Large...")
        eat_params = list(map(id, model.ssl_model.model.parameters()))
        weights_params = list(map(id, [model.ssl_model.layer_weights]))
        base_params = filter(lambda p: id(p) not in eat_params and id(p) not in weights_params, model.parameters())

        optimizer = torch.optim.AdamW([
            {'params': model.ssl_model.model.parameters(), 'lr': 5e-6},  # Backbone (Slow)
            {'params': [model.ssl_model.layer_weights], 'lr': 1e-3},     # Weights (Fast)
            {'params': base_params, 'lr': args.lr}                       # AASIST (Normal)
        ], weight_decay=args.weight_decay)
    else:
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
            is_train=True, 
            is_eval=False
        )
        eval_set = ADD_Dataset(args, list_IDs=file_eval, labels=d_label, is_eval=True, algo=None)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        eer = eval_to_score_file(args.eval_output, f'{config.metadata_json_file}/{args.test_meta_json}')
        sys.exit()

    # --- Training Setup ---
    d_label_trn, file_train = genSpoof_list(
        dir_meta=f'{config.metadata_json_file}/{args.train_meta_json}', 
        is_train=True, 
        is_eval=False
    )
    print(f'Train data: {len(file_train)} trials')

    # CRITICAL: algo=None because we now use GPU augmentation
    train_set = ADD_Dataset(args, list_IDs=file_train, labels=d_label_trn, algo=None)

    # OPTIMIZATION: High-throughput Data Loader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,        # Required for non_blocking=True
        prefetch_factor=2,      # Preload 2 batches per worker
        persistent_workers=True # Keep workers alive between epochs
    )

    d_label_dev, file_dev = genSpoof_list(
        dir_meta=f'{config.metadata_json_file}/{args.dev_meta_json}', 
        is_train=False, 
        is_eval=False
    )
    print(f'Validation data: {len(file_dev)} trials')

    dev_set = ADD_Dataset(args, list_IDs=file_dev, labels=d_label_dev, algo=None)

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

        # Call train epoch with GPU augmentations
        running_loss = train_epoch(
            train_loader, model, optimizer, device,
            args.accumulation_steps, rawboost_fn=rawboost
        )

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