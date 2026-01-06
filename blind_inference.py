import argparse
import os
import sys
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm
import librosa
import numpy as np

# Import networks
from networks.w2v2_aasist import Model as w2v2_aasist
from networks.aasist import Model as aasist
from networks.beats_aasist import Model as beats_aasist
from networks.eat_aasist import Model as eat_aasist  
from networks.eat_lrg_aasist_copy import Model as eat_lrg_aasist  
import config

# --- Blind Dataset ---
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class BlindDataset(Dataset):
    def __init__(self, file_list, args):
        self.file_list = file_list
        self.args = args
        
        TARGET_SR_16KHZ = 16000
        TARGET_SR_48KHZ = 48000
        
        if args.model == 'clap_htsat_fused_aasist':
            self.target_sr = TARGET_SR_48KHZ
        else: 
            self.target_sr = TARGET_SR_16KHZ
            
        original_duration_s = 64600 / TARGET_SR_16KHZ
        self.cut = int(original_duration_s * self.target_sr)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filepath = self.file_list[index]
        X, fs = librosa.load(filepath, sr=self.target_sr)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        filename = os.path.basename(filepath)
        return x_inp, filename

def main():
    parser = argparse.ArgumentParser(description='Blind Inference System')
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing .wav files')
    
    parser.add_argument('--base_output_dir', type=str, 
                        default=r'D:\Uni\SEM3\ENVSDD\code\EnvSDD_project\eval_output',
                        help='Base folder where results will be stored')
    parser.add_argument('--output_filename', type=str, required=True,
                        help='Name of the output file (e.g., eval_predictions.txt)')
    parser.add_argument('--model', type=str, default='beats_aasist', 
                        choices=['aasist', 'w2v2_aasist', 'beats_aasist', 'eat_aasist', 'eat_lrg_aasist'])
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    device = args.device


    ckpt_name = os.path.splitext(os.path.basename(args.model_path))[0]
    save_folder = os.path.join(args.base_output_dir, ckpt_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created new directory: {save_folder}")
        
    final_output_path = os.path.join(save_folder, args.output_filename)
    print(f"Output will be saved to: {final_output_path}")

    print(f'Loading model: {args.model}...')
    if args.model == 'w2v2_aasist':
        model = w2v2_aasist(args, device)
    elif args.model == 'aasist':
        aasist_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        model = aasist(aasist_config)
    elif args.model == 'beats_aasist':
        model = beats_aasist(args, device)
    
    elif args.model == 'eat_aasist':
        print("Initializing EAT-AASIST...")
        model = eat_aasist(args, device)
    elif args.model == 'eat_lrg_aasist':
        print("Initializing EAT-AASIST...")
        model = eat_lrg_aasist(args, device)
    else:
        print('Model not found')
        sys.exit()

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    search_path = os.path.join(args.input_dir, '*.wav')
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        print(f"Error: No .wav files found in {args.input_dir}")
        sys.exit()
        
    print(f"Found {len(file_list)} files.")
    dataset = BlindDataset(file_list, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with open(final_output_path, 'w') as fh:
        with torch.no_grad():
            for batch_x, filenames in tqdm(loader):
                batch_x = batch_x.to(device)
                batch_out = model(batch_x)
                
                # Extract score (Index 1 = Bonafide/Real score)
                batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
                
                for fname, score in zip(filenames, batch_score):
                    fh.write(f'{fname}|{score}\n')

    print(f"Done. Scores saved to: {final_output_path}")

if __name__ == '__main__':
    main()