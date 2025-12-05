import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch import Tensor
from torch.utils.data import Dataset

# --- Local Imports ---
try:
    import eval_metrics as em
except ImportError:
    import sys
    print("Warning: eval_metrics not found. Evaluation functions may fail.", file=sys.stderr)


TARGET_SR_16KHZ = 16000
TARGET_SR_48KHZ = 48000
MAX_LEN_DEFAULT = 64600


def pad_tensor(x: Tensor, max_len: int = MAX_LEN_DEFAULT) -> Tensor:
    """
    Pads or crops a tensor to a fixed length in the time dimension.
    
    Args:
        x (Tensor): Input tensor of shape (Time,) or (Channels, Time).
        max_len (int): Target length.
        
    Returns:
        Tensor: Padded/Cropped tensor.
    """
    # Ensure (Channels, Time) format
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Crop if too long
    if x.shape[1] >= max_len:
        return x[:, :max_len]

    # Pad with zeros if too short
    diff = max_len - x.shape[1]
    # Pad last dimension: (padding_left, padding_right)
    return F.pad(x, (0, diff))


def genSpoof_list(
    dir_meta: str, 
    is_train: bool = False, 
    is_eval: bool = False
) -> Union[Tuple[Dict[str, int], List[str]], List[str]]:
    """
    Parses the metadata JSON file to generate file lists and labels.

    Args:
        dir_meta (str): Path to the metadata JSON file.
        is_train (bool): Flag indicating if this is for training.
        is_eval (bool): Flag indicating if this is for evaluation (no labels returned).

    Returns:
        tuple or list: (labels_dict, file_list) if not is_eval, else file_list.
    """
    d_meta = {}
    file_list = []

    with open(dir_meta, 'r') as f:
        data = json.load(f)

    for item in data:
        key = item['file_path']
        file_list.append(key)
        
        if not is_eval:
            # Map 'real' to 1, others (spoof) to 0
            d_meta[key] = 1 if item['label'] == 'real' else 0

    return (d_meta, file_list) if not is_eval else file_list


def eval_to_score_file(score_file: str, key_json_file: str) -> float:
    """
    Calculates EER given a score file and a ground truth JSON file.

    Args:
        score_file (str): Path to the output score file from the model.
        key_json_file (str): Path to the ground truth JSON metadata.

    Returns:
        float: The Equal Error Rate (EER).
    """
    # Load ground truth
    with open(key_json_file, 'r') as f:
        cm_data = json.load(f)
    
    # Load model scores
    # Assuming format: filename|score
    scores = pd.read_csv(score_file, sep='|', header=None)
    
    # Merge on filename to ensure alignment
    cm_df = pd.DataFrame(cm_data)
    merged = scores.merge(cm_df, left_on=0, right_on='file_path', how='inner')
    
    # Extract bona fide (real) and spoof scores
    bona = merged[merged["label"] == "real"][1].values
    spoof = merged[merged["label"] == "fake"][1].values
    
    # Compute EER
    eer = em.compute_eer(bona, spoof)[0]
    print(f"EER: {eer * 100:.2f}%")
    
    return eer


class ADD_Dataset(Dataset):
    def __init__(
        self, 
        args, 
        list_IDs: List[str], 
        labels: Optional[Dict[str, int]], 
        is_eval: bool = False, 
        algo: Optional[int] = None
    ):
        """
        Dataset class for Audio Deepfake Detection.

        Args:
            args: Argument parser namespace containing model settings.
            list_IDs (List[str]): List of file paths.
            labels (Dict[str, int]): Dictionary mapping file paths to labels.
            is_eval (bool): If True, returns (waveform, filepath).
            algo (int, optional): Legacy argument for CPU augmentation (unused now).
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.args = args
        self.is_eval = is_eval
        
        # Determine target sample rate based on model type
        if getattr(args, 'model', '') == 'clap_htsat_fused_aasist':
            self.target_sr = TARGET_SR_48KHZ
        else:
            self.target_sr = TARGET_SR_16KHZ
            
        # Calculate cut length based on ratio relative to 16kHz default
        self.cut = int((MAX_LEN_DEFAULT / TARGET_SR_16KHZ) * self.target_sr)
        
        # Cache for resamplers to avoid initialization overhead
        self.resamplers = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        """Lazy loader for torchaudio Resamplers."""
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = T.Resample(orig_sr, self.target_sr)
        return self.resamplers[orig_sr]

    def __len__(self) -> int:
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, int], Tuple[Tensor, str]]:
        filepath = self.list_IDs[index]
        
        try:
            # 1. Load Audio
            X, fs = torchaudio.load(filepath)
            
            # 2. Convert Stereo to Mono (Fast)
            if X.shape[0] > 1:
                X = X[:1]
            
            # 3. Resample (Only if necessary)
            if fs != self.target_sr:
                X = self._get_resampler(fs)(X)
            
            # 4. Pad or Crop to fixed length
            X = pad_tensor(X, self.cut).squeeze(0)
            
        except Exception as e:
            # Fallback for corrupt files: return silent tensor
            # print(f"Error loading {filepath}: {e}") # Optional logging
            X = torch.zeros(self.cut)

        if self.is_eval:
            return X, filepath
            
        return X, self.labels[filepath]