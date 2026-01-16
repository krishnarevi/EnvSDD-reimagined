import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
from torch import Tensor
from torch.utils.data import Dataset

# --- Local Imports ---
try:
    import eval_metrics as em
except ImportError:
    import sys
    # print("Warning: eval_metrics not found.", file=sys.stderr)


TARGET_SR_16KHZ = 16000
TARGET_SR_48KHZ = 48000
MAX_LEN_DEFAULT = 64600


def pad_tensor(x: Tensor, max_len: int = MAX_LEN_DEFAULT) -> Tensor:
    """Pads or crops a tensor to a fixed length."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[1] >= max_len:
        return x[:, :max_len]
    diff = max_len - x.shape[1]
    return F.pad(x, (0, diff))


def target_augmentation(waveform: Tensor, sample_rate: int) -> Tensor:
    """
    Implements targeted augmentations for robustness against unknown sources.
    1. Codec Augmentation (MP3/Telephony simulation)
    2. Loudness/Gain Normalization
    """
    # 1. Gain Augmentation (Loudness Robustness)
    # Scale volume randomly between -6dB and +6dB
    if random.random() < 0.5:
        gain_db = random.uniform(-6, 6)
        waveform = F_audio.gain(waveform, gain_db)

    # 2. Codec/Compression Augmentation (Unknown Source Robustness)
    # Apply to 50% of samples
    if random.random() < 0.5:
        encoding_type = random.choice(["ulaw", "alaw"])
        # Apply encoding (compression) and immediate decoding
        waveform = F_audio.apply_codec(waveform, sample_rate, format="wav", encoding=encoding_type)

    return waveform


def genSpoof_list(dir_meta: str, is_train: bool = False, is_eval: bool = False) -> Union[Tuple[Dict[str, int], List[str]], List[str]]:
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        data = json.load(f)
    for item in data:
        key = item['file_path']
        file_list.append(key)
        if not is_eval:
            d_meta[key] = 1 if item['label'] == 'real' else 0
    return (d_meta, file_list) if not is_eval else file_list


def eval_to_score_file(score_file: str, key_json_file: str) -> float:
    with open(key_json_file, 'r') as f:
        cm_data = json.load(f)
    scores = pd.read_csv(score_file, sep='|', header=None)
    cm_df = pd.DataFrame(cm_data)
    merged = scores.merge(cm_df, left_on=0, right_on='file_path', how='inner')
    bona = merged[merged["label"] == "real"][1].values
    spoof = merged[merged["label"] == "fake"][1].values
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
        algo: Optional[int] = None,
        augment: bool = False  # <--- NEW FLAG FOR ABLATION
    ):
        self.list_IDs = list_IDs
        self.labels = labels
        self.args = args
        self.is_eval = is_eval
        self.augment = augment # Store the flag

        if getattr(args, 'model', '') == 'clap_htsat_fused_aasist':
            self.target_sr = TARGET_SR_48KHZ
        else:
            self.target_sr = TARGET_SR_16KHZ
            
        self.cut = int((MAX_LEN_DEFAULT / TARGET_SR_16KHZ) * self.target_sr)
        self.resamplers = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = T.Resample(orig_sr, self.target_sr)
        return self.resamplers[orig_sr]

    def __len__(self) -> int:
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, int], Tuple[Tensor, str]]:
        filepath = self.list_IDs[index]
        try:
            X, fs = torchaudio.load(filepath)
            if X.shape[0] > 1:
                X = X[:1]
            if fs != self.target_sr:
                X = self._get_resampler(fs)(X)
            
            X = pad_tensor(X, self.cut).squeeze(0)
            
            # --- APPLY AUGMENTATION HERE ---
            # Only augment if flag is True and we represent training data (labels exist)
            if self.augment and not self.is_eval and self.labels is not None:
                X = target_augmentation(X, self.target_sr)
            # -------------------------------

        except Exception as e:
            X = torch.zeros(self.cut)

        if self.is_eval:
            return X, filepath
            
        return X, self.labels[filepath]