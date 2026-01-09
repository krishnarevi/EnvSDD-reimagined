import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from transformers import AutoModel
from .w2v2_aasist import Residual_block, GraphAttentionLayer, GraphPool, HtrgGraphAttentionLayer

class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        self.device = device
        
        # 1. Load EAT-Large
        self.model_id = "worstchan/EAT-large_epoch20_finetune_AS2M"
        print(f"[EAT-LRG] Loading EAT-Large from: {self.model_id}")
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self.model.eval() 
        
        # EAT-Large output dimension is 1024
        self.out_dim = 1024
        self.norm_mean = -4.268
        self.norm_std = 4.569
        
        # Hook Setup (Unchanged)
        self._hook_features = None
        target_module = None
        
        search_paths = [['eat', 'encoder', 'layer'], ['base_model', 'encoder', 'layer'], ['blocks'], ['layers'], ['model', 'blocks']]
        for path in search_paths:
            curr = self.model
            found = True
            for part in path:
                if hasattr(curr, part): curr = getattr(curr, part)
                else: found = False; break
            if found and isinstance(curr, nn.ModuleList):
                target_module = curr[-1]
                break
        
        if target_module is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList):
                    target_module = module[-1]

        if target_module is None:
            raise RuntimeError("Could not locate Transformer Blocks in EAT model.")

        target_module.register_forward_hook(self._hook_fn)
        self.layer_weights = nn.Parameter(torch.ones(1)) 

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple): self._hook_features = output[0]
        else: self._hook_features = output

    def _preprocess_batch(self, waveforms):
        """
        Matches EAT-Base logic using Kaldi fbank.
        """
        mel_batch = []
        for wav in waveforms:
            # 1. Zero Mean
            wav = wav - wav.mean()
            
            # 2. Kaldi Fbank (Strict compliance params)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            
            mel = kaldi.fbank(
                wav,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            ) 
            
            # 3. Padding / Trimming to multiple of 16
            n_frames = mel.shape[0]
            target_len = ((n_frames + 15) // 16) * 16
            
            diff = target_len - n_frames
            if diff > 0:
                mel = F.pad(mel, (0, 0, 0, diff)) 
            elif diff < 0:
                mel = mel[:target_len, :]
            
            mel_batch.append(mel)
            
        mels = torch.stack(mel_batch)
        mels = (mels - self.norm_mean) / (self.norm_std * 2)
        mels = mels.unsqueeze(1) # (Batch, 1, Time, Freq)
        
        return mels.to(self.device)

    def extract_feat(self, x):
        # x shape: (Batch, Time)
        
        # --- 1. Correct Preprocessing ---
        with torch.no_grad():
             # x comes in as (Batch, Time), needs to be list or split for loop if on CPU
             # but _preprocess_batch handles tensor iteration
             mels = self._preprocess_batch(x)

        # --- 2. Forward Pass ---
        self._hook_features = None
        try:
            _ = self.model(mels)
        except Exception:
            pass 
            
        if self._hook_features is None:
            raise RuntimeError("Hook failed.")
            
        features = self._hook_features
        if features.dim() == 2: features = features.unsqueeze(1)
        return features

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]

        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128) 

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
        )
        
        self.pos_S = nn.Parameter(torch.randn(1, 64, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # x: (Batch, Samples) - Squeeze last dim if (Batch, Samples, 1) or similar
        if x.dim() == 3: x = x.squeeze(-1)
            
        x_ssl_feat = self.ssl_model.extract_feat(x)
        x = self.LL(x_ssl_feat) 
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1) 
        
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        w = self.attention(x)
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) 
        if e_S.shape[1] != self.pos_S.shape[1]:
            pos_S_resized = F.interpolate(self.pos_S.transpose(1, 2), size=e_S.shape[1], mode='linear', align_corners=False).transpose(1, 2)
            e_S = e_S + pos_S_resized
        else:
            e_S = e_S + self.pos_S 
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return output