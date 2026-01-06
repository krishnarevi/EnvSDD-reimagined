import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from transformers import AutoModel
from .w2v2_aasist import Residual_block, GraphAttentionLayer, GraphPool, HtrgGraphAttentionLayer

# ==========================================
# EAT Front-End Wrapper
# ==========================================
class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        self.device = device
        self.model_id = "worstchan/EAT-base_epoch30_finetune_AS2M"
        
        print(f"Loading EAT model from: {self.model_id}")
        # trust_remote_code=True is required as EAT is a custom architecture on Hub
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self.model.eval() # Start in eval mode
        self.model.to(device)
        
        # EAT-base output dimension is 768
        self.out_dim = 768 

        # Normalization stats from EAT documentation
        self.norm_mean = -4.268
        self.norm_std = 4.569
        self.target_length = 1024 # Recommended by docs, but we adapt logic for 4s

    def _preprocess_batch(self, waveforms):
        """
        Mimics the exact preprocessing from EAT repository:
        Waveform -> Zero Mean -> Kaldi Fbank -> Pad (mod 16) -> Normalize
        """
        mel_batch = []
        
        # torchaudio.compliance.kaldi.fbank processes single waveforms best.
        # We loop over the batch. Since batch size is usually 16-32, this is acceptable.
        for wav in waveforms:
            # 1. Zero Mean (as per snippet)
            wav = wav - wav.mean()
            
            # 2. Kaldi Fbank (Strict compliance params from snippet)
            # Input to kaldi.fbank must be (1, time)
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
            ) # Output: (Time, Freq)
            
            # 3. Padding / Trimming to multiple of 16
            n_frames = mel.shape[0]
            # Calculate target length: round up to nearest 16
            # For 4s (approx 400 frames), this ensures compatibility.
            target_len = ((n_frames + 15) // 16) * 16
            
            diff = target_len - n_frames
            if diff > 0:
                # Pad (left, right, top, bottom) -> (0, 0, 0, diff) for (Time, Freq)
                # Note: PyTorch pad args are (last_dim_left, last_dim_right, 2nd_last_left...)
                # So for (Time, Freq), we want to pad Time dimension.
                # Actually, snippet uses ZeroPad2d which takes (left, right, top, bottom).
                # Here we treat (Time, Freq) as image-like if unsqueezed.
                # Let's simple use F.pad for the Time dimension (dim 0)
                mel = F.pad(mel, (0, 0, 0, diff)) 
            elif diff < 0:
                mel = mel[:target_len, :]
            
            mel_batch.append(mel)
            
        # Stack: (Batch, Time, Freq)
        mels = torch.stack(mel_batch)
        
        # 4. Normalize (as per snippet)
        mels = (mels - self.norm_mean) / (self.norm_std * 2)
        
        # 5. EAT expects (Batch, 1, Time, Freq)
        mels = mels.unsqueeze(1)
        
        return mels.to(self.device)

    def extract_feat(self, input_data):
        # input_data: (Batch, Length) raw waveform
        
        # 1. Preprocess (Wave -> Mel -> Norm)
        with torch.no_grad():
            x_mel = self._preprocess_batch(input_data)
        
        # 2. Forward Pass
        # The EAT model typically returns a dictionary or tensor.
        # Based on snippet: model.extract_features(mel)
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)
            
        features = self.model.extract_features(x_mel)
        
        # Features shape check. 
        # If granularities='all', it might include CLS.
        # We need (Batch, Time, Dim) for AASIST.
        # If output includes CLS (usually index 0 or separate), we might want to slice it.
        # However, for AASIST graph temporal nodes, keeping CLS as a "global node" 
        # is actually beneficial, or we can treat it as just another frame.
        
        return features

# ==========================================
# AASIST Back-End
# ==========================================
class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]

        # EAT Front-end
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
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
        
        # Position Encoding
        # EAT @ 4s (approx 400 frames) -> Pooling -> AASIST Graph
        # We need to allow dynamic position encoding or set it large enough.
        # Standard AASIST uses fixed size. We initialize safely.
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
        # x: (Batch, Samples)
        # 1. Extract EAT features
        # Output shape: (Batch, Frames, 768)
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        
        # 2. Projection
        x = self.LL(x_ssl_feat) # (Batch, Frames, 128)
        
        # 3. Format for AASIST (Batch, 1, Dim, Frames)
        x = x.transpose(1, 2)   # (Batch, 128, Frames)
        x = x.unsqueeze(dim=1)  # (Batch, 1, 128, Frames)
        
        # 4. Encoder
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        # Spectral SA
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) 
        
        # Dynamic Positional Encoding Adjustment
        # If the frame count mismatches the pre-defined pos_S, we interpolate
        if e_S.shape[1] != self.pos_S.shape[1]:
            pos_S_resized = F.interpolate(
                self.pos_S.transpose(1, 2), 
                size=e_S.shape[1], 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            e_S = e_S + pos_S_resized
        else:
            e_S = e_S + self.pos_S 
        
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        
        # Temporal SA
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # Learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Inference 2
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

        # Readout
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return output

if __name__ == "__main__":
    # Sanity check
    model = Model(None, 'cuda' if torch.cuda.is_available() else 'cpu')
    print("Model Loaded.")
    # Create 4-second dummy audio
    x = torch.rand(2, 64000).to(model.device) # 4s * 16k
    y = model(x)
    print("Output shape:", y.shape)