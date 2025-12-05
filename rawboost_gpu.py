import torch
import torch.nn.functional as F
import torch.fft

class RawBoostGPU(torch.nn.Module):
    def __init__(self, algo_id: int = 5, device: str = 'cuda'):
        """
        Args:
            algo_id (int): The algorithm strategy ID.
            device (str): Computation device ('cuda' or 'cpu').
        """
        super().__init__()
        self.algo_id = algo_id
        self.device = device

        # FIR filter for colored noise [1, -0.95, 0.8].
        # Registered as a buffer so it moves with the model (e.g., model.to(device)).
        fir_kernel = torch.tensor([1.0, -0.95, 0.8], device=device).view(1, 1, 3)
        self.register_buffer('fir_kernel', fir_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input batch of shape (Batch, Time).
        
        Returns:
            torch.Tensor: Augmented batch of shape (Batch, Time).
        """
        if self.algo_id == 5:
            x = self._apply_algo_5(x)

        # 20% chance to apply reverb per batch
        if torch.rand(1, device=self.device) < 0.2:
            x = self._apply_reverb(x)

        return x

    def _apply_algo_5(self, x: torch.Tensor) -> torch.Tensor:
        """Applies impulsive and colored noise augmentation."""
        batch_size, time_steps = x.shape
        
        # --- 1. Impulsive Noise ---
        # Generate random SNR between [10, 40]
        snr = torch.rand(batch_size, 1, device=self.device) * 30 + 10
        
        # Create impulsive noise mask (3% of samples)
        impulse_noise = torch.randn_like(x)
        mask = torch.rand_like(x) >= 0.97
        impulse_noise = impulse_noise * mask

        # Calculate scaling factors
        signal_power = x.pow(2).mean(dim=1, keepdim=True)
        noise_power = impulse_noise.pow(2).mean(dim=1, keepdim=True)
        
        scale = torch.sqrt(signal_power / (noise_power + 1e-9)) / (10 ** (snr / 20.0))
        x = x + scale * impulse_noise

        # --- 2. Colored Noise ---
        # Generate white noise and pad for convolution
        white_noise = torch.randn(batch_size, 1, time_steps, device=self.device)
        white_noise_pad = F.pad(white_noise, (2, 0))

        # Apply FIR Filter via Conv1d
        colored_noise = F.conv1d(white_noise_pad, self.fir_kernel)
        colored_noise = colored_noise.view(batch_size, time_steps)

        # Generate new SNR for colored noise
        snr_colored = torch.rand(batch_size, 1, device=self.device) * 30 + 10

        # Calculate scaling factors for colored noise
        colored_noise_power = colored_noise.pow(2).mean(dim=1, keepdim=True)
        current_signal_power = x.pow(2).mean(dim=1, keepdim=True)
        
        scale_colored = torch.sqrt(current_signal_power / (colored_noise_power + 1e-9)) / (10 ** (snr_colored / 20.0))
        
        x = x + scale_colored * colored_noise

        return x

    def _apply_reverb(self, x: torch.Tensor) -> torch.Tensor:
        """Applies synthetic reverb using FFT convolution."""
        batch_size, time_steps = x.shape

        # Generate random RIR length: 0.15s - 0.45s (assuming 16kHz sample rate)
        rir_duration = 0.15 + torch.rand(1).item() * 0.3
        rir_len = int(16000 * rir_duration)

        # Create exponential decay impulse response (IR)
        decay = torch.exp(-torch.linspace(0, 5, rir_len, device=self.device))
        noise = torch.randn(rir_len, device=self.device)
        impulse_response = decay * noise
        
        # Normalize IR
        impulse_response = impulse_response / (torch.norm(impulse_response) + 1e-9)
        impulse_response = impulse_response.view(1, -1)  # Shape: (1, L)

        # FFT Convolution (Frequency Domain)
        n_fft = time_steps + rir_len - 1
        x_fft = torch.fft.rfft(x, n=n_fft)
        ir_fft = torch.fft.rfft(impulse_response, n=n_fft)

        # Broadcast multiply and Inverse FFT
        out_fft = x_fft * ir_fft
        out = torch.fft.irfft(out_fft, n=n_fft)

        # Crop to original length
        out = out[:, :time_steps]

        # Mix original signal with reverb (Wet/Dry mix)
        alpha = torch.rand(batch_size, 1, device=self.device) * 0.25 + 0.1
        x = (1 - alpha) * x + alpha * out

        return x