import math

import torch
from torch.utils.data import TensorDataset

from .demod import (
    apply_cpp_interleaver,
    apply_fixed_random_interleaver,
    cpp_modulate_bits,
    fixed_random_interleaver_indices,
    mode_type_from_mod_type,
)
from .factory import build_decoder_aware_codec
from .spatial_channel import SpatialChannelSampler


class CodedMIMODataGenerator:
    def __init__(self, config):
        self.cfg = config
        self.device = config.DEVICE
        self.dtype = config.dtype
        self.mode_type = int(
            getattr(
                config,
                "DECODER_AWARE_MODE_TYPE",
                mode_type_from_mod_type(config.ModType),
            )
        )
        self.codec = build_decoder_aware_codec(config)
        self.interleaver_rows = getattr(config, "DECODER_AWARE_INTERLEAVER_ROWS", None)
        self.random_interleaver_enabled = bool(
            getattr(config, "DECODER_AWARE_RANDOM_INTERLEAVER_ENABLED", False)
        )
        if self.random_interleaver_enabled:
            self.random_interleaver, self.random_deinterleaver = (
                fixed_random_interleaver_indices(
                    self.codec.params.out_len,
                    int(
                        getattr(
                            config, "DECODER_AWARE_RANDOM_INTERLEAVER_SEED", 20260811
                        )
                    ),
                    device=self.device,
                )
            )
        else:
            self.random_interleaver = self.random_deinterleaver = None
        self.bits_per_mimo_block = self.mode_type * int(config.Nt)
        self.allow_mimo_padding = bool(
            getattr(config, "DECODER_AWARE_ALLOW_MIMO_PADDING", False)
        )
        if (
            self.codec.params.out_len % self.bits_per_mimo_block != 0
            and not self.allow_mimo_padding
        ):
            raise ValueError(
                "DECODER_AWARE_CODEWORD_LENGTH must be divisible by "
                f"mode_type * Nt = {self.bits_per_mimo_block}"
            )
        self.blocks_per_frame = math.ceil(
            self.codec.params.out_len / self.bits_per_mimo_block
        )
        self.transmitted_bits_per_frame = (
            self.blocks_per_frame * self.bits_per_mimo_block
        )
        self.mimo_padding_bits = (
            self.transmitted_bits_per_frame - self.codec.params.out_len
        )
        self.csi_estimation_nmse = float(getattr(config, "CSI_ESTIMATION_NMSE", 0.0))
        if not 0.0 <= self.csi_estimation_nmse <= 2.0:
            raise ValueError("CSI_ESTIMATION_NMSE must lie in [0, 2]")
        self.csi_estimation_rho = 1.0 - 0.5 * self.csi_estimation_nmse
        self.channel = SpatialChannelSampler(config)

    def pad_physical_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Append deterministic Bernoulli-distributed dummy bits for MIMO fill.

        Reusing the leading physical bits avoids online randomness when a
        compact fixed dataset is rematerialized.  Decoder and detector losses
        crop the corresponding LLRs back to the true codeword length.
        """
        if self.mimo_padding_bits == 0:
            return bits
        if bits.dim() != 2 or bits.shape[1] != self.codec.params.out_len:
            raise ValueError(
                "physical bits must have shape "
                f"[batch, {self.codec.params.out_len}] before MIMO padding"
            )
        return torch.cat([bits, bits[:, : self.mimo_padding_bits]], dim=1)

    def _sample_channel_real_equivalent(self, batch_size: int) -> torch.Tensor:
        channel = self.channel.sample_complex(batch_size)
        real, imag = channel.real.to(self.dtype), channel.imag.to(self.dtype)
        return torch.cat(
            [torch.cat([real, -imag], dim=2), torch.cat([imag, real], dim=2)], dim=1
        )

    def _sample_true_channel_from_estimate(
        self, estimate: torch.Tensor
    ) -> torch.Tensor:
        """Draw the conditional true channel for the configured CSI NMSE.

        Both the estimate and the resulting true channel retain the configured
        channel marginal.  With ``rho = 1 - NMSE/2``, the expected normalized
        squared difference is exactly the requested NMSE.
        """
        if self.csi_estimation_nmse <= 0.0:
            return estimate
        innovation = self._sample_channel_real_equivalent(int(estimate.shape[0]))
        rho = float(self.csi_estimation_rho)
        return rho * estimate + math.sqrt(max(0.0, 1.0 - rho * rho)) * innovation

    def create_compact_latent_dataset_frames(
        self, num_frames: int, eb_n0_db: float
    ) -> TensorDataset:
        """Pre-generate a fixed coded dataset in compact CPU storage.

        Only random latent quantities are stored: complex H components, AWGN,
        information bits and rate-matched bits.  H is stored once rather than
        duplicated in real-equivalent form, and floating tensors use fp16
        storage.  Materialization performs deterministic algebra only; it never
        draws data online and therefore every epoch traverses the same samples.
        """
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        frame_batch = int(
            getattr(
                self.cfg,
                "MULTICODE_CREATE_FRAME_BATCH_SIZE",
                getattr(self.cfg, "DECODER_AWARE_CREATE_FRAME_BATCH_SIZE", 64),
            )
        )
        if frame_batch <= 0:
            raise ValueError("MULTICODE_CREATE_FRAME_BATCH_SIZE must be positive")

        channel_components = []
        csi_innovation_components = []
        noise_values = []
        info_values = []
        tx_values = []
        generated = 0
        while generated < num_frames:
            current = min(frame_batch, num_frames - generated)
            info_bits = self.codec.random_info_bits(current, device=self.device)
            tx_bits = self.codec.encode(info_bits)
            blocks = self.blocks_per_frame
            flat_blocks = current * blocks
            channel = self.channel.sample_complex(flat_blocks)
            components = torch.view_as_real(channel).reshape(
                current, blocks, int(self.cfg.Nr), int(self.cfg.Nt), 2
            )
            if self.csi_estimation_nmse > 0.0:
                innovation = self.channel.sample_complex(flat_blocks)
                csi_innovation_components.append(
                    torch.view_as_real(innovation)
                    .reshape(current, blocks, int(self.cfg.Nr), int(self.cfg.Nt), 2)
                    .to(dtype=torch.float16)
                    .cpu()
                )

            bits_per_symbol = math.log2(self.cfg.ModType)
            code_rate = float(self.codec.params.code_rate)
            sigma2 = (int(self.cfg.Nt) * int(self.cfg.Nr)) / (
                10 ** (float(eb_n0_db) / 10.0)
                * code_rate
                * bits_per_symbol
                * int(self.cfg.Nt)
            )
            noise_std = math.sqrt(sigma2 / 2.0)
            noise = (
                torch.randn(
                    flat_blocks,
                    2 * int(self.cfg.Nr),
                    1,
                    device=self.device,
                    dtype=self.dtype,
                )
                * noise_std
            )
            noise = noise.reshape(current, blocks, 2 * int(self.cfg.Nr), 1)

            channel_components.append(components.to(dtype=torch.float16).cpu())
            noise_values.append(noise.to(dtype=torch.float16).cpu())
            info_values.append(info_bits.to(dtype=torch.uint8).cpu())
            tx_values.append(tx_bits.to(dtype=torch.uint8).cpu())
            generated += current

        tensors = [
            torch.cat(channel_components, dim=0),
            torch.cat(noise_values, dim=0),
            torch.cat(info_values, dim=0),
            torch.cat(tx_values, dim=0),
        ]
        if csi_innovation_components:
            tensors.append(torch.cat(csi_innovation_components, dim=0))
        return TensorDataset(*tensors)

    def materialize_compact_latent_batch(
        self, batch, eb_n0_db: float, *, compute_auxiliary: bool = True
    ):
        """Materialize channels, received vectors, labels and the normalized Gram matrix."""
        if len(batch) == 4:
            components, noise, info_bits, tx_bits = batch
            csi_innovation = None
        elif len(batch) == 5:
            components, noise, info_bits, tx_bits, csi_innovation = batch
            if self.csi_estimation_nmse <= 0.0:
                raise ValueError(
                    "A CSI innovation tensor requires CSI_ESTIMATION_NMSE > 0"
                )
        else:
            raise ValueError("compact latent batches must contain 4 or 5 tensors")
        components = components.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        if csi_innovation is not None:
            csi_innovation = csi_innovation.to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
        noise = noise.to(device=self.device, dtype=self.dtype, non_blocking=True)
        info_bits = info_bits.to(
            device=self.device, dtype=torch.long, non_blocking=True
        )
        tx_bits = tx_bits.to(device=self.device, dtype=torch.long, non_blocking=True)

        frames, blocks = components.shape[:2]
        estimated_real = components[..., 0].reshape(
            frames * blocks, int(self.cfg.Nr), int(self.cfg.Nt)
        )
        estimated_imag = components[..., 1].reshape(
            frames * blocks, int(self.cfg.Nr), int(self.cfg.Nt)
        )
        H = torch.cat(
            [
                torch.cat([estimated_real, -estimated_imag], dim=2),
                torch.cat([estimated_imag, estimated_real], dim=2),
            ],
            dim=1,
        )
        if csi_innovation is None:
            true_H = H
        else:
            rho = float(self.csi_estimation_rho)
            innovation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
            innovation_real = csi_innovation[..., 0].reshape(
                frames * blocks, int(self.cfg.Nr), int(self.cfg.Nt)
            )
            innovation_imag = csi_innovation[..., 1].reshape(
                frames * blocks, int(self.cfg.Nr), int(self.cfg.Nt)
            )
            true_real = rho * estimated_real + innovation_scale * innovation_real
            true_imag = rho * estimated_imag + innovation_scale * innovation_imag
            true_H = torch.cat(
                [
                    torch.cat([true_real, -true_imag], dim=2),
                    torch.cat([true_imag, true_real], dim=2),
                ],
                dim=1,
            )

        mod_bits = apply_fixed_random_interleaver(tx_bits, self.random_interleaver)
        mod_bits = apply_cpp_interleaver(mod_bits, self.interleaver_rows)
        mod_bits = self.pad_physical_bits(mod_bits)
        symbols = cpp_modulate_bits(mod_bits, mode_type=self.mode_type).to(
            device=self.device, dtype=self.dtype
        )
        real_x = symbols[:, :, 0].reshape(frames, blocks, int(self.cfg.Nt))
        imag_x = symbols[:, :, 1].reshape(frames, blocks, int(self.cfg.Nt))
        x_gt = torch.cat([real_x, imag_x], dim=2).unsqueeze(-1)
        x_flat = x_gt.reshape(frames * blocks, 2 * int(self.cfg.Nt), 1)
        noise_flat = noise.reshape(frames * blocks, 2 * int(self.cfg.Nr), 1)
        # The physical observation uses the true channel, while every detector
        # and all detector-derived auxiliaries receive only its estimate H.
        y = torch.bmm(true_H, x_flat) + noise_flat

        if compute_auxiliary:
            bits_per_symbol = math.log2(self.cfg.ModType)
            code_rate = float(self.codec.params.code_rate)
            sigma2 = (int(self.cfg.Nt) * int(self.cfg.Nr)) / (
                10 ** (float(eb_n0_db) / 10.0)
                * code_rate
                * bits_per_symbol
                * int(self.cfg.Nt)
            )
            eye = torch.eye(
                2 * int(self.cfg.Nt), device=self.device, dtype=self.dtype
            ).unsqueeze(0)
            A = torch.bmm(H.transpose(1, 2), H) + float(sigma2) * eye
            scaling = torch.max(
                torch.sum(torch.abs(A), dim=2), dim=1, keepdim=True
            ).values.unsqueeze(2)
            A_norm = A / scaling
        else:
            A_norm = None
            scaling = None

        H = H.reshape(frames, blocks, 2 * int(self.cfg.Nr), 2 * int(self.cfg.Nt))
        y = y.reshape(frames, blocks, 2 * int(self.cfg.Nr), 1)
        if compute_auxiliary:
            A_norm = A_norm.reshape(
                frames, blocks, 2 * int(self.cfg.Nt), 2 * int(self.cfg.Nt)
            )
            scaling = scaling.squeeze(2).reshape(frames, blocks, 1)
        return H, y, A_norm, x_gt, scaling, info_bits, tx_bits
