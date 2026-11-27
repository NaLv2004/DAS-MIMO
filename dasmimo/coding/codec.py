from typing import List, Optional, Tuple

import torch

from .crc import append_cb_crc, append_tb_crc
from .params import infer_ldpc_params, load_base_graph


class LDPC5GCodec:
    family = "ldpc5g"
    supports_parity_supervision = True

    def __init__(
        self,
        codeword_length: int,
        code_rate: float,
        device: Optional[torch.device] = None,
        rv: int = 0,
        qm: int = 1,
        nlayers: int = 1,
    ):
        self.params = infer_ldpc_params(
            codeword_length, code_rate, rv=rv, qm=qm, nlayers=nlayers
        )
        self.device = device or torch.device("cpu")
        self.base_graph = torch.as_tensor(
            load_base_graph(self.params.bgn, self.params.set_idx),
            dtype=torch.long,
            device=self.device,
        )
        self._rm_indices: Optional[List[torch.Tensor]] = None
        self._deconcat_slices: Optional[List[slice]] = None

    def to(self, device: torch.device) -> "LDPC5GCodec":
        self.device = device
        self.base_graph = self.base_graph.to(device)
        self._rm_indices = None
        return self

    def random_info_bits(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        device = device or self.device
        return torch.randint(
            0, 2, (batch_size, self.params.info_len), device=device, dtype=torch.long
        )

    def encode(self, info_bits: torch.Tensor) -> torch.Tensor:
        codeblocks = self.segment(info_bits)
        encoded = self.encode_codeblocks(codeblocks)
        return self.rate_match(encoded)

    def segment(self, info_bits: torch.Tensor) -> torch.Tensor:
        p = self.params
        if info_bits.dim() != 2 or info_bits.shape[1] != p.info_len:
            raise ValueError(f"info_bits must have shape [batch, {p.info_len}]")

        bits = info_bits.to(device=self.device, dtype=torch.long)
        tb = append_tb_crc(bits, p.tb_crc_len).to(self.device)
        batch = tb.shape[0]
        codeblocks = torch.full(
            (batch, p.c, p.k), -1, dtype=torch.long, device=self.device
        )

        if p.c == 1:
            codeblocks[:, 0, : p.b_with_crc] = tb
            return codeblocks

        payload_per_cb = p.k_cb_bit - p.tb_crc_len
        padded_tb = torch.zeros(
            batch, payload_per_cb * p.c, dtype=torch.long, device=self.device
        )
        padded_tb[:, : p.b_with_crc] = tb
        chunks = padded_tb.view(batch, p.c, payload_per_cb)
        for cb in range(p.c):
            cb_with_crc = append_cb_crc(chunks[:, cb, :], p.tb_crc_len).to(self.device)
            codeblocks[:, cb, : p.k_cb_bit] = cb_with_crc[:, : p.k_cb_bit]
        return codeblocks

    def encode_codeblocks(self, codeblocks: torch.Tensor) -> torch.Tensor:
        p = self.params
        if codeblocks.dim() != 3 or codeblocks.shape[1:] != (p.c, p.k):
            raise ValueError(f"codeblocks must have shape [batch, {p.c}, {p.k}]")

        batch, c, _ = codeblocks.shape
        flat = codeblocks.to(self.device).reshape(batch * c, p.k)
        encoded_full = self._encode_flat_codeblocks(flat)
        filler_mask = flat < 0
        if filler_mask.any():
            encoded_full[:, : p.k] = torch.where(
                filler_mask,
                torch.full_like(encoded_full[:, : p.k], -1),
                encoded_full[:, : p.k],
            )
        punctured = encoded_full[:, 2 * p.zc :]
        return punctured.view(batch, c, p.n_punctured)

    def _encode_flat_codeblocks(self, flat_codeblocks: torch.Tensor) -> torch.Tensor:
        p = self.params
        bg = torch.remainder(self.base_graph, p.zc)
        bg = torch.where(self.base_graph < 0, torch.full_like(bg, -1), bg)
        batch = flat_codeblocks.shape[0]
        kb = p.k // p.zc
        msg = torch.where(
            flat_codeblocks < 0, torch.zeros_like(flat_codeblocks), flat_codeblocks
        )
        msg = msg.to(torch.long)
        info_vec = msg.view(batch, kb, p.zc).transpose(1, 2).contiguous()

        p1 = bg[:, :kb]
        d = torch.zeros(batch, p.zc, p.mb, dtype=torch.long, device=self.device)
        for row in range(p.mb):
            acc = torch.zeros(batch, p.zc, dtype=torch.long, device=self.device)
            for col in range(kb):
                shift = int(p1[row, col].item())
                if shift >= 0:
                    acc = acc + torch.roll(info_vec[:, :, col], shifts=-shift, dims=1)
            d[:, :, row] = acc

        d0 = d[:, :, :4]
        m1, m2, m3, m4 = self._first_four_parity_blocks(d0)
        tmpm = torch.stack([m1, m2, m3, m4], dim=2)

        p3 = bg[4:, kb : kb + 4]
        parity_tail = torch.zeros(
            batch, p.zc, p.mb - 4, dtype=torch.long, device=self.device
        )
        for row in range(p.mb - 4):
            acc = torch.zeros(batch, p.zc, dtype=torch.long, device=self.device)
            for col in range(4):
                shift = int(p3[row, col].item())
                if shift >= 0:
                    acc = acc + torch.roll(tmpm[:, :, col], shifts=-shift, dims=1)
            parity_tail[:, :, row] = acc + d[:, :, row + 4]

        parity = torch.cat(
            [
                m1,
                m2,
                m3,
                m4,
                parity_tail.transpose(1, 2).reshape(batch, (p.mb - 4) * p.zc),
            ],
            dim=1,
        )
        parity = torch.remainder(parity, 2)
        return torch.cat([msg, parity], dim=1)

    def _first_four_parity_blocks(
        self, d0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.params
        htype_table = ((3, 3, 3, 3, 3, 3, 2, 3), (4, 4, 4, 1, 4, 4, 4, 1))
        htype = htype_table[p.bgn - 1][p.set_idx - 1]

        if htype == 1:
            m1 = d0.sum(dim=2)
            m1tmp = torch.roll(m1, shifts=-1, dims=1)
            m2 = d0[:, :, 0] + m1tmp
            m3 = d0[:, :, 1] + m2
            m4 = d0[:, :, 2] + m1 + m3
        elif htype == 2:
            m1tmp = d0.sum(dim=2)
            shift = 105 % p.zc
            m1 = torch.roll(m1tmp, shifts=shift, dims=1) if shift > 0 else m1tmp
            m2 = d0[:, :, 0] + m1
            m4 = d0[:, :, 3] + m1
            m3 = d0[:, :, 2] + m4
        elif htype == 3:
            m1 = d0.sum(dim=2)
            m1tmp = torch.roll(m1, shifts=-1, dims=1)
            m2 = d0[:, :, 0] + m1tmp
            m3 = d0[:, :, 1] + m1 + m2
            m4 = d0[:, :, 2] + m3
        else:
            m1 = torch.roll(d0.sum(dim=2), shifts=1, dims=1)
            m2 = d0[:, :, 0] + m1
            m3 = d0[:, :, 1] + m2
            m4 = d0[:, :, 3] + m1

        return (
            torch.remainder(m1, 2),
            torch.remainder(m2, 2),
            torch.remainder(m3, 2),
            torch.remainder(m4, 2),
        )

    def rate_match(self, encoded_punctured: torch.Tensor) -> torch.Tensor:
        p = self.params
        if encoded_punctured.dim() != 3 or encoded_punctured.shape[1:] != (
            p.c,
            p.n_punctured,
        ):
            raise ValueError(
                f"encoded_punctured must have shape [batch, {p.c}, {p.n_punctured}]"
            )

        indices = self.rate_match_indices()
        parts = []
        for cb, idx in enumerate(indices):
            gathered = encoded_punctured[:, cb, :].index_select(1, idx)
            if p.qm != 1:
                gathered = self._bit_interleave(gathered, p.qm)
            parts.append(gathered)
        return torch.cat(parts, dim=1)

    def rate_recover(
        self, decoder_llr: torch.Tensor, filler_llr: float = 1.0e6
    ) -> torch.Tensor:
        p = self.params
        if decoder_llr.dim() != 2 or decoder_llr.shape[1] != p.out_len:
            raise ValueError(f"decoder_llr must have shape [batch, {p.out_len}]")

        llr = decoder_llr.to(self.device)
        batch = llr.shape[0]
        recovered = torch.zeros(
            batch, p.c, p.n_full, dtype=llr.dtype, device=self.device
        )
        offset = 0
        indices = self.rate_match_indices()
        for cb, e in enumerate(p.e_array):
            part = llr[:, offset : offset + e]
            offset += e
            if p.qm != 1:
                part = self._bit_deinterleave(part, p.qm)

            temp = torch.zeros(
                batch, p.n_punctured, dtype=llr.dtype, device=self.device
            )
            if p.filler_end_punctured > p.filler_start_punctured:
                temp[:, p.filler_start_punctured : p.filler_end_punctured] = filler_llr
            temp.scatter_add_(1, indices[cb].expand(batch, -1), part)
            recovered[:, cb, 2 * p.zc :] = temp
        return recovered

    def desegment_llr(self, codeblock_llr: torch.Tensor) -> torch.Tensor:
        p = self.params
        if codeblock_llr.dim() != 3 or codeblock_llr.shape[1:] != (p.c, p.n_full):
            raise ValueError(
                f"codeblock_llr must have shape [batch, {p.c}, {p.n_full}]"
            )

        if p.c == 1:
            return codeblock_llr[:, 0, : p.info_len]

        payload_per_cb = p.k_cb_bit - p.tb_crc_len
        parts = [codeblock_llr[:, cb, :payload_per_cb] for cb in range(p.c)]
        return torch.cat(parts, dim=1)[:, : p.info_len]

    def rate_match_indices(self) -> List[torch.Tensor]:
        if self._rm_indices is None:
            p = self.params
            indices = []
            filler_start = p.filler_start_punctured
            filler_end = p.filler_end_punctured
            for cb, e in enumerate(p.e_array):
                out = []
                j = 0
                while len(out) < e:
                    idx = (p.k0[cb] + j) % p.n_punctured
                    j += 1
                    if filler_start <= idx < filler_end:
                        continue
                    out.append(idx)
                indices.append(
                    torch.as_tensor(out, dtype=torch.long, device=self.device)
                )
            self._rm_indices = indices
        return self._rm_indices

    @staticmethod
    def _bit_interleave(bits: torch.Tensor, qm: int) -> torch.Tensor:
        e = bits.shape[1]
        if e % qm != 0:
            raise ValueError("E must be divisible by Qm")
        return (
            bits.view(bits.shape[0], qm, e // qm)
            .transpose(1, 2)
            .reshape(bits.shape[0], e)
        )

    @staticmethod
    def _bit_deinterleave(bits: torch.Tensor, qm: int) -> torch.Tensor:
        e = bits.shape[1]
        if e % qm != 0:
            raise ValueError("E must be divisible by Qm")
        return (
            bits.view(bits.shape[0], e // qm, qm)
            .transpose(1, 2)
            .reshape(bits.shape[0], e)
        )
