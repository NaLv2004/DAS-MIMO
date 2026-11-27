import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .baselines import classical
from .baselines.admm_net import ADMMNet
from .baselines.channelnet import ChannelNetMLP
from .baselines.detnet import ComplexQAMDetNet
from .coding.channel import CodedMIMODataGenerator
from .coding.demod import (
    apply_cpp_deinterleaver,
    apply_cpp_interleaver,
    cpp_soft_demod_real_mimo,
    cpp_symbol_llr_to_decoder_llr,
)
from .coding.factory import build_decoder_aware_decoder
from .coding.tasks import code_descriptor
from .model import DASDetector

ROOT = Path(__file__).resolve().parents[1]
DAS = ("das_sparse", "das_dense")
LEARNED = DAS + ("admm_net", "detnet", "channelnet")
DETECTORS = LEARNED + ("mmse", "amp8", "amp15", "ep3", "ep7", "kbest8")


def config(name, device):
    path = ROOT / "configs" / (name + ".json" if name in DAS else "das_sparse.json")
    cfg = SimpleNamespace(
        **json.loads(path.read_text()), DEVICE=torch.device(device), dtype=torch.float32
    )
    cfg.DECODER_AWARE_CODEWORD_LENGTH = 1024
    cfg.DECODER_AWARE_CODE_RATE = 0.5
    return cfg


def detector(name, cfg, weights=None):
    pam = classical.pam_constellation(16).sort().values
    if name in DAS:
        model = DASDetector(cfg)
    elif name == "detnet":
        model = ComplexQAMDetNet(2 * cfg.Nt, pam, num_layers=10)
    elif name == "admm_net":
        model = ADMMNet(2 * cfg.Nt, 2 * cfg.Nr, pam, num_layers=10)
    elif name == "channelnet":
        model = ChannelNetMLP(feature_dim=10, iterations=10, classes=4)
    elif name in DETECTORS:
        return None
    else:
        raise ValueError(name)
    if weights:
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
    return model.to(cfg.DEVICE)


class Link:
    def __init__(self, cfg, rate, snr):
        self.cfg = SimpleNamespace(**vars(cfg))
        self.cfg.DECODER_AWARE_CODE_RATE = rate
        self.snr = snr
        self.rate = rate
        self.generator = CodedMIMODataGenerator(self.cfg)
        self.codec = self.generator.codec
        self.decoder = build_decoder_aware_decoder(self.cfg, self.codec).to(cfg.DEVICE)
        self.levels = classical.pam_constellation(16, device=cfg.DEVICE)
        self.scale = cfg.DECODER_AWARE_LLR_SCALE

    def descriptor(self, cfg):
        p = self.codec.params
        return code_descriptor(
            1024,
            self.rate,
            p.bgn,
            p.zc,
            1024,
            1024,
            cfg.MULTICODE_CODE_RATE_MIN,
            cfg.MULTICODE_CODE_RATE_MAX,
        )

    def materialize(self, compact):
        H, y, A, x, s, info, bits = self.generator.materialize_compact_latent_batch(
            compact, self.snr
        )
        n, blocks = H.shape[:2]
        return (
            H.flatten(0, 1),
            y.flatten(0, 1),
            A.flatten(0, 1),
            x.flatten(0, 1),
            s.flatten(0, 1),
            info,
            bits,
            n,
            blocks,
        )

    def bit_scores(self, x, n):
        return cpp_soft_demod_real_mimo(x, nt=self.cfg.Nt, mode_type=4).reshape(n, -1)

    def decode(self, scores):
        prior = apply_cpp_deinterleaver(
            cpp_symbol_llr_to_decoder_llr(scores, scale=self.scale), None
        )
        return self.codec.desegment_llr(self.decoder(self.codec.rate_recover(prior)))

    def output(self, name, model, batch, cfg, threshold=0.01):
        H, y, A, x, s, info, bits, n, blocks = batch
        if name in DAS:
            trajectory = model(
                H,
                y,
                A,
                s,
                use_threshold=not model.training,
                threshold=threshold,
                code_descriptor=self.descriptor(cfg),
            )
            return trajectory[-1], trajectory
        if name in LEARNED:
            result = model(H, y)
            return result.final, result
        gram = H.transpose(1, 2).bmm(H)
        noise = (
            (A * s.reshape(-1, 1, 1) - gram)
            .diagonal(dim1=1, dim2=2)
            .mean(1)
            .clamp_min(1e-12)
        )
        if name == "mmse":
            return classical.mmse_detect(H, y, noise), None
        if name.startswith("amp"):
            return classical.io_lama_detect(
                H,
                y,
                noise,
                self.levels,
                rx_ant_num=cfg.Nr,
                iterations=int(name[3:]),
                damping=0.5,
            ), None
        if name.startswith("ep"):
            return classical.ep_detect(
                H, y, noise, self.levels, iterations=int(name[2:]), delta=0.5
            ), None
        _, paths, metrics = classical.kbest_detect(H, y, self.levels, k=8)
        scores = classical.kbest_bit_llr(
            paths, metrics, noise, alphabet_size=4, method="maxlog"
        )
        return None, scores.reshape(n, -1) / self.scale

    def information_llr(self, name, output, n):
        x, aux = output
        return self.decode(aux if name == "kbest8" else self.bit_scores(x, n))

    def training_loss(self, name, model, batch, cfg, bank, epoch, baseline_weights):
        x, aux = self.output(name, model, batch, cfg)
        _, _, _, truth, _, info, bits, n, _ = batch
        llr = self.decode(self.bit_scores(x, n))
        signed = llr * (1 - 2 * info.to(llr))
        logp = F.logsigmoid(2 * signed)
        decoder = (
            -(1 - logp.exp()).clamp_min(1e-6).pow(cfg.DECODER_AWARE_FOCAL_GAMMA) * logp
        ).mean()
        if name in DAS:
            targets = apply_cpp_interleaver(bits, None).to(llr)
            perf = sum(
                0.8 ** (len(aux) - i - 1)
                * F.binary_cross_entropy_with_logits(
                    2 * self.bit_scores(layer, n), targets
                )
                for i, layer in enumerate(aux)
            )
            penalties = []
            for block in (*model.phis, *model.psis):
                for node in block.nodes:
                    for kind, indices in node.valid_ops_map.items():
                        weights = node.op_weights[kind]
                        cost = weights.new_tensor(
                            [
                                cfg.HIGH_COMPLEXITY_OP_L1_WEIGHT
                                if node.ops_registry[i]["name"]
                                in {"Mul_H", "Mul_HT", "Mul_A", "Mul_Ainv"}
                                else 1
                                for i in indices
                            ]
                        )
                        penalties.append((weights.abs() * cost).sum())
            structure = (
                torch.stack(penalties).sum()
                * cfg.SPARSITY_LAMBDA
                * min((epoch - 1) / max(cfg.SPARSITY_RAMP_EPOCHS - 1, 1), 1.0)
            )
            regularization, _ = model.code_architecture_regularization(bank)
            return (
                decoder
                + math.exp(-(epoch - 1) / 100) * perf
                + structure
                + regularization
            )
        perf = (
            model.paper_loss(aux, truth, model.levels)
            if name == "channelnet"
            else model.paper_loss(aux, truth)
        )
        return baseline_weights[0] * perf + baseline_weights[1] * decoder
