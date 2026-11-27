import ctypes as C
import math

import torch

from .runtime import configure_grouped, make_inputs, pointer


class Networks:
    def __init__(self, lib, models):
        self.lib = lib
        self.weights = []
        torch.backends.cuda.matmul.allow_tf32 = False
        if "admm_net" in models:
            self.register_admm(models["admm_net"])
        if "detnet" in models:
            self.register_detnet(models["detnet"])
        self.ad = []
        for threads in (64, 128, 256):
            fn = getattr(lib, f"admm_{threads}")
            configure_grouped(fn)
            self.ad.append((threads, fn))
        self.det = lib.det_full
        configure_grouped(self.det)
        lib.plan_det.argtypes = [C.c_int]

    def register_admm(self, model):
        lib = self.lib
        ad = model.cpu()
        levels = ad.levels
        assert torch.allclose(levels, torch.tensor([-3.0, -1.0, 1.0, 3.0]) * levels[2])
        tensors = [
            ad.theta,
            ad.theta * ((levels[2] + levels[3]) / 2),
            ad.delta,
            ad.alpha,
            1 - ad.alpha * ad.beta,
            ad.gamma,
            1 - 2 * ad.beta,
            (levels[1:2] - levels[0:1]) / 2,
        ]
        packed = torch.cat(tensors).detach().contiguous()
        lib.set_admm.argtypes = [C.c_void_p]
        assert lib.set_admm(pointer(packed)) == 0
        ad.cuda()

    def register_detnet(self, det):
        lib = self.lib
        lib.set_det.argtypes = [C.c_int] + [C.c_void_p] * 4 + [C.c_float] * 2

        assert det.residual_alpha == 0.9
        # Fuse the joint-QAM expectation into the frozen output affine map.
        readout = torch.zeros(16, 8, 16, dtype=torch.float64)
        for k in range(8):
            readout[k, k] = det.constellation_real.cpu().double()
            readout[k + 8, k] = det.constellation_imag.cpu().double()
        readout = readout.reshape(16, 128)
        for j, layer in enumerate(det.layers):
            proj = readout @ layer.W2.weight.cpu().double()
            bias = readout @ layer.W2.bias.cpu().double()
            if j < 8:
                proj = torch.cat((proj, layer.W3.weight.cpu().double()))
                bias = torch.cat((bias, layer.W3.bias.cpu().double()))
            wi = layer.W1.weight if j else layer.W1.weight[:, :16]
            w = [
                x.detach().float().contiguous().cuda()
                for x in (wi.T, layer.W1.bias, proj.T, bias)
            ]
            self.weights.extend(w)
            lib.set_det(
                j, *(pointer(x) for x in w), float(layer.delta1), float(layer.delta2)
            )

    def inputs(self, p, m, seed):
        group = make_inputs(p, m, seed, torch.device("cuda"))
        n = p * m
        sizes = [
            p * 1024,
            n * 16,
            n * 80,
            n * 80,
            n * 128,
            ((n * 80 + 63) // 64) * 64 + (8 << 18),
        ]
        return (*group[:4], *(torch.empty(s, device="cuda") for s in sizes))

    def variants(self, name, p, m):
        if name == "detnet":
            candidates = []
            for log_chunks in range(5):
                chunks = 2**log_chunks
                if p % chunks or (p == 128 and p * m // chunks < 4096):
                    continue
                assert self.lib.plan_det(p * m // chunks) == 0
                candidates.extend(
                    (
                        f"cublasLt_fp32_{j}_tiles{chunks}",
                        self.det,
                        1 + j + 4 * log_chunks,
                    )
                    for j in range(4)
                )
            return candidates
        return [
            (f"fused_{t}_threads", fn, b)
            for t, fn in self.ad
            for b in sorted(
                {1, 2, math.ceil(m / (t // 16)), max(1, math.ceil(m / (t // 8)))}
            )
        ]
