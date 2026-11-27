import math

import torch


def pam_constellation(mod_type: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if mod_type == 16:
        values = [-1.0, -3.0, 1.0, 3.0]
        scale = 10.0**-0.5
    elif mod_type == 4:
        values = [-1.0, 1.0]
        scale = 2.0**-0.5
    elif mod_type == 64:
        values = [-3.0, -1.0, -5.0, -7.0, 3.0, 1.0, 5.0, 7.0]
        scale = 42.0**-0.5
    elif mod_type == 256:
        values = [
            -5.0,
            -7.0,
            -3.0,
            -1.0,
            -11.0,
            -9.0,
            -13.0,
            -15.0,
            5.0,
            7.0,
            3.0,
            1.0,
            11.0,
            9.0,
            13.0,
            15.0,
        ]
        scale = 170.0**-0.5
    else:
        raise ValueError("Only square QAM modulations 4/16/64/256 are supported")
    return torch.tensor(values, device=device, dtype=dtype) * scale


def mmse_detect(
    H: torch.Tensor, y: torch.Tensor, noise_var: torch.Tensor
) -> torch.Tensor:
    Ht = H.transpose(1, 2)
    Hty = torch.bmm(Ht, y)
    eye = torch.eye(H.shape[2], device=H.device, dtype=H.dtype).unsqueeze(0)
    A = torch.bmm(Ht, H) + noise_var.view(-1, 1, 1).to(H.dtype) * eye
    return torch.linalg.solve(A, Hty)


def kbest_detect(
    H: torch.Tensor,
    y: torch.Tensor,
    constellation: torch.Tensor,
    k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched real-domain K-best detector matching ``Single/K_Best.m``.

    A reduced QR factorization gives ``z = Q.T @ y`` and the upper-triangular
    system ``R``.  The tree is expanded from the last real coordinate to the
    first.  At each depth the globally best ``k`` partial Euclidean distances
    are retained.  PEDs are updated incrementally; this is algebraically equal
    to the MATLAB implementation's repeated full suffix calculation but avoids
    recomputing already accumulated rows.

    Returns ``(best_estimate, path_indices, path_metrics)``.  Paths are ordered
    as ``[real streams, imaginary streams]`` and metrics are the unnormalised
    squared Euclidean distances ``||z-Rx||^2``.
    """
    if H.dim() != 3 or y.dim() != 3 or y.shape[2] != 1:
        raise ValueError("H and y must have shapes [B,M,N] and [B,M,1]")
    if H.shape[0] != y.shape[0] or H.shape[1] != y.shape[1]:
        raise ValueError("H and y batch/observation dimensions must match")
    if H.shape[1] < H.shape[2]:
        raise ValueError("K-best requires a determined or overdetermined real system")
    if int(k) <= 0:
        raise ValueError("k must be positive")
    const = constellation.to(device=H.device, dtype=H.dtype).reshape(-1)
    if const.numel() < 2:
        raise ValueError("constellation must contain at least two PAM levels")

    Q, R = torch.linalg.qr(H, mode="reduced")
    z = torch.bmm(Q.transpose(1, 2), y).squeeze(-1)
    batch, variables = z.shape
    alphabet = int(const.numel())
    paths = torch.empty(batch, 1, 0, device=H.device, dtype=torch.long)
    path_values = torch.empty(batch, 1, 0, device=H.device, dtype=H.dtype)
    metrics = torch.zeros(batch, 1, device=H.device, dtype=H.dtype)

    for coordinate in range(variables - 1, -1, -1):
        depth = variables - coordinate - 1
        if depth:
            suffix = (
                path_values * R[:, coordinate, coordinate + 1 :].unsqueeze(1)
            ).sum(dim=2)
        else:
            suffix = metrics.new_zeros(batch, metrics.shape[1])
        residual = (
            z[:, coordinate].view(batch, 1, 1)
            - suffix.unsqueeze(2)
            - R[:, coordinate, coordinate].view(batch, 1, 1)
            * const.view(1, 1, alphabet)
        )
        candidates = metrics.unsqueeze(2) + residual.square()
        flat = candidates.flatten(1)
        survivors = min(int(k), int(flat.shape[1]))
        metrics, choices = torch.topk(
            flat, survivors, dim=1, largest=False, sorted=True
        )
        parents = torch.div(choices, alphabet, rounding_mode="floor")
        symbols = choices.remainder(alphabet)
        if depth:
            parent_paths = paths.gather(
                1, parents.unsqueeze(2).expand(batch, survivors, depth)
            )
            parent_values = path_values.gather(
                1, parents.unsqueeze(2).expand(batch, survivors, depth)
            )
        else:
            parent_paths = paths.expand(batch, survivors, 0)
            parent_values = path_values.expand(batch, survivors, 0)
        paths = torch.cat((symbols.unsqueeze(2), parent_paths), dim=2)
        path_values = torch.cat((const[symbols].unsqueeze(2), parent_values), dim=2)

    best = const[paths[:, 0, :]].unsqueeze(2)
    return best, paths, metrics


def kbest_path_bits(path_indices: torch.Tensor, alphabet_size: int) -> torch.Tensor:
    """Map PAM-index paths to physical per-block QAM bits.

    PAM table indices are the MSB-first binary labels used by the repository's
    C++ mapper.  The returned order is per complex stream: real-axis bits then
    imaginary-axis bits, matching ``cpp_modulate_bits``.
    """
    if path_indices.dim() != 3 or path_indices.shape[2] % 2:
        raise ValueError("path_indices must have shape [B,K,2*Nt]")
    alphabet = int(alphabet_size)
    if alphabet < 2:
        raise ValueError("alphabet_size must be at least two")
    bits_axis = int(round(math.log2(alphabet)))
    if 2**bits_axis != alphabet:
        raise ValueError("PAM alphabet size inferred from paths is not a power of two")
    shifts = torch.arange(
        bits_axis - 1, -1, -1, device=path_indices.device, dtype=torch.long
    )
    labels = path_indices.unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(1)
    nt = path_indices.shape[2] // 2
    return torch.cat((labels[:, :, :nt], labels[:, :, nt:]), dim=3).reshape(
        path_indices.shape[0], path_indices.shape[1], 2 * nt * bits_axis
    )


def kbest_bit_llr(
    path_indices: torch.Tensor,
    path_metrics: torch.Tensor,
    noise_var: torch.Tensor,
    alphabet_size: int,
    method: str = "logsumexp",
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Compute bit LLR ``log P(b=1|y)-log P(b=0|y)`` from a K-best list.

    ``method='logsumexp'`` marginalizes every retained path.
    ``method='maxlog'`` uses only the strongest path for each bit value.  If a
    list contains only one value of a bit, the strongest-to-weakest path-metric
    spread supplies a finite signed fallback, as requested for an incomplete
    K-best list.  Since repository ``noise_var`` is complex AWGN variance and
    each real component has variance ``noise_var/2``, the real-domain
    likelihood score is exactly ``-PED/noise_var``.
    """
    if path_indices.dim() != 3 or path_metrics.dim() != 2:
        raise ValueError("paths/metrics must have shapes [B,K,N] and [B,K]")
    if path_indices.shape[:2] != path_metrics.shape:
        raise ValueError("paths and metrics batch/list dimensions must match")
    method = str(method).lower().replace("-", "")
    if method not in {"logsumexp", "maxlog"}:
        raise ValueError("method must be 'logsumexp' or 'maxlog'")
    noise = noise_var.reshape(-1).to(path_metrics).clamp_min(float(eps))
    if noise.numel() == 1:
        noise = noise.expand(path_metrics.shape[0])
    elif noise.numel() != path_metrics.shape[0]:
        raise ValueError("noise_var must be scalar or contain one value per batch item")
    scores = -path_metrics / noise[:, None]
    bits = kbest_path_bits(path_indices, alphabet_size=alphabet_size)
    output = path_metrics.new_empty(path_metrics.shape[0], bits.shape[2])
    spread = (path_metrics.max(dim=1).values - path_metrics.min(dim=1).values) / noise
    negative_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)

    # The loop is only over the fixed modulation-bit positions (32 for the
    # 16x8, 16-QAM system); every frame and every path remain GPU-batched.
    for position in range(bits.shape[2]):
        bit = bits[:, :, position]
        has_zero = bit.eq(0).any(dim=1)
        has_one = bit.eq(1).any(dim=1)
        score_zero = scores.masked_fill(bit.ne(0), negative_inf)
        score_one = scores.masked_fill(bit.ne(1), negative_inf)
        if method == "logsumexp":
            llr = torch.logsumexp(score_one, dim=1) - torch.logsumexp(score_zero, dim=1)
        else:
            llr = score_one.max(dim=1).values - score_zero.max(dim=1).values
        only_one = has_one & ~has_zero
        only_zero = has_zero & ~has_one
        llr = torch.where(only_one, spread, llr)
        llr = torch.where(only_zero, -spread, llr)
        output[:, position] = llr
    return output


def io_lama_detect(
    H: torch.Tensor,
    y: torch.Tensor,
    noise_var: torch.Tensor,
    constellation: torch.Tensor,
    rx_ant_num: int,
    iterations: int = 8,
    damping: float = 1.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Real-equivalent implementation of the IO-LAMA recursion.

    This implements Algorithm 1 in Jeon et al., "Optimality of Large MIMO
    Detection via Approximate Message Passing" (arXiv:1510.06095v1).  ``H``
    and ``y`` use the standard real-equivalent ordering
    ``[real antennas, imag antennas]``.  The scalar denoiser uses the uniform
    PAM prior induced by a uniform square-QAM prior.

    The input generator uses H_ij ~ CN(0, 1), whereas the paper assumes
    H_ij ~ CN(0, 1/Nr).  Scaling both H and y by sqrt(Nr), and the complex
    noise variance by Nr, gives exactly the paper's observation model.  No
    per-realization column normalization is applied.  ``damping=1`` is the
    exact paper recursion.  Values below one apply fixed-point damping to the
    complete candidate state ``(shat, tau, residual)``.

    The returned statistic is the paper's final Gaussian output z^t.  Its
    effective variance is intentionally not folded into the output, so callers
    can retain the repository's existing variance-free soft demodulator.
    """
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    damping = float(damping)
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    batch, real_observations, real_variables = H.shape
    if real_observations != 2 * int(rx_ant_num) or real_variables % 2:
        raise ValueError("H must be a standard real-equivalent MIMO channel")

    dtype = H.dtype
    nt = real_variables // 2
    nr = int(rx_ant_num)
    beta = float(nt) / float(nr)
    normalization = math.sqrt(float(nr))
    H_normalized = H / normalization
    y_normalized = y / normalization
    n0 = noise_var.reshape(batch).to(dtype).clamp_min(eps) / float(nr)

    const = constellation.to(device=H.device, dtype=dtype).reshape(-1)
    prior_mean_axis = const.mean()
    prior_variance_axis = (const - prior_mean_axis).square().mean()
    prior_variance_complex = 2.0 * prior_variance_axis

    shat = prior_mean_axis.expand(batch, real_variables).clone()
    residual = y_normalized.clone()
    tau = beta * prior_variance_complex / n0
    Ht_normalized = H_normalized.transpose(1, 2)
    const_view = const.view(1, 1, -1)
    z = None

    for _ in range(int(iterations)):
        z = shat + torch.bmm(Ht_normalized, residual).squeeze(-1)
        effective_variance = (n0 * (1.0 + tau)).clamp_min(eps)
        scores = -(z.unsqueeze(-1) - const_view).square()
        weights = torch.softmax(scores / effective_variance[:, None, None], dim=-1)

        shat_candidate = (weights * const_view).sum(dim=-1)
        posterior_axis_variance = (
            weights * (const_view - shat_candidate.unsqueeze(-1)).square()
        ).sum(dim=-1)
        posterior_complex_variance = (
            posterior_axis_variance[:, :nt] + posterior_axis_variance[:, nt:]
        ).mean(dim=1)
        tau_candidate = beta * posterior_complex_variance / n0
        onsager = tau_candidate / (1.0 + tau)
        residual_candidate = (
            y_normalized
            - torch.bmm(H_normalized, shat_candidate.unsqueeze(-1))
            + onsager[:, None, None] * residual
        )
        shat = (1.0 - damping) * shat + damping * shat_candidate
        tau = (1.0 - damping) * tau + damping * tau_candidate
        residual = (1.0 - damping) * residual + damping * residual_candidate

    return z.unsqueeze(-1)


def ep_detect(
    H: torch.Tensor,
    y: torch.Tensor,
    noise_var: torch.Tensor,
    constellation: torch.Tensor,
    iterations: int = 5,
    delta: float = 0.5,
    min_variance: float = 5.0e-7,
    eps: float = 1.0e-12,
    return_variance: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Expectation propagation with damped updates and discrete PAM moments."""
    B, _, n_vars = H.shape
    dtype = H.dtype
    Ht = H.transpose(1, 2)
    HtH = torch.bmm(Ht, H)
    Hty = torch.bmm(Ht, y).squeeze(-1)
    noise = noise_var.view(B).to(dtype).clamp_min(eps)

    Alpha = torch.full((B, n_vars), 2.0, device=H.device, dtype=dtype)
    Gamma = torch.zeros(B, n_vars, device=H.device, dtype=dtype)
    Alpha_new = torch.zeros_like(Alpha)
    Gamma_new = torch.zeros_like(Gamma)
    torch.eye(n_vars, device=H.device, dtype=dtype).unsqueeze(0)
    const = constellation.to(device=H.device, dtype=dtype).view(1, 1, -1)

    def posterior(alpha, gamma):
        precision = HtH / noise.view(B, 1, 1) + torch.diag_embed(alpha)
        sigma = torch.linalg.inv(precision)
        rhs = Hty / noise.view(B, 1) + gamma
        mu = torch.bmm(sigma, rhs.unsqueeze(-1)).squeeze(-1)
        return sigma, mu

    Sigma_q, Mu_q = posterior(Alpha, Gamma)
    for _ in range(int(iterations)):
        sig = torch.diagonal(Sigma_q, dim1=-2, dim2=-1).clamp_min(eps)
        denom = (1.0 - sig * Alpha).clamp_min(eps)
        h2 = (sig / denom).clamp_min(eps)
        t = h2 * (Mu_q / sig - Gamma)

        log_prob = -(t.unsqueeze(-1) - const).pow(2) / (2.0 * h2.unsqueeze(-1))
        log_prob = log_prob - log_prob.max(dim=2, keepdim=True).values
        prob = torch.exp(log_prob)
        prob = prob / prob.sum(dim=2, keepdim=True).clamp_min(eps)
        mu_p = (prob * const).sum(dim=2)
        sigma2_p = (prob * const.pow(2)).sum(dim=2) - mu_p.pow(2)
        sigma2_p = sigma2_p.clamp_min(float(min_variance))

        temp_alpha = 1.0 / sigma2_p - 1.0 / h2
        temp_gamma = mu_p / sigma2_p - t / h2
        valid = temp_alpha > float(min_variance)
        Alpha_new = torch.where(valid, temp_alpha, Alpha_new)
        Gamma_new = torch.where(valid, temp_gamma, Gamma_new)

        Alpha = float(delta) * Alpha_new + (1.0 - float(delta)) * Alpha
        Gamma = float(delta) * Gamma_new + (1.0 - float(delta)) * Gamma
        Sigma_q, Mu_q = posterior(Alpha, Gamma)

    mean = Mu_q.unsqueeze(-1)
    if return_variance:
        variance = torch.diagonal(Sigma_q, dim1=-2, dim2=-1).clamp_min(eps)
        return mean, variance
    return mean
