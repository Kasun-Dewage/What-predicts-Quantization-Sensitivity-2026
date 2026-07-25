import math
import torch
import torch.nn.functional as F


def rtn_quantize(W, bits, group_size=0):
    W_f = W.float()
    m, n = W_f.shape
    qmax = 2 ** (bits - 1) - 1
    if group_size is None or group_size <= 0 or group_size >= n:
        maxval = W_f.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = maxval / qmax
        W_q = (W_f / scale).round().clamp(-qmax, qmax) * scale
        return W_q.to(W.dtype)
    pad_n = ((n + group_size - 1) // group_size) * group_size
    if pad_n != n:
        pad = torch.zeros(m, pad_n - n, dtype=W_f.dtype, device=W_f.device)
        W_pad = torch.cat([W_f, pad], dim=1)
    else:
        W_pad = W_f
    W_g = W_pad.reshape(m, -1, group_size)
    maxval = W_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = maxval / qmax
    W_q_g = (W_g / scale).round().clamp(-qmax, qmax) * scale
    W_q = W_q_g.reshape(m, -1)[:, :n]
    return W_q.to(W.dtype)


def gptq_quantize(W, H, bits, group_size=128, percdamp=0.01, act_order=False):
    orig_dtype = W.dtype
    device = W.device
    W = W.detach().clone().float()
    H = H.detach().clone().float().to(device)
    rows, cols = W.shape
    qmax = 2 ** (bits - 1) - 1
    dead = torch.diag(H) == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    damp = percdamp * torch.mean(torch.diag(H)).clamp(min=1e-8).item()
    idx = torch.arange(cols, device=device)
    H[idx, idx] = H[idx, idx] + damp
    L = None
    for attempt in range(4):
        try:
            L = torch.linalg.cholesky(H)
            break
        except Exception:
            H[idx, idx] = H[idx, idx] + damp * (10 ** attempt)
    if L is None:
        return rtn_quantize(W.to(orig_dtype), bits, group_size if group_size > 0 else 128)
    Hinv_full = torch.cholesky_inverse(L)
    try:
        Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
    except Exception:
        return rtn_quantize(W.to(orig_dtype), bits, group_size if group_size > 0 else 128)
    Q = torch.zeros_like(W)
    gsize = group_size if (group_size and group_size > 0) else cols
    for i1 in range(0, cols, gsize):
        i2 = min(i1 + gsize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        maxval = W1.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        s_vec = (maxval / qmax).squeeze(1)
        ncol = i2 - i1
        for j in range(ncol):
            w = W1[:, j]
            d = Hinv1[j, j]
            q = (w / s_vec).round().clamp(-qmax, qmax) * s_vec
            Q1[:, j] = q
            err = (w - q) / d
            if j + 1 < ncol:
                W1[:, j + 1:] = W1[:, j + 1:] - err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            Err1[:, j] = err
        Q[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:] = W[:, i2:] - Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        invperm = torch.argsort(perm)
        Q = Q[:, invperm]
    del H, Hinv, Hinv_full, L
    return Q.to(orig_dtype)


def awq_quantize(W, x_abs, bits, group_size=128, n_grid=20):
    orig_dtype = W.dtype
    device = W.device
    W_f = W.detach().float()
    x_abs = x_abs.detach().float().to(device).clamp(min=1e-6)
    w_abs = W_f.abs().mean(dim=0).clamp(min=1e-6)
    best_loss = float("inf")
    best_s = torch.ones_like(x_abs)
    for k in range(n_grid):
        alpha = k / max(1, n_grid - 1)
        s_raw = x_abs.pow(alpha) / w_abs.pow(1.0 - alpha)
        s = s_raw / s_raw.mean().clamp(min=1e-8)
        s = s.clamp(min=1e-4)
        W_s = W_f * s.unsqueeze(0)
        W_q_s = rtn_quantize(W_s, bits, group_size)
        W_deq = W_q_s.float() / s.unsqueeze(0)
        diff = (W_f - W_deq) * x_abs.unsqueeze(0)
        loss = float((diff * diff).sum().item())
        if loss < best_loss:
            best_loss = loss
            best_s = s
    W_s = W_f * best_s.unsqueeze(0)
    W_q_s = rtn_quantize(W_s, bits, group_size)
    W_deq = W_q_s.float() / best_s.unsqueeze(0)
    return W_deq.to(orig_dtype)


def _hadamard(n, device):
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def quip_quantize(W, bits, group_size=0, seed=0):
    orig_dtype = W.dtype
    device = W.device
    W_f = W.detach().float()
    m, n = W_f.shape
    n_pad = 1 << max(1, (n - 1).bit_length())
    pad = n_pad - n
    if pad > 0:
        W_p = F.pad(W_f, (0, pad))
    else:
        W_p = W_f
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    s_in = (torch.randint(0, 2, (n_pad,), device=device, generator=g, dtype=torch.float32) * 2 - 1)
    s_out = (torch.randint(0, 2, (m,), device=device, generator=g, dtype=torch.float32) * 2 - 1)
    Hmat = _hadamard(n_pad, device) / math.sqrt(n_pad)
    W_rot = (W_p * s_in.unsqueeze(0)) @ Hmat
    W_rot = W_rot * s_out.unsqueeze(1)
    W_q_rot = rtn_quantize(W_rot, bits, group_size)
    W_back = W_q_rot.float() * s_out.unsqueeze(1)
    W_back = W_back @ Hmat.T
    W_back = W_back * s_in.unsqueeze(0)
    if pad > 0:
        W_back = W_back[:, :n]
    del Hmat
    return W_back.to(orig_dtype)


def quantize_dispatch(method, W, bits, group_size=0, stats=None, seed=0):
    method = (method or "rtn").lower()
    if method == "rtn":
        return rtn_quantize(W, bits, group_size)
    if method == "gptq":
        if stats is None or stats.get("H", None) is None:
            return rtn_quantize(W, bits, group_size if group_size > 0 else 128)
        gs = group_size if group_size and group_size > 0 else 128
        return gptq_quantize(W, stats["H"], bits, group_size=gs)
    if method == "awq":
        if stats is None or stats.get("x_abs", None) is None:
            return rtn_quantize(W, bits, group_size if group_size > 0 else 128)
        gs = group_size if group_size and group_size > 0 else 128
        return awq_quantize(W, stats["x_abs"], bits, group_size=gs)
    if method == "quip":
        return quip_quantize(W, bits, group_size, seed=seed)
    raise ValueError("Unknown quant method: {}".format(method))


def quantize_slice_dispatch(method, W_full, s0, s1, bits, group_size=0, stats=None, seed=0):
    W_slice = W_full[s0:s1]
    sub_stats = None
    if stats is not None:
        sub_stats = {}
        if "H" in stats and stats["H"] is not None:
            sub_stats["H"] = stats["H"]
        if "x_abs" in stats and stats["x_abs"] is not None:
            sub_stats["x_abs"] = stats["x_abs"]
    return quantize_dispatch(method, W_slice, bits, group_size, sub_stats, seed)
