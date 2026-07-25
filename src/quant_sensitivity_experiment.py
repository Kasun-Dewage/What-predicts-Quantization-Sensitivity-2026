import math
import json
import os
import re
import gc
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from quant_methods import (
    rtn_quantize,
    quantize_dispatch,
)
from downstream_eval import run_downstream, summarize_downstream


MODELS = OrderedDict([
    ("llama1-7b",    {"hf_name": "huggyllama/llama-7b",              "arch": "llama"}),
    ("llama2-7b",    {"hf_name": "meta-llama/Llama-2-7b-hf",         "arch": "llama"}),
    ("llama3-8b",    {"hf_name": "meta-llama/Meta-Llama-3-8B",       "arch": "llama"}),
    ("llama3.2-1b",  {"hf_name": "meta-llama/Llama-3.2-1B",          "arch": "llama"}),
    ("mistral-7b",   {"hf_name": "mistralai/Mistral-7B-v0.1",        "arch": "llama"}),
    ("qwen2.5-1.5b", {"hf_name": "Qwen/Qwen2.5-1.5B",                "arch": "llama"}),
    ("qwen2.5-7b",   {"hf_name": "Qwen/Qwen2.5-7B",                  "arch": "llama"}),
    ("opt-1.3b",     {"hf_name": "facebook/opt-1.3b",                "arch": "opt"}),
    ("opt-6.7b",     {"hf_name": "facebook/opt-6.7b",                "arch": "opt"}),
    ("opt-13b",      {"hf_name": "facebook/opt-13b",                 "arch": "opt"}),
    ("gpt-j-6b",     {"hf_name": "EleutherAI/gpt-j-6B",              "arch": "opt"}),
    ("falcon-7b",    {"hf_name": "tiiuae/falcon-7b",                 "arch": "falcon"}),
    ("pythia-6.9b",  {"hf_name": "EleutherAI/pythia-6.9b",           "arch": "neox"}),
])


COMP_ORDER = ["Q", "K", "V", "O"]


def parse_layer_index(name):
    for pat in [r"layers\.(\d+)\.", r"layer\.(\d+)\.", r"\.h\.(\d+)\."]:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return -1


def compute_mp_metrics(W):
    W_2d = W.detach().float().view(W.shape[0], -1)
    m, n = W_2d.shape
    gamma = max(m, n) / min(m, n)
    S = torch.linalg.svdvals(W_2d)
    sq = S ** 2
    sigma_sq = float(sq.median()) / (1.0 + gamma)
    lambda_plus = sigma_sq * (1.0 + math.sqrt(gamma)) ** 2
    outlier_mask = sq > lambda_plus
    n_outliers = int(outlier_mask.sum().item())
    total_energy = float(sq.sum())
    outlier_energy = float(sq[outlier_mask].sum()) if total_energy > 1e-12 else 0.0
    energy_ratio = outlier_energy / total_energy if total_energy > 1e-12 else 0.0
    W_np = W_2d.cpu().numpy()
    std = float(W_np.std())
    if std > 1e-12:
        entry_mask = np.abs(W_np) > 4.0 * std
    else:
        entry_mask = np.zeros(W_np.shape, dtype=bool)
    entry_outlier_count = int(entry_mask.sum())
    entry_total = int(entry_mask.size)
    entry_outlier_density = entry_outlier_count / entry_total if entry_total > 0 else 0.0
    w_flat = W_np.ravel()
    kurt = float(sp_stats.kurtosis(w_flat, fisher=True))
    return {
        "n_outliers": n_outliers,
        "energy_ratio": energy_ratio,
        "lambda_plus": float(lambda_plus),
        "entry_outlier_count": entry_outlier_count,
        "entry_outlier_density": entry_outlier_density,
        "kurtosis": kurt,
        "shape": [m, n],
        "gamma": gamma,
    }


def relative_recon_error(W_orig, W_q):
    diff_norm = (W_orig.float() - W_q.float()).norm(p="fro")
    orig_norm = W_orig.float().norm(p="fro").clamp(min=1e-12)
    return float(diff_norm / orig_norm)


def hessian_weighted_recon_error(W_orig, W_q, input_var):
    if input_var is None:
        return float("nan")
    delta = (W_orig.float() - W_q.float())
    iv = torch.as_tensor(input_var, device=delta.device, dtype=delta.dtype)
    iv = iv.clamp(min=0.0)
    weighted_sq = (delta * delta) * iv.unsqueeze(0)
    m = delta.shape[0]
    val = float(weighted_sq.sum() / max(1, m))
    return val


def get_eval_tokens(tokenizer, n_tokens, device, dataset_name="wikitext2"):
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif dataset_name == "c4":
        try:
            ds = load_dataset(
                "allenai/c4", "en", split="validation",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}
            )
            target_docs = 2000 if (n_tokens is None or n_tokens <= 0 or n_tokens >= 131072) else 2000
            text = "\n\n".join(ds["text"][:target_docs])
        except Exception:
            ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            buffer = []
            total = 0
            cap = n_tokens if (n_tokens and n_tokens > 0) else 131072
            for ex in ds:
                buffer.append(ex["text"])
                total += len(ex["text"])
                if total > 10 * cap:
                    break
            text = "\n\n".join(buffer)
    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        text = "\n\n".join(ds["sentence"])
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
    enc = tokenizer(text, return_tensors="pt")
    if n_tokens is None or n_tokens <= 0:
        ids = enc.input_ids.to(device)
    else:
        ids = enc.input_ids[:, :n_tokens].to(device)
    return ids


def get_calibration_tokens(tokenizer, n_tokens, device, source="c4"):
    if source == "c4":
        try:
            ds = load_dataset(
                "allenai/c4", "en", split="train",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"}
            )
            text = "\n\n".join(ds["text"][:4000])
        except Exception:
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            buffer = []
            total = 0
            cap = max(n_tokens, 2048)
            for ex in ds:
                buffer.append(ex["text"])
                total += len(ex["text"])
                if total > 10 * cap:
                    break
            text = "\n\n".join(buffer)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[:, :n_tokens].to(device)
    return ids


def compute_perplexity(model, input_ids, block_size=1024):
    model.eval()
    seq_len = input_ids.shape[1]
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, seq_len - 1, block_size):
            end = min(start + block_size + 1, seq_len)
            chunk = input_ids[:, start:end]
            if chunk.shape[1] < 2:
                continue
            out = model(chunk, use_cache=False)
            logits = out.logits[:, :-1, :].float().contiguous()
            targets = chunk[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += targets.numel()
            del out, logits, targets
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def get_proj_info_llama(model):
    entries = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index(name)
        if layer_idx < 0:
            continue
        comp = None
        if name.endswith(".q_proj"):
            comp = "Q"
        elif name.endswith(".k_proj"):
            comp = "K"
        elif name.endswith(".v_proj"):
            comp = "V"
        elif name.endswith(".o_proj"):
            comp = "O"
        if comp is not None:
            entries.append((layer_idx, comp, name, module))
    entries.sort(key=lambda x: (x[0], COMP_ORDER.index(x[1])))
    return entries


def get_proj_info_falcon(model):
    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    head_dim = hidden // n_heads
    n_kv = getattr(cfg, "num_kv_heads", None) or getattr(cfg, "n_head_kv", None) or 1
    q_size = n_heads * head_dim
    k_size = n_kv * head_dim
    v_size = n_kv * head_dim
    entries = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index(name)
        if layer_idx < 0:
            continue
        if name.endswith(".query_key_value"):
            entries.append({
                "layer_idx": layer_idx,
                "kind": "fused_qkv",
                "name": name,
                "module": module,
                "q_size": q_size,
                "k_size": k_size,
                "v_size": v_size,
            })
        elif ".self_attention.dense" in name and name.endswith(".dense"):
            entries.append({
                "layer_idx": layer_idx,
                "kind": "O",
                "name": name,
                "module": module,
            })
    entries.sort(key=lambda x: x["layer_idx"])
    return entries


def get_proj_info_opt(model):
    entries = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index(name)
        if layer_idx < 0:
            continue
        comp = None
        if name.endswith(".q_proj"):
            comp = "Q"
        elif name.endswith(".k_proj"):
            comp = "K"
        elif name.endswith(".v_proj"):
            comp = "V"
        elif name.endswith(".out_proj"):
            comp = "O"
        if comp is not None:
            entries.append((layer_idx, comp, name, module))
    entries.sort(key=lambda x: (x[0], COMP_ORDER.index(x[1])))
    return entries


def get_proj_info_neox(model):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    head_dim = hidden // n_heads
    q_size = n_heads * head_dim
    entries = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index(name)
        if layer_idx < 0:
            continue
        if name.endswith(".query_key_value"):
            entries.append({
                "layer_idx": layer_idx,
                "kind": "fused_qkv",
                "name": name,
                "module": module,
                "q_size": q_size,
                "k_size": q_size,
                "v_size": q_size,
            })
        elif name.endswith(".dense"):
            entries.append({
                "layer_idx": layer_idx,
                "kind": "O",
                "name": name,
                "module": module,
            })
    entries.sort(key=lambda x: x["layer_idx"])
    return entries


def capture_calibration_stats(model, module_set, input_ids, block_size,
                              need_var=True, need_H=False, need_abs=False):
    var_sum = {}
    H_sum = {}
    abs_sum = {}
    n_samples = {}
    handles = []

    def make_hook(key):
        def hook(module, inputs, outputs):
            x = inputs[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() != 2:
                return
            x_f = x.detach().float()
            if need_var:
                sq = (x_f * x_f).sum(dim=0).cpu()
                if key in var_sum:
                    var_sum[key] = var_sum[key] + sq
                else:
                    var_sum[key] = sq
            if need_H:
                h = (x_f.t() @ x_f).cpu()
                if key in H_sum:
                    H_sum[key] = H_sum[key] + h
                else:
                    H_sum[key] = h
            if need_abs:
                a = x_f.abs().sum(dim=0).cpu()
                if key in abs_sum:
                    abs_sum[key] = abs_sum[key] + a
                else:
                    abs_sum[key] = a
            n_samples[key] = n_samples.get(key, 0) + x_f.shape[0]
        return hook

    for key, module in module_set.items():
        handles.append(module.register_forward_hook(make_hook(key)))

    model.eval()
    with torch.no_grad():
        seq_len = input_ids.shape[1]
        for start in range(0, seq_len - 1, block_size):
            end = min(start + block_size + 1, seq_len)
            chunk = input_ids[:, start:end]
            if chunk.shape[1] < 2:
                continue
            model(chunk, use_cache=False)

    for h in handles:
        h.remove()

    stats = {}
    for key in n_samples:
        n = max(1, n_samples[key])
        item = {}
        if need_var and key in var_sum:
            item["input_var"] = (var_sum[key] / float(n)).numpy()
        if need_H and key in H_sum:
            item["H"] = H_sum[key] / float(n)
        if need_abs and key in abs_sum:
            item["x_abs"] = abs_sum[key] / float(n)
        stats[key] = item

    del var_sum, H_sum, abs_sum
    gc.collect()
    return stats


def collect_modules_for_hooks(arch, proj_info):
    module_set = {}
    if arch in ("llama", "opt"):
        for layer_idx, comp, mod_name, module in proj_info:
            module_set[mod_name] = module
    else:
        seen = set()
        for item in proj_info:
            if item["name"] in seen:
                continue
            seen.add(item["name"])
            module_set[item["name"]] = item["module"]
    return module_set


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _layer_fully_done(entry, bits_list):
    if entry is None:
        return False
    for b in bits_list:
        if "ppl_delta_{}bit".format(b) not in entry:
            return False
    return True


def run_sensitivity_llama(model, proj_info, input_ids, bits_list, block_size,
                          baseline_ppl, ppl_cap_mult, group_size, method,
                          calib_stats, cpu_offload, eval_tokens_map,
                          results_existing, chk_path, chk_payload):
    results = OrderedDict()
    if results_existing:
        for k, v in results_existing.items():
            results[k] = dict(v) if isinstance(v, dict) else v
    n_total = len(proj_info) * len(bits_list)
    done = 0
    for layer_idx, comp, mod_name, module in proj_info:
        key = str(layer_idx)
        if key not in results:
            results[key] = {}
        existing_entry = results[key].get(comp, None)
        if _layer_fully_done(existing_entry, bits_list):
            done += len(bits_list)
            print("  [skip] L{} {} already done".format(layer_idx, comp))
            continue
        W_orig_gpu = module.weight.data.clone()
        mp = compute_mp_metrics(W_orig_gpu)
        entry = {
            "mp_n_outliers": mp["n_outliers"],
            "mp_energy_ratio": mp["energy_ratio"],
            "mp_lambda_plus": mp["lambda_plus"],
            "entry_outlier_count": mp["entry_outlier_count"],
            "entry_outlier_density": mp["entry_outlier_density"],
            "kurtosis": mp["kurtosis"],
            "shape": mp["shape"],
            "gamma": mp["gamma"],
        }
        if existing_entry is not None:
            for k2, v2 in existing_entry.items():
                entry[k2] = v2
        stats = calib_stats.get(mod_name, {}) if calib_stats else {}
        iv = stats.get("input_var", None)
        if cpu_offload:
            W_orig_store = W_orig_gpu.cpu()
            del W_orig_gpu
            W_orig_for_calc = W_orig_store
        else:
            W_orig_store = W_orig_gpu
            W_orig_for_calc = W_orig_gpu
        for bits in bits_list:
            if "ppl_delta_{}bit".format(bits) in entry:
                done += 1
                continue
            W_src = W_orig_for_calc.to(module.weight.device, non_blocking=True) if cpu_offload else W_orig_for_calc
            q_stats = {"H": stats.get("H", None), "x_abs": stats.get("x_abs", None)}
            W_q = quantize_dispatch(method, W_src, bits, group_size, q_stats, seed=layer_idx * 4 + COMP_ORDER.index(comp))
            recon_err = relative_recon_error(W_src, W_q)
            h_sens = hessian_weighted_recon_error(W_src, W_q, iv) if iv is not None else float("nan")
            module.weight.data.copy_(W_q)
            ppl_q = compute_perplexity(model, input_ids, block_size)
            ppl_q_eval = {}
            if eval_tokens_map:
                for ds_name, ids in eval_tokens_map.items():
                    if ds_name == "wikitext2":
                        continue
                    p = compute_perplexity(model, ids, block_size)
                    ppl_q_eval[ds_name] = p
            module.weight.data.copy_(W_src)
            if cpu_offload:
                del W_src
            del W_q
            if ppl_q > baseline_ppl * ppl_cap_mult or not math.isfinite(ppl_q):
                ppl_q = float("nan")
            ppl_delta = ppl_q - baseline_ppl if math.isfinite(ppl_q) else float("nan")
            entry["recon_error_{}bit".format(bits)] = recon_err
            entry["hessian_sens_{}bit".format(bits)] = h_sens
            entry["ppl_quant_{}bit".format(bits)] = ppl_q
            entry["ppl_delta_{}bit".format(bits)] = ppl_delta
            for ds_name, p in ppl_q_eval.items():
                entry["ppl_quant_{}_{}bit".format(ds_name, bits)] = p
            done += 1
            print("  [{}/{}] L{:>2} {} {}bit  m={}  gamma={:.2f}  recon={:.5f}  hsens={:.4e}  ppl={:.3f}  delta={:+.3f}".format(
                done, n_total, layer_idx, comp, bits, method,
                mp["gamma"], recon_err, h_sens,
                ppl_q if math.isfinite(ppl_q) else -1,
                ppl_delta if math.isfinite(ppl_delta) else float("nan"),
            ))
        if not cpu_offload:
            del W_orig_store
        results[key][comp] = entry
        if chk_path:
            chk_payload["layers"] = {k: v for k, v in results.items()}
            save_checkpoint(chk_path, chk_payload)
        torch.cuda.empty_cache()
    return results


def run_sensitivity_falcon(model, proj_info, input_ids, bits_list, block_size,
                           baseline_ppl, ppl_cap_mult, group_size, method,
                           calib_stats, cpu_offload, eval_tokens_map,
                           results_existing, chk_path, chk_payload):
    results = OrderedDict()
    if results_existing:
        for k, v in results_existing.items():
            results[k] = dict(v) if isinstance(v, dict) else v
    sub_items = []
    for item in proj_info:
        if item["kind"] == "fused_qkv":
            q_s, k_s, v_s = item["q_size"], item["k_size"], item["v_size"]
            W_full_shape = item["module"].weight.shape[0]
            if W_full_shape == q_s + k_s + v_s:
                slices = [("Q", 0, q_s), ("K", q_s, q_s + k_s), ("V", q_s + k_s, q_s + k_s + v_s)]
            else:
                third = W_full_shape // 3
                slices = [("Q", 0, third), ("K", third, 2 * third), ("V", 2 * third, W_full_shape)]
            for comp, s0, s1 in slices:
                sub_items.append((item["layer_idx"], comp, item["name"], item["module"], s0, s1))
        else:
            W_shape = item["module"].weight.shape[0]
            sub_items.append((item["layer_idx"], "O", item["name"], item["module"], 0, W_shape))
    sub_items.sort(key=lambda x: (x[0], COMP_ORDER.index(x[1])))
    n_total = len(sub_items) * len(bits_list)
    done = 0
    for layer_idx, comp, mod_name, module, s0, s1 in sub_items:
        key = str(layer_idx)
        if key not in results:
            results[key] = {}
        existing_entry = results[key].get(comp, None)
        if _layer_fully_done(existing_entry, bits_list):
            done += len(bits_list)
            print("  [skip] L{} {} already done".format(layer_idx, comp))
            continue
        W_full_orig_gpu = module.weight.data.clone()
        W_slice_gpu = W_full_orig_gpu[s0:s1].clone()
        mp = compute_mp_metrics(W_slice_gpu)
        entry = {
            "mp_n_outliers": mp["n_outliers"],
            "mp_energy_ratio": mp["energy_ratio"],
            "mp_lambda_plus": mp["lambda_plus"],
            "entry_outlier_count": mp["entry_outlier_count"],
            "entry_outlier_density": mp["entry_outlier_density"],
            "kurtosis": mp["kurtosis"],
            "shape": mp["shape"],
            "gamma": mp["gamma"],
        }
        if existing_entry is not None:
            for k2, v2 in existing_entry.items():
                entry[k2] = v2
        stats = calib_stats.get(mod_name, {}) if calib_stats else {}
        iv = stats.get("input_var", None)
        if cpu_offload:
            W_full_store = W_full_orig_gpu.cpu()
            W_slice_store = W_slice_gpu.cpu()
            del W_full_orig_gpu, W_slice_gpu
        else:
            W_full_store = W_full_orig_gpu
            W_slice_store = W_slice_gpu
        for bits in bits_list:
            if "ppl_delta_{}bit".format(bits) in entry:
                done += 1
                continue
            W_slice = W_slice_store.to(module.weight.device, non_blocking=True) if cpu_offload else W_slice_store
            W_full = W_full_store.to(module.weight.device, non_blocking=True) if cpu_offload else W_full_store
            q_stats = {"H": stats.get("H", None), "x_abs": stats.get("x_abs", None)}
            W_q_slice = quantize_dispatch(method, W_slice, bits, group_size, q_stats, seed=layer_idx * 4 + COMP_ORDER.index(comp))
            recon_err = relative_recon_error(W_slice, W_q_slice)
            h_sens = hessian_weighted_recon_error(W_slice, W_q_slice, iv) if iv is not None else float("nan")
            W_patched = W_full.clone()
            W_patched[s0:s1] = W_q_slice
            module.weight.data.copy_(W_patched)
            ppl_q = compute_perplexity(model, input_ids, block_size)
            ppl_q_eval = {}
            if eval_tokens_map:
                for ds_name, ids in eval_tokens_map.items():
                    if ds_name == "wikitext2":
                        continue
                    p = compute_perplexity(model, ids, block_size)
                    ppl_q_eval[ds_name] = p
            module.weight.data.copy_(W_full)
            del W_q_slice, W_patched
            if cpu_offload:
                del W_slice, W_full
            if ppl_q > baseline_ppl * ppl_cap_mult or not math.isfinite(ppl_q):
                ppl_q = float("nan")
            ppl_delta = ppl_q - baseline_ppl if math.isfinite(ppl_q) else float("nan")
            entry["recon_error_{}bit".format(bits)] = recon_err
            entry["hessian_sens_{}bit".format(bits)] = h_sens
            entry["ppl_quant_{}bit".format(bits)] = ppl_q
            entry["ppl_delta_{}bit".format(bits)] = ppl_delta
            for ds_name, p in ppl_q_eval.items():
                entry["ppl_quant_{}_{}bit".format(ds_name, bits)] = p
            done += 1
            print("  [{}/{}] L{:>2} {} {}bit  m={}  gamma={:.2f}  recon={:.5f}  hsens={:.4e}  ppl={:.3f}  delta={:+.3f}".format(
                done, n_total, layer_idx, comp, bits, method,
                mp["gamma"], recon_err, h_sens,
                ppl_q if math.isfinite(ppl_q) else -1,
                ppl_delta if math.isfinite(ppl_delta) else float("nan"),
            ))
        if not cpu_offload:
            del W_full_store, W_slice_store
        results[key][comp] = entry
        if chk_path:
            chk_payload["layers"] = {k: v for k, v in results.items()}
            save_checkpoint(chk_path, chk_payload)
        torch.cuda.empty_cache()
    return results


def find_dominant_component(layer_results, bits=3):
    totals = {"Q": 0.0, "K": 0.0, "V": 0.0, "O": 0.0}
    counts = {"Q": 0, "K": 0, "V": 0, "O": 0}
    for layer_key, comps in layer_results.items():
        for comp, info in comps.items():
            if comp not in totals:
                continue
            delta = info.get("ppl_delta_{}bit".format(bits), float("nan"))
            if math.isfinite(delta) and delta > 0:
                totals[comp] += delta
                counts[comp] += 1
    dominant = max(totals, key=lambda k: totals[k])
    return dominant, totals, counts


def bootstrap_dominance_ci(layer_results, bits=3, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    per_layer_pos = []
    for layer_key, comps in layer_results.items():
        row = {"Q": 0.0, "K": 0.0, "V": 0.0, "O": 0.0}
        for comp, info in comps.items():
            if comp not in row:
                continue
            d = info.get("ppl_delta_{}bit".format(bits), float("nan"))
            if math.isfinite(d) and d > 0:
                row[comp] = d
        per_layer_pos.append(row)
    n = len(per_layer_pos)
    if n == 0:
        return {"Q": (0, 0, 0), "K": (0, 0, 0), "V": (0, 0, 0), "O": (0, 0, 0)}
    shares = {c: [] for c in COMP_ORDER}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tot = {c: 0.0 for c in COMP_ORDER}
        for i in idx:
            r = per_layer_pos[i]
            for c in COMP_ORDER:
                tot[c] += r[c]
        S = sum(tot.values())
        if S <= 0:
            continue
        for c in COMP_ORDER:
            shares[c].append(tot[c] / S)
    out = {}
    for c in COMP_ORDER:
        if shares[c]:
            arr = np.array(shares[c])
            out[c] = (float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        else:
            out[c] = (0.0, 0.0, 0.0)
    return out


def _apply_quant_llama(proj_info, bit_map, group_size, method, calib_stats):
    orig_list = []
    for layer_idx, comp, mod_name, module in proj_info:
        W_orig = module.weight.data.clone()
        orig_list.append((module, W_orig))
        bits = bit_map.get(comp, None)
        if bits is None:
            continue
        stats = calib_stats.get(mod_name, {}) if calib_stats else {}
        q_stats = {"H": stats.get("H", None), "x_abs": stats.get("x_abs", None)}
        W_q = quantize_dispatch(method, W_orig, bits, group_size, q_stats, seed=layer_idx * 4 + COMP_ORDER.index(comp))
        module.weight.data.copy_(W_q)
        del W_q
    return orig_list


def _apply_quant_falcon(proj_info, bit_map, group_size, method, calib_stats):
    orig_list = []
    for item in proj_info:
        module = item["module"]
        mod_name = item["name"]
        W_full_orig = module.weight.data.clone()
        orig_list.append((module, W_full_orig))
        stats = calib_stats.get(mod_name, {}) if calib_stats else {}
        q_stats = {"H": stats.get("H", None), "x_abs": stats.get("x_abs", None)}
        if item["kind"] == "fused_qkv":
            q_s, k_s, v_s = item["q_size"], item["k_size"], item["v_size"]
            W_full_shape = W_full_orig.shape[0]
            if W_full_shape == q_s + k_s + v_s:
                slices = [("Q", 0, q_s), ("K", q_s, q_s + k_s), ("V", q_s + k_s, q_s + k_s + v_s)]
            else:
                third = W_full_shape // 3
                slices = [("Q", 0, third), ("K", third, 2 * third), ("V", 2 * third, W_full_shape)]
            W_new = W_full_orig.clone()
            for comp, s0, s1 in slices:
                bits = bit_map.get(comp, None)
                if bits is None:
                    continue
                W_new[s0:s1] = quantize_dispatch(method, W_full_orig[s0:s1], bits, group_size, q_stats,
                                                 seed=item["layer_idx"] * 4 + COMP_ORDER.index(comp))
            module.weight.data.copy_(W_new)
            del W_new
        else:
            bits = bit_map.get("O", None)
            if bits is None:
                continue
            W_q = quantize_dispatch(method, W_full_orig, bits, group_size, q_stats,
                                    seed=item["layer_idx"] * 4 + COMP_ORDER.index("O"))
            module.weight.data.copy_(W_q)
            del W_q
    return orig_list


def _restore_weights(orig_list):
    for module, W_orig in orig_list:
        module.weight.data.copy_(W_orig)


def run_mp_experiment(model, proj_info, arch, input_ids, block_size,
                      baseline_ppl, dominant_comp, ppl_cap_mult, group_size,
                      eval_tokens_map, method, calib_stats,
                      downstream_cfg=None, tokenizer=None):
    allocations = []
    allocations.append(("uniform_4bit", {"Q": 4, "K": 4, "V": 4, "O": 4}))
    allocations.append(("uniform_3bit", {"Q": 3, "K": 3, "V": 3, "O": 3}))
    ca = {c: 3 for c in COMP_ORDER}
    ca[dominant_comp] = 4
    allocations.append(("ca_{}_at_4_rest_3".format(dominant_comp), ca))
    inv = {c: 4 for c in COMP_ORDER}
    inv[dominant_comp] = 3
    allocations.append(("inv_{}_at_3_rest_4".format(dominant_comp), inv))

    apply_fn = _apply_quant_llama if arch in ("llama", "opt") else _apply_quant_falcon

    results = {}
    for name, bit_map in allocations:
        orig_list = apply_fn(proj_info, bit_map, group_size, method, calib_stats)
        ppl_q = compute_perplexity(model, input_ids, block_size)
        extra_ppls = {}
        if eval_tokens_map:
            for ds_name, ids in eval_tokens_map.items():
                if ds_name == "wikitext2":
                    continue
                extra_ppls[ds_name] = compute_perplexity(model, ids, block_size)
        downstream = {}
        if downstream_cfg is not None and downstream_cfg.get("tasks"):
            downstream = run_downstream(
                model, tokenizer,
                tasks=downstream_cfg["tasks"],
                batch_size=downstream_cfg.get("batch_size", 8),
                num_fewshot=downstream_cfg.get("num_fewshot", 0),
                limit=downstream_cfg.get("limit", None),
                mmlu_fewshot=downstream_cfg.get("mmlu_fewshot", 5),
            )
        _restore_weights(orig_list)
        del orig_list
        gc.collect()
        torch.cuda.empty_cache()
        if ppl_q > baseline_ppl * ppl_cap_mult or not math.isfinite(ppl_q):
            ppl_q = float("nan")
        delta = ppl_q - baseline_ppl if math.isfinite(ppl_q) else float("nan")
        avg_bits = sum(bit_map.values()) / len(bit_map)
        results[name] = {
            "bit_map": bit_map,
            "avg_bits": avg_bits,
            "ppl": ppl_q,
            "delta": delta,
            "extra_ppls": extra_ppls,
            "downstream": downstream,
            "downstream_summary": summarize_downstream(downstream),
        }
        print("  MP [{:<30}] avg={:.2f}bit  ppl={:.4f}  delta={:+.4f}".format(
            name, avg_bits, ppl_q if math.isfinite(ppl_q) else -1,
            delta if math.isfinite(delta) else float("nan")))
        if downstream:
            for t, m in downstream.items():
                if isinstance(m, dict) and m.get("acc") is not None:
                    print("      {:<20s} acc={:.4f}".format(t, float(m["acc"])))
    return results


def _infer_proj_group_key(comp, shape):
    if comp in ("Q", "O"):
        return "QO_square"
    if shape and len(shape) == 2:
        mn = min(shape[0], shape[1])
        mx = max(shape[0], shape[1])
        gamma = mx / mn if mn > 0 else 1.0
        if gamma > 1.5:
            return "KV_rect"
    return "KV_square"


def plot_scatter(model_key, method, baseline_ppl, layer_results, bits_list, output_dir):
    comp_colors = {"Q": "#e41a1c", "K": "#2166ac", "V": "#1a9641", "O": "#7b2d8b"}
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Outlier Density (|w|>4std)"),
        ("kurtosis", "Weight Kurtosis"),
    ]
    n_rows = len(bits_list)
    n_cols = len(x_metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for bi, bits in enumerate(bits_list):
        data_by_comp = {c: {"x": {m: [] for m, _ in x_metrics}, "y": []} for c in COMP_ORDER}
        for layer_key in sorted(layer_results.keys(), key=lambda z: int(z) if z.isdigit() else z):
            for comp, info in layer_results[layer_key].items():
                delta = info.get("ppl_delta_{}bit".format(bits), float("nan"))
                if not math.isfinite(delta):
                    continue
                if comp not in data_by_comp:
                    continue
                data_by_comp[comp]["y"].append(delta)
                for mkey, _ in x_metrics:
                    data_by_comp[comp]["x"][mkey].append(info.get(mkey, float("nan")))
        for ci, (mkey, mlabel) in enumerate(x_metrics):
            ax = axes[bi][ci]
            all_x, all_y = [], []
            for comp in COMP_ORDER:
                xs = np.array(data_by_comp[comp]["x"][mkey])
                ys = np.array(data_by_comp[comp]["y"])
                valid = np.isfinite(xs) & np.isfinite(ys)
                if valid.sum() == 0:
                    continue
                ax.scatter(xs[valid], ys[valid], c=comp_colors[comp], s=14, alpha=0.75,
                           label=comp, zorder=3)
                all_x.extend(xs[valid].tolist())
                all_y.extend(ys[valid].tolist())
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            if len(all_x) > 3:
                rho, p_val = sp_stats.spearmanr(all_x, all_y)
                r_lin, _ = sp_stats.pearsonr(all_x, all_y)
                slope, intercept, _, _, _ = sp_stats.linregress(all_x, all_y)
                x_line = np.linspace(all_x.min(), all_x.max(), 200)
                ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.2, zorder=4)
                ax.set_title(
                    "{}bit: {}\nSpearman r={:.3f} (p={:.2e}), Pearson r={:.3f}".format(
                        bits, mlabel, rho, p_val, r_lin),
                    fontsize=9,
                )
            else:
                ax.set_title("{}bit: {}".format(bits, mlabel), fontsize=9)
            ax.set_xlabel(mlabel, fontsize=8)
            ax.set_ylabel("PPL Delta (quantized - baseline)", fontsize=8)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.legend(title="Proj", fontsize=7, markerscale=1.5)
    fig.suptitle(
        "{} [{}] (baseline PPL = {:.2f}): Quantization Sensitivity vs MP Spectral Metrics".format(
            model_key, method, baseline_ppl),
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, "{}_{}_quant_scatter.png".format(model_key, method))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved scatter plot: {}".format(out_path))


def plot_heatmaps(model_key, method, layer_results, bits_list, output_dir):
    for bits in bits_list:
        layers_sorted = sorted(layer_results.keys(), key=lambda z: int(z) if z.isdigit() else z)
        present_comps = []
        for c in COMP_ORDER:
            if any(c in layer_results[l] for l in layers_sorted):
                present_comps.append(c)
        if not present_comps:
            continue
        metrics = [
            ("mp_energy_ratio", "MP Energy Ratio", "Reds"),
            ("entry_outlier_density", "Entry Outlier Density", "Reds"),
            ("kurtosis", "Kurtosis", "Purples"),
            ("ppl_delta_{}bit".format(bits), "PPL Delta {}bit".format(bits), "hot_r"),
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), max(4, len(layers_sorted) * 0.28) + 1))
        for mi, (mkey, mlabel, cmap) in enumerate(metrics):
            ax = axes[mi]
            mat = []
            for l in layers_sorted:
                row = [layer_results[l].get(c, {}).get(mkey, float("nan")) for c in present_comps]
                mat.append(row)
            mat = np.array(mat, dtype=float)
            im = ax.imshow(mat, aspect="auto", cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(mlabel, fontsize=9)
            ax.set_xticks(range(len(present_comps)))
            ax.set_xticklabels(present_comps, fontsize=8)
            step = max(1, len(layers_sorted) // 12)
            ax.set_yticks(range(0, len(layers_sorted), step))
            ax.set_yticklabels([layers_sorted[i] for i in range(0, len(layers_sorted), step)], fontsize=7)
            ax.set_ylabel("Layer" if mi == 0 else "")
        fig.suptitle("{} [{}] {}bit: Spectral Metrics vs Quantization Sensitivity".format(
            model_key, method, bits), fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(output_dir, "{}_{}_quant_heatmap_{}bit.png".format(model_key, method, bits))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved heatmap: {}".format(out_path))


def print_correlation_summary(model_key, method, layer_results, bits_list):
    x_metrics = [
        ("mp_energy_ratio", "MP Energy Ratio"),
        ("entry_outlier_density", "Entry Density"),
        ("mp_n_outliers", "N Outliers"),
        ("kurtosis", "Kurtosis"),
    ]
    group_keys = ["QO_square", "KV_square", "KV_rect"]
    group_desc = {
        "QO_square": "Q/O (square)",
        "KV_square": "K/V (square/MHA)",
        "KV_rect":   "K/V (rect/GQA)",
    }
    sep = "=" * 90
    print("\n" + sep)
    print("Correlation summary: {} [method={}]".format(model_key, method))
    print(sep)
    header = "{:<30} {:>10} {:>12} {:>10} {:>10}".format(
        "Metric (x) vs PPL Delta (y)", "Bits", "Proj Group", "Spearman r", "Pearson r")
    print(header)
    print("-" * 90)
    for bits in bits_list:
        all_x_by_group = {gk: {mk: [] for mk, _ in x_metrics} for gk in group_keys}
        all_y_by_group = {gk: [] for gk in group_keys}
        for layer_key in layer_results:
            for comp, info in layer_results[layer_key].items():
                delta = info.get("ppl_delta_{}bit".format(bits), float("nan"))
                if not math.isfinite(delta):
                    continue
                shape = info.get("shape", None)
                gkey = _infer_proj_group_key(comp, shape)
                all_y_by_group[gkey].append(delta)
                for mkey, _ in x_metrics:
                    all_x_by_group[gkey][mkey].append(info.get(mkey, float("nan")))
        for gkey in group_keys:
            y_arr = np.array(all_y_by_group[gkey])
            for mkey, mlabel in x_metrics:
                x_arr = np.array(all_x_by_group[gkey][mkey])
                valid = np.isfinite(x_arr) & np.isfinite(y_arr)
                if valid.sum() < 4:
                    continue
                rho, p_s = sp_stats.spearmanr(x_arr[valid], y_arr[valid])
                r_lin, _ = sp_stats.pearsonr(x_arr[valid], y_arr[valid])
                print("{:<30} {:>10} {:>12} {:>10.4f} {:>10.4f}".format(
                    mlabel, bits, group_desc[gkey], rho, r_lin))
    print(sep + "\n")


def _out_suffix(method, group_size):
    parts = ["_{}".format(method)]
    if group_size and group_size > 0:
        parts.append("_g{}".format(group_size))
    return "".join(parts)


def run_model(model_key, bits_list, n_tokens, block_size, ppl_cap_mult,
              output_dir, run_mp=False, mp_dominant=None, group_size=0,
              compute_hessian_sens=False, eval_datasets=None, cpu_offload=True,
              method="rtn", n_calib_tokens=2048, calib_source="c4",
              downstream_tasks=None, downstream_batch=8,
              downstream_fewshot=0, downstream_mmlu_fewshot=5,
              downstream_limit=None, downstream_baseline=True,
              save_checkpointing=True):
    if eval_datasets is None:
        eval_datasets = ["wikitext2"]
    cfg = MODELS[model_key]
    hf_name, arch = cfg["hf_name"], cfg["arch"]
    print("\n" + "#" * 80)
    print("Model: {}  ({})".format(model_key, hf_name))
    print("  method={}  group_size={}  hessian_sens={}  eval_datasets={}  cpu_offload={}".format(
        method, group_size, compute_hessian_sens, eval_datasets, cpu_offload))
    print("  n_tokens={}  n_calib_tokens={}  calib_source={}".format(
        n_tokens, n_calib_tokens, calib_source))
    if downstream_tasks:
        print("  downstream_tasks={}  limit={}  fewshot={}/{}".format(
            downstream_tasks, downstream_limit, downstream_fewshot, downstream_mmlu_fewshot))
    print("#" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    eval_tokens_map = {}
    for ds in eval_datasets:
        try:
            toks = get_eval_tokens(tokenizer, n_tokens, device, ds)
            eval_tokens_map[ds] = toks
            print("  Loaded {} tokens={}".format(ds, toks.shape[1]))
        except Exception as exc:
            print("  WARNING: failed to load {}: {}".format(ds, exc))

    primary_ds = "wikitext2" if "wikitext2" in eval_tokens_map else list(eval_tokens_map.keys())[0]
    input_ids = eval_tokens_map[primary_ds]
    print("Computing baseline perplexity on {} tokens ({})...".format(input_ids.shape[1], primary_ds))
    baseline_ppl = compute_perplexity(model, input_ids, block_size)
    print("Baseline PPL ({}) = {:.4f}".format(primary_ds, baseline_ppl))

    extra_baselines = {}
    for ds_name, ids in eval_tokens_map.items():
        if ds_name == primary_ds:
            continue
        p = compute_perplexity(model, ids, block_size)
        extra_baselines[ds_name] = p
        print("Baseline PPL ({}) = {:.4f}".format(ds_name, p))

    baseline_downstream = {}
    if downstream_tasks and downstream_baseline:
        print("Running downstream on baseline model...")
        baseline_downstream = run_downstream(
            model, tokenizer,
            tasks=downstream_tasks,
            batch_size=downstream_batch,
            num_fewshot=downstream_fewshot,
            limit=downstream_limit,
            mmlu_fewshot=downstream_mmlu_fewshot,
        )
        for t, m in baseline_downstream.items():
            if isinstance(m, dict) and m.get("acc") is not None:
                print("  baseline {:<20s} acc={:.4f}".format(t, float(m["acc"])))

    if arch == "falcon":
        proj_info = get_proj_info_falcon(model)
    elif arch == "neox":
        proj_info = get_proj_info_neox(model)
    elif arch == "opt":
        proj_info = get_proj_info_opt(model)
    else:
        proj_info = get_proj_info_llama(model)

    need_H = (method == "gptq")
    need_abs = (method == "awq")
    need_var = bool(compute_hessian_sens)
    calib_stats = None
    if need_H or need_abs or need_var:
        print("Capturing calibration stats (var={}, H={}, abs={})...".format(need_var, need_H, need_abs))
        if method in ("gptq", "awq"):
            calib_ids = get_calibration_tokens(tokenizer, n_calib_tokens, device, source=calib_source)
            print("  calib tokens={}".format(calib_ids.shape[1]))
        else:
            calib_ids = input_ids
        module_set = collect_modules_for_hooks(arch, proj_info)
        calib_stats = capture_calibration_stats(
            model, module_set, calib_ids, block_size,
            need_var=need_var, need_H=need_H, need_abs=need_abs,
        )
        print("  Captured stats for {} modules".format(len(calib_stats)))
        if method in ("gptq", "awq"):
            del calib_ids
            gc.collect()
            torch.cuda.empty_cache()

    suffix = _out_suffix(method, group_size)
    chk_path = None
    chk_payload = None
    layers_existing = None
    if save_checkpointing:
        chk_path = os.path.join(output_dir, "{}_quant_sensitivity{}_checkpoint.json".format(model_key, suffix))
        existing = load_checkpoint(chk_path)
        if existing and existing.get("method", method) == method and existing.get("model", model_key) == model_key:
            layers_existing = existing.get("layers", None)
            print("Resuming from checkpoint: {} ({} layers)".format(chk_path, len(layers_existing) if layers_existing else 0))
        chk_payload = {
            "model": model_key,
            "hf_name": hf_name,
            "arch": arch,
            "method": method,
            "bits": bits_list,
            "group_size": group_size,
            "baseline_ppl": baseline_ppl,
            "baseline_ppl_extra": extra_baselines,
            "layers": {},
        }

    if arch == "falcon" or arch == "neox":
        layer_results = run_sensitivity_falcon(
            model, proj_info, input_ids, bits_list, block_size, baseline_ppl, ppl_cap_mult,
            group_size, method, calib_stats, cpu_offload, eval_tokens_map,
            layers_existing, chk_path, chk_payload)
    else:
        layer_results = run_sensitivity_llama(
            model, proj_info, input_ids, bits_list, block_size, baseline_ppl, ppl_cap_mult,
            group_size, method, calib_stats, cpu_offload, eval_tokens_map,
            layers_existing, chk_path, chk_payload)

    mp_results = None
    dom_info = None
    if run_mp:
        ref_bits = min(bits_list)
        auto_dom, totals, counts = find_dominant_component(layer_results, bits=ref_bits)
        dom = mp_dominant if mp_dominant in COMP_ORDER else auto_dom
        ci = bootstrap_dominance_ci(layer_results, bits=ref_bits, n_boot=1000)
        dom_info = {
            "auto_dominant": auto_dom,
            "used_dominant": dom,
            "totals_{}bit".format(ref_bits): totals,
            "counts": counts,
            "bootstrap_shares_95ci": ci,
        }
        print("\n--- Mixed-Precision Experiment (dominant component: {}) ---".format(dom))
        print("  Sum(PPL delta)@{}bit per component: {}".format(ref_bits,
              {k: round(v, 4) for k, v in totals.items()}))
        for c, (mean, lo, hi) in ci.items():
            print("  Share CI {} : mean={:.3f} 95% CI=[{:.3f}, {:.3f}]".format(c, mean, lo, hi))
        downstream_cfg = None
        if downstream_tasks:
            downstream_cfg = {
                "tasks": downstream_tasks,
                "batch_size": downstream_batch,
                "num_fewshot": downstream_fewshot,
                "mmlu_fewshot": downstream_mmlu_fewshot,
                "limit": downstream_limit,
            }
        mp_results = run_mp_experiment(
            model, proj_info, arch, input_ids, block_size, baseline_ppl, dom,
            ppl_cap_mult, group_size, eval_tokens_map, method, calib_stats,
            downstream_cfg=downstream_cfg, tokenizer=tokenizer)

    result_data = {
        "model": model_key,
        "hf_name": hf_name,
        "arch": arch,
        "method": method,
        "n_tokens": int(input_ids.shape[1]),
        "n_calib_tokens": n_calib_tokens if (need_H or need_abs) else None,
        "calib_source": calib_source if (need_H or need_abs) else None,
        "bits": bits_list,
        "baseline_ppl": baseline_ppl,
        "baseline_ppl_extra": extra_baselines,
        "baseline_downstream": baseline_downstream,
        "baseline_downstream_summary": summarize_downstream(baseline_downstream),
        "group_size": group_size,
        "eval_datasets": list(eval_tokens_map.keys()),
        "primary_dataset": primary_ds,
        "layers": {k: v for k, v in layer_results.items()},
    }
    if mp_results is not None:
        result_data["mp_experiment"] = mp_results
        result_data["mp_dominant_info"] = dom_info

    out_json = os.path.join(output_dir, "{}_quant_sensitivity{}.json".format(model_key, suffix))
    with open(out_json, "w") as f:
        json.dump(result_data, f, indent=2)
    print("Saved JSON: {}".format(out_json))
    if chk_path and os.path.exists(chk_path):
        try:
            os.remove(chk_path)
        except Exception:
            pass
    print_correlation_summary(model_key, method, layer_results, bits_list)
    plot_scatter(model_key, method, baseline_ppl, layer_results, bits_list, output_dir)
    plot_heatmaps(model_key, method, layer_results, bits_list, output_dir)
    del model
    if calib_stats is not None:
        del calib_stats
    gc.collect()
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 3])
    parser.add_argument("--n_tokens", type=int, default=16384,
                        help="Eval tokens; use -1 for full dataset (full WikiText-2 test, C4 validation).")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--ppl_cap_mult", type=float, default=50.0)
    parser.add_argument("--output_dir", type=str, default="./output/quant_sensitivity")
    parser.add_argument("--mp", action="store_true")
    parser.add_argument("--mp_dominant", type=str, default=None,
                        choices=[None, "Q", "K", "V", "O"])
    parser.add_argument("--group_size", type=int, default=0)
    parser.add_argument("--compute_hessian_sens", action="store_true")
    parser.add_argument("--eval_datasets", nargs="+", default=["wikitext2"],
                        choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--method", type=str, default="rtn",
                        choices=["rtn", "gptq", "awq", "quip"])
    parser.add_argument("--n_calib_tokens", type=int, default=2048)
    parser.add_argument("--calib_source", type=str, default="c4",
                        choices=["c4", "wikitext2"])
    parser.add_argument("--downstream_tasks", nargs="+", default=None,
                        help="e.g. hellaswag arc_easy arc_challenge piqa winogrande mmlu")
    parser.add_argument("--downstream_batch", type=int, default=8)
    parser.add_argument("--downstream_fewshot", type=int, default=0)
    parser.add_argument("--downstream_mmlu_fewshot", type=int, default=5)
    parser.add_argument("--downstream_limit", type=int, default=None)
    parser.add_argument("--no_downstream_baseline", action="store_true")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for k, v in MODELS.items():
            print("{:25s}  ->  {}  [{}]".format(k, v["hf_name"], v["arch"]))
        return
    os.makedirs(args.output_dir, exist_ok=True)
    model_keys = args.models if args.models else list(MODELS.keys())
    model_keys = [m for m in model_keys if m in MODELS]
    missing = [m for m in (args.models or []) if m not in MODELS]
    if missing:
        print("WARNING: unknown model keys ignored: {}".format(missing))
    all_results = {}
    for mk in model_keys:
        try:
            result_data = run_model(
                mk, args.bits, args.n_tokens, args.block_size,
                args.ppl_cap_mult, args.output_dir,
                run_mp=args.mp, mp_dominant=args.mp_dominant,
                group_size=args.group_size,
                compute_hessian_sens=args.compute_hessian_sens,
                eval_datasets=args.eval_datasets,
                cpu_offload=(not args.no_cpu_offload),
                method=args.method,
                n_calib_tokens=args.n_calib_tokens,
                calib_source=args.calib_source,
                downstream_tasks=args.downstream_tasks,
                downstream_batch=args.downstream_batch,
                downstream_fewshot=args.downstream_fewshot,
                downstream_mmlu_fewshot=args.downstream_mmlu_fewshot,
                downstream_limit=args.downstream_limit,
                downstream_baseline=(not args.no_downstream_baseline),
                save_checkpointing=(not args.no_checkpoint),
            )
            all_results[mk] = result_data
        except Exception as exc:
            import traceback
            print("FAILED {}: {}".format(mk, exc))
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
    suffix = _out_suffix(args.method, args.group_size)
    combined_path = os.path.join(args.output_dir, "all_quant_sensitivity{}.json".format(suffix))
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("All results saved: {}".format(combined_path))


if __name__ == "__main__":
    main()