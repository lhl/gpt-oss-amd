#!/usr/bin/env python3
"""
Export a Hugging Face GPT-OSS checkpoint (safetensors) to gpt-oss-amd .bin format.

See WORKING.md for details and usage examples.
"""
from __future__ import annotations
import argparse, json, struct, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

I32_LE = "<i"
F32_LE = "<f"

@dataclass
class ExportConfig:
    vocab_size: int
    hidden_dim: int
    n_experts: int
    experts_per_token: int
    intermediate_dim: int
    n_layers: int
    head_dim: int
    n_attn_heads: int
    n_kv_heads: int
    seq_len: int
    initial_context_length: int
    rope_theta: float
    rope_scaling_factor: float
    sliding_window: int
    swiglu_limit: float
    def pack_le(self) -> bytes:
        # Must match src/run.cpp:Config field order and types exactly
        buf = b""
        buf += struct.pack(I32_LE, int(self.vocab_size))
        buf += struct.pack(I32_LE, int(self.hidden_dim))
        buf += struct.pack(I32_LE, int(self.n_experts))
        buf += struct.pack(I32_LE, int(self.experts_per_token))
        buf += struct.pack(I32_LE, int(self.intermediate_dim))
        buf += struct.pack(I32_LE, int(self.n_layers))
        buf += struct.pack(I32_LE, int(self.head_dim))
        buf += struct.pack(I32_LE, int(self.n_attn_heads))
        buf += struct.pack(I32_LE, int(self.n_kv_heads))
        buf += struct.pack(I32_LE, int(self.seq_len))
        buf += struct.pack(I32_LE, int(self.initial_context_length))
        buf += struct.pack(F32_LE, float(self.rope_theta))
        buf += struct.pack(F32_LE, float(self.rope_scaling_factor))
        buf += struct.pack(I32_LE, int(self.sliding_window))
        buf += struct.pack(F32_LE, float(self.swiglu_limit))
        return buf

def open_all_safetensors(dir_path: Path) -> Dict[str, Tuple[Path, str]]:
    m: Dict[str, Tuple[Path, str]] = {}
    for fp in dir_path.glob("**/*.safetensors"):
        try:
            with safe_open(fp, framework='pt') as f:
                for k in f.keys():
                    arr = f.get_tensor(k)
                    m[k] = (fp, str(arr.dtype))
        except Exception:
            continue
    return m

def load_tensor(fp: Path, key: str) -> np.ndarray:
    with safe_open(fp, framework='pt') as f:
        import torch
        t=f.get_tensor(key)
    if hasattr(t,'to'):
        return t.detach().cpu().to(torch.float32).numpy()
    return np.array(t, dtype=np.float32)

def guess_mapper(keys: Iterable[str]) -> str:
    """Best-effort mapper detection.

    Prefer the HF layout if both patterns appear (some repos ship both fused
    and HF-style weights). Fall back to fused GPT-OSS layout otherwise.
    """
    ks = set(keys)
    # Prefer HF-style if present
    if any("model.layers.0.self_attn.q_proj.weight" in k for k in ks):
        return "hf_gpt_oss"
    # Original fused GPT-OSS layout
    if any("attn.qkv.weight" in k for k in ks):
        return "gpt_oss"
    return "unknown"

def export_gpt_oss(model_dir: Path, out_path: Path) -> None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        print(f"[ERROR] Missing {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg_json = json.loads(cfg_path.read_text())
    def getv(d: dict, *names, default=None, required=False):
        for n in names:
            if n in d:
                return d[n]
        if required:
            raise KeyError(f"Missing required config field among {names}")
        return default
    vocab_size = int(getv(cfg_json, "vocab_size", required=True))
    hidden_dim = int(getv(cfg_json, "hidden_dim", "d_model", required=True))
    n_layers = int(getv(cfg_json, "n_layers", "num_hidden_layers", required=True))
    n_attn_heads = int(getv(cfg_json, "n_attn_heads", "num_attention_heads", required=True))
    n_kv_heads = int(getv(cfg_json, "n_kv_heads", "num_key_value_heads", default=n_attn_heads))
    head_dim = int(getv(cfg_json, "head_dim", default=hidden_dim // n_attn_heads))
    seq_len = int(getv(cfg_json, "seq_len", "max_position_embeddings", default=2048))
    initial_context_length = int(getv(cfg_json, "initial_context_length", default=seq_len))
    rope_theta = float(getv(cfg_json, "rope_theta", default=150000.0))
    rope_scaling_factor = float(getv(cfg_json, "rope_scaling_factor", default=32.0))
    sliding_window = int(getv(cfg_json, "sliding_window", default=0))
    swiglu_limit = float(getv(cfg_json, "swiglu_limit", default=7.0))
    n_experts = int(getv(cfg_json, "n_experts", default=0))
    experts_per_token = int(getv(cfg_json, "experts_per_token", default=0))
    intermediate_dim = int(getv(cfg_json, "intermediate_dim", "ffn_hidden_size", default=4 * hidden_dim))
    expcfg = ExportConfig(
        vocab_size=vocab_size, hidden_dim=hidden_dim, n_experts=n_experts,
        experts_per_token=experts_per_token, intermediate_dim=intermediate_dim,
        n_layers=n_layers, head_dim=head_dim, n_attn_heads=n_attn_heads,
        n_kv_heads=n_kv_heads, seq_len=seq_len, initial_context_length=initial_context_length,
        rope_theta=rope_theta, rope_scaling_factor=rope_scaling_factor,
        sliding_window=sliding_window, swiglu_limit=swiglu_limit,
    )
    key_to_file = open_all_safetensors(model_dir)
    if not key_to_file:
        print(f"[ERROR] No .safetensors found under {model_dir}", file=sys.stderr)
        sys.exit(3)
    def find_key(suffix: str) -> Optional[Tuple[str, Path]]:
        for k, (fp, _) in key_to_file.items():
            if k.endswith(suffix):
                return k, fp
        return None
    tok_key = find_key("token_embedding.weight") or find_key("tok_embeddings.weight") or find_key("embed_tokens.weight")
    out_key = find_key("unembedding.weight") or find_key("lm_head.weight") or find_key("output.weight")
    if not tok_key or not out_key:
        print("[ERROR] Could not locate token embedding or unembedding weights.")
        print("Known keys (last 50):")
        for k in list(key_to_file.keys())[-50:]:
            print("  ", k)
        sys.exit(4)
    token_embedding = load_tensor(tok_key[1], tok_key[0])
    out_weight = load_tensor(out_key[1], out_key[0])
    def layer_key(i: int, suffix: str) -> Optional[Tuple[str, Path]]:
        for k, (fp, _) in key_to_file.items():
            if f"layers.{i}." in k and k.endswith(suffix):
                return k, fp
        return None
    rms_attn_list: List[np.ndarray] = []
    rms_ffn_list: List[np.ndarray] = []
    w_qkv_list: List[np.ndarray] = []
    b_qkv_list: List[np.ndarray] = []
    w_o_list: List[np.ndarray] = []
    b_o_list: List[np.ndarray] = []
    attn_sinks_list: List[np.ndarray] = []
    w_router_list: List[np.ndarray] = []
    b_router_list: List[np.ndarray] = []
    w_mlp1_list: List[np.ndarray] = []
    b_mlp1_list: List[np.ndarray] = []
    w_mlp2_list: List[np.ndarray] = []
    b_mlp2_list: List[np.ndarray] = []
    for i in range(n_layers):
        rms_attn = layer_key(i, "attn.norm.scale") or layer_key(i, "input_layernorm.weight")
        rms_ffn = layer_key(i, "mlp.norm.scale") or layer_key(i, "post_attention_layernorm.weight")
        if not rms_attn or not rms_ffn:
            print(f"[ERROR] Missing RMSNorm scales for layer {i}")
            sys.exit(5)
        rms_attn_list.append(load_tensor(rms_attn[1], rms_attn[0]).reshape(-1))
        rms_ffn_list.append(load_tensor(rms_ffn[1], rms_ffn[0]).reshape(-1))
        qkv = layer_key(i, "attn.qkv.weight")
        qkv_b = layer_key(i, "attn.qkv.bias")
        o_w = layer_key(i, "attn.o.weight")
        o_b = layer_key(i, "attn.o.bias")
        if not (qkv and qkv_b and o_w and o_b):
            print(f"[ERROR] Missing QKV/O weights for layer {i}. Unsupported mapping.")
            sys.exit(6)
        w_qkv_list.append(load_tensor(qkv[1], qkv[0]))
        b_qkv_list.append(load_tensor(qkv_b[1], qkv_b[0]))
        w_o_list.append(load_tensor(o_w[1], o_w[0]))
        b_o_list.append(load_tensor(o_b[1], o_b[0]))
        sinks = layer_key(i, "attn.sinks")
        if sinks:
            attn_sinks_list.append(load_tensor(sinks[1], sinks[0]).reshape(-1))
        else:
            attn_sinks_list.append(np.zeros((n_attn_heads,), dtype=np.float32))
        w_router = layer_key(i, "mlp.gate.weight")
        b_router = layer_key(i, "mlp.gate.bias")
        if w_router and b_router:
            w_router_list.append(load_tensor(w_router[1], w_router[0]))
            b_router_list.append(load_tensor(b_router[1], b_router[0]).reshape(-1))
        else:
            if expcfg.n_experts > 0:
                print(f"[ERROR] Expected MoE router for layer {i} but not found.")
                sys.exit(7)
        w_mlp1 = layer_key(i, "mlp.mlp1_weight")
        b_mlp1 = layer_key(i, "mlp.mlp1_bias")
        w_mlp2 = layer_key(i, "mlp.mlp2_weight")
        b_mlp2 = layer_key(i, "mlp.mlp2_bias")
        if all([w_mlp1, b_mlp1, w_mlp2, b_mlp2]):
            w_mlp1_list.append(load_tensor(w_mlp1[1], w_mlp1[0]))
            b_mlp1_list.append(load_tensor(b_mlp1[1], b_mlp1[0]))
            w_mlp2_list.append(load_tensor(w_mlp2[1], w_mlp2[0]))
            b_mlp2_list.append(load_tensor(b_mlp2[1], b_mlp2[0]))
        else:
            if expcfg.n_experts > 0:
                print(f"[ERROR] Expected MoE expert weights for layer {i} but not found.")
                sys.exit(8)
    rms_out_key = find_key("norm.scale") or find_key("final_layernorm.weight")
    if not rms_out_key:
        print("[ERROR] Missing final norm scale (rms_out_w)")
        sys.exit(9)
    rms_out_w = load_tensor(rms_out_key[1], rms_out_key[0]).reshape(-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(expcfg.pack_le())
        def write_arr(arr: np.ndarray):
            arr = np.asarray(arr, dtype=np.float32, order="C")
            f.write(arr.tobytes(order="C"))
        write_arr(token_embedding)
        write_arr(out_weight)
        write_arr(np.stack(rms_attn_list, axis=0))
        write_arr(np.stack(rms_ffn_list, axis=0))
        write_arr(rms_out_w)
        write_arr(np.stack(w_qkv_list, axis=0))
        write_arr(np.stack(b_qkv_list, axis=0))
        write_arr(np.stack(w_o_list, axis=0))
        write_arr(np.stack(b_o_list, axis=0))
        write_arr(np.stack(attn_sinks_list, axis=0))
        if w_router_list and w_mlp1_list and w_mlp2_list:
            write_arr(np.stack(w_router_list, axis=0))
            write_arr(np.stack(b_router_list, axis=0))
            write_arr(np.stack(w_mlp1_list, axis=0))
            write_arr(np.stack(b_mlp1_list, axis=0))
            write_arr(np.stack(w_mlp2_list, axis=0))
            write_arr(np.stack(b_mlp2_list, axis=0))
    print(f"[OK] Exported model to {out_path}")

def resolve_model(model_id: str, revision: Optional[str], local_dir: Optional[str]) -> Path:
    if local_dir:
        p = Path(local_dir)
        if not p.exists():
            print(f"[ERROR] local directory not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p
    cache_dir = snapshot_download(repo_id=model_id, revision=revision, allow_patterns=["*.json", "*.safetensors"], local_files_only=False)
    return Path(cache_dir)



def export_hf_gpt_oss(model_dir: Path, out_path: Path) -> None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        print(f"[ERROR] Missing {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(cfg_path.read_text())
    vocab_size = int(cfg.get("vocab_size"))
    hidden_dim = int(cfg.get("hidden_size", cfg.get("hidden_dim")))
    n_layers = int(cfg.get("num_hidden_layers"))
    n_attn_heads = int(cfg.get("num_attention_heads"))
    n_kv_heads = int(cfg.get("num_key_value_heads", n_attn_heads))
    head_dim = int(cfg.get("head_dim", hidden_dim // max(1,n_attn_heads)))
    seq_len = int(cfg.get("max_position_embeddings", 2048))
    initial_context_length = int(cfg.get("initial_context_length", seq_len))
    rope_theta = float(cfg.get("rope_theta", 150000.0))
    rope_scaling_factor = float((cfg.get("rope_scaling") or {}).get("factor", 32.0))
    sliding_window = int(cfg.get("sliding_window", 0))
    swiglu_limit = float(cfg.get("swiglu_limit", 7.0))
    n_experts = int(cfg.get("num_local_experts", 0))
    experts_per_token = int(cfg.get("experts_per_token", cfg.get("num_experts_per_tok", 0)))

    expcfg = ExportConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_experts=n_experts,
        experts_per_token=experts_per_token,
        intermediate_dim=int(cfg.get("intermediate_size", hidden_dim)),
        n_layers=n_layers,
        head_dim=head_dim,
        n_attn_heads=n_attn_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        initial_context_length=initial_context_length,
        rope_theta=rope_theta,
        rope_scaling_factor=rope_scaling_factor,
        sliding_window=sliding_window,
        swiglu_limit=swiglu_limit,
    )

    ktf = open_all_safetensors(model_dir)
    def K(*parts):
        return ".".join(parts)
    def get(k):
        if k not in ktf:
            return None
        fp,_=ktf[k]
        return load_tensor(fp,k)
    # Embeddings
    token_embedding = get(K("model","embed_tokens","weight"))
    out_weight = get("lm_head.weight")
    if token_embedding is None or out_weight is None:
        print("[ERROR] Missing embeddings in HF snapshot", file=sys.stderr)
        sys.exit(11)

    # Per-layer
    rms_attn_list=[]; rms_ffn_list=[]; w_qkv_list=[]; b_qkv_list=[]; w_o_list=[]; b_o_list=[]; sinks_list=[]
    w_router_list=[]; b_router_list=[]; w_mlp1_list=[]; b_mlp1_list=[]; w_mlp2_list=[]; b_mlp2_list=[]

    for i in range(n_layers):
        rms_attn = get(K("model","layers",str(i),"input_layernorm","weight"))
        rms_ffn = get(K("model","layers",str(i),"post_attention_layernorm","weight"))
        if rms_attn is None or rms_ffn is None:
            print(f"[ERROR] Missing norms for layer {i}", file=sys.stderr); sys.exit(12)
        rms_attn_list.append(rms_attn.reshape(-1))
        rms_ffn_list.append(rms_ffn.reshape(-1))
        # QKV fuse
        q = get(K("model","layers",str(i),"self_attn","q_proj","weight"))
        k = get(K("model","layers",str(i),"self_attn","k_proj","weight"))
        v = get(K("model","layers",str(i),"self_attn","v_proj","weight"))
        qb = get(K("model","layers",str(i),"self_attn","q_proj","bias"))
        kb = get(K("model","layers",str(i),"self_attn","k_proj","bias"))
        vb = get(K("model","layers",str(i),"self_attn","v_proj","bias"))
        if any(x is None for x in (q,k,v,qb,kb,vb)):
            print(f"[ERROR] Missing QKV for layer {i}", file=sys.stderr); sys.exit(13)
        w_qkv_list.append(np.concatenate([q,k,v], axis=0))
        b_qkv_list.append(np.concatenate([qb.reshape(-1),kb.reshape(-1),vb.reshape(-1)], axis=0))
        o = get(K("model","layers",str(i),"self_attn","o_proj","weight"))
        ob = get(K("model","layers",str(i),"self_attn","o_proj","bias"))
        if o is None or ob is None:
            print(f"[ERROR] Missing O for layer {i}", file=sys.stderr); sys.exit(14)
        w_o_list.append(o)
        b_o_list.append(ob.reshape(-1))
        sinks = get(K("model","layers",str(i),"self_attn","sinks"))
        sinks_list.append(sinks.reshape(-1) if sinks is not None else np.zeros((n_attn_heads,),dtype=np.float32))
        # Router
        wr = get(K("model","layers",str(i),"mlp","router","weight"))
        br = get(K("model","layers",str(i),"mlp","router","bias"))
        if wr is not None and br is not None and n_experts>0:
            w_router_list.append(wr.T)  # to (H, E)
            b_router_list.append(br.reshape(-1))
        # Experts
        gu = get(K("model","layers",str(i),"mlp","experts","gate_up_proj"))
        gub = get(K("model","layers",str(i),"mlp","experts","gate_up_proj_bias"))
        dn = get(K("model","layers",str(i),"mlp","experts","down_proj"))
        dnb = get(K("model","layers",str(i),"mlp","experts","down_proj_bias"))
        if n_experts>0 and all(x is not None for x in (gu,gub,dn,dnb)):
            # gu: (E, H, 2I) -> (E, 2I, H)
            w_mlp1_list.append(np.transpose(gu,(0,2,1)))
            b_mlp1_list.append(gub)
            # dn: (E, H, I) already matches (E, H, I)
            w_mlp2_list.append(dn)
            b_mlp2_list.append(dnb)

    # Final norm
    rms_out = get("model.norm.weight")
    if rms_out is None:
        rms_out = get("final_layernorm.weight")
    if rms_out is None:
        print("[ERROR] Missing final norm", file=sys.stderr); sys.exit(15)

    # Sanity for MoE: if experts are expected but expert weights are missing (likely quantized
    # snapshot with blocks/scales), instruct user to dequantize first.
    if n_experts>0:
        have_router = len(w_router_list)>0
        have_experts = len(w_mlp1_list)>0 and len(w_mlp2_list)>0
        if have_router and not have_experts:
            import sys
            msg = (
                "[ERROR] MoE router found but expert weights missing. "
                "If your snapshot is quantized (e.g., mxfp4), dequantize to BF16 first:\
"
                "  python3 tools/dequantize_to_bf16.py 20b --src <HF-snapshot> --dst <bf16-dir>\\n"
                "Then export with --snapshot <bf16-dir>"
            )
            print("".join(msg), file=sys.stderr)
            sys.exit(16)

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(expcfg.pack_le())
        def W(a):
            a=np.asarray(a, dtype=np.float32, order='C'); f.write(a.tobytes(order='C'))
        W(token_embedding); W(out_weight)
        W(np.stack(rms_attn_list,0)); W(np.stack(rms_ffn_list,0)); W(rms_out.reshape(-1))
        W(np.stack(w_qkv_list,0)); W(np.stack(b_qkv_list,0))
        W(np.stack(w_o_list,0)); W(np.stack(b_o_list,0))
        W(np.stack(sinks_list,0))
        if w_router_list and w_mlp1_list and w_mlp2_list:
            W(np.stack(w_router_list,0)); W(np.stack(b_router_list,0))
            W(np.stack(w_mlp1_list,0)); W(np.stack(b_mlp1_list,0))
            W(np.stack(w_mlp2_list,0)); W(np.stack(b_mlp2_list,0))
    print(f"[OK] Exported model to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export HF GPT-OSS model to gpt-oss-amd .bin")
    ap.add_argument("--model-id", default=None, help="Hugging Face model id, e.g. openai/gpt-oss-120b")
    ap.add_argument("--revision", default=None, help="Optional HF revision")
    ap.add_argument("--snapshot", default=None, help="Local snapshot directory to use instead of model id")
    ap.add_argument("-o", "--out", required=True, help="Output .bin path")
    ap.add_argument("--print-keys", action="store_true", help="Print safetensors keys and exit")
    args = ap.parse_args()
    model_dir = resolve_model(args.model_id or "", args.revision, args.snapshot)
    if args.print_keys:
        ktf = open_all_safetensors(model_dir)
        print("Found keys (sample):")
        for k in list(ktf.keys())[:200]:
            print("  ", k)
        print(f"Total keys: {len(ktf)}")
        return
    ktf = open_all_safetensors(model_dir)
    mapper = guess_mapper(ktf.keys())
    if mapper == "gpt_oss":
        export_gpt_oss(model_dir, Path(args.out))
    elif mapper == "hf_gpt_oss":
        export_hf_gpt_oss(model_dir, Path(args.out))
    else:
        print("[ERROR] Unsupported or unknown model mapping. Use --print-keys to inspect keys.")
        sys.exit(10)

if __name__ == "__main__":
    main()