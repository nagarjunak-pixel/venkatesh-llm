#!/usr/bin/env python3
"""Write a GGUF file for the Venkatesh Bommeparthi GPT-2-style model directly,
using the `gguf` PyPI package (GgufWriter). Avoids the broken llama.cpp
convert_hf_to_gguf.py (its gguf-py/conversion.py submodule is missing here).

Produces ollama/hf/model.gguf that Ollama can load as arch 'gpt2'.
"""
import json, os
import torch
import numpy as np
from gguf import GGUFWriter, GGUFEndian

ROOT = "/Users/venkataswaraswamy/Desktop/venkatesh_llm"
HF = os.path.join(ROOT, "ollama", "hf")
OUT = os.path.join(HF, "model.gguf")

ckpt = torch.load(os.path.join(ROOT, "checkpoints", "checkpoint.pt"), map_location="cpu")
sd = ckpt["model_state_dict"]
vocab_size = int(ckpt.get("vocab_size", 182))
n_embd, n_layer, n_head, block_size = 128, 5, 4, 128

writer = GGUFWriter(OUT, "gpt2", endianness=GGUFEndian.LITTLE)

# --- general metadata ---
writer.add_architecture()
writer.add_name("Venkatesh Bommeparthi")
writer.add_description("Custom 1,031,478-param char-level GPT-2-style LLM, trained from scratch on the owner's iCloud text. Educational demo.")
writer.add_author("Venkatesh Bommeparthi")
writer.add_license("MIT")

# --- gpt2 hyperparams ---
writer.add_context_length(block_size)
writer.add_embedding_length(n_embd)
writer.add_block_count(n_layer)
writer.add_feed_forward_length(4 * n_embd)
writer.add_head_count(n_head)
writer.add_head_count_kv(n_head)
writer.add_vocab_size(vocab_size)
writer.add_rope_dimension_count(0)
writer.add_key_length(n_embd // n_head)
writer.add_value_length(n_embd // n_head)
writer.add_layer_norm_eps(1e-5)
writer.add_file_type(1)  # f16

# --- tensors (f16) ---
def add_t(name, t):
    arr = t.detach().cpu().to(torch.float16).numpy()
    writer.add_tensor(name, arr)

add_t("token_embd.weight", sd["transformer.wte.weight"])
add_t("position_embd.weight", sd["transformer.wpe.weight"])

for i in range(n_layer):
    p = f"transformer.h.{i}."
    add_t(f"blk.{i}.ln1.weight", sd[p + "ln_1.weight"])
    add_t(f"blk.{i}.ln1.bias", sd[p + "ln_1.bias"])
    add_t(f"blk.{i}.attn_qkv.weight", sd[p + "attn.c_attn.weight"])
    add_t(f"blk.{i}.attn_qkv.bias", sd[p + "attn.c_attn.bias"])
    add_t(f"blk.{i}.attn_output.weight", sd[p + "attn.c_proj.weight"])
    add_t(f"blk.{i}.attn_output.bias", sd[p + "attn.c_proj.bias"])
    add_t(f"blk.{i}.ln2.weight", sd[p + "ln_2.weight"])
    add_t(f"blk.{i}.ln2.bias", sd[p + "ln_2.bias"])
    add_t(f"blk.{i}.ffn_up.weight", sd[p + "mlp.c_fc.weight"])
    add_t(f"blk.{i}.ffn_up.bias", sd[p + "mlp.c_fc.bias"])
    add_t(f"blk.{i}.ffn_down.weight", sd[p + "mlp.c_proj.weight"])
    add_t(f"blk.{i}.ffn_down.bias", sd[p + "mlp.c_proj.bias"])

add_t("output_norm.weight", sd["transformer.ln_f.weight"])
add_t("output_norm.bias", sd["transformer.ln_f.bias"])
add_t("output.weight", sd["transformer.wte.weight"])  # tied lm_head

# --- tokenizer: char vocab (182 tokens) ---
with open(os.path.join(ROOT, "vocab.json")) as f:
    raw = json.load(f)
char2idx = raw.get("char2idx") or raw.get("vocab")
chars = sorted(char2idx, key=lambda c: char2idx[c])
assert len(chars) == vocab_size

writer.add_tokenizer_model("gpt2")
writer.add_token_list(chars)
writer.add_token_scores([-float(i) for i in range(len(chars))])
writer.add_bos_token_id(0)
writer.add_eos_token_id(0)

writer.write_header_to_file()
writer.close()

print("WROTE", OUT)
print("size bytes:", os.path.getsize(OUT))
