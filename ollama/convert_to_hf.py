#!/usr/bin/env python3
"""Convert the Venkatesh Bommeparthi custom GPT-2-style model -> HuggingFace format
for llama.cpp GGUF conversion. Writes to ollama/hf/ :
  - config.json (gpt2 arch, our params)
  - pytorch_model.bin (state_dict with HF GPT-2 tensor names, already matching)
  - tokenizer.json (char vocab, HF format so llama.cpp embeds it)
  - special_tokens_map.json, tokenizer_config.json
Run AFTER this: see instructions printed at end.
"""
import json, os, sys
import torch

ROOT = "/Users/venkataswaraswamy/Desktop/venkatesh_llm"
HF = os.path.join(ROOT, "ollama", "hf")
os.makedirs(HF, exist_ok=True)

# 1. Load our checkpoint
ckpt = torch.load(os.path.join(ROOT, "checkpoints", "checkpoint.pt"), map_location="cpu")
sd = ckpt["model_state_dict"]
vocab_size = ckpt.get("vocab_size", 182)
n_embd, n_layer, n_head, block_size = 128, 5, 4, 128

# 2. config.json (GPT-2 arch so convert_hf_to_gguf recognises it)
cfg = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "n_embd": n_embd,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_positions": block_size,
    "n_ctx": block_size,
    "vocab_size": vocab_size,
    "block_size": block_size,
    "bos_token_id": 0,
    "eos_token_id": 0,
    "tie_word_embeddings": True,
}
with open(os.path.join(HF, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# 3. pytorch_model.bin — names already match HF GPT-2 (transformer.* and lm_head)
#    Our lm_head.weight is tied to wte, so add it explicitly for HF.
hf_sd = dict(sd)
if "lm_head.weight" not in hf_sd:
    hf_sd["lm_head.weight"] = sd["transformer.wte.weight"]
torch.save(hf_sd, os.path.join(HF, "pytorch_model.bin"))

# 4. Char tokenizer -> HF tokenizer.json
with open(os.path.join(ROOT, "vocab.json")) as f:
    raw = json.load(f)
vocab = raw.get("char2idx") or raw.get("vocab")  # {char: idx}
chars = sorted(vocab, key=lambda c: vocab[c])
assert len(chars) == vocab_size, f"vocab mismatch {len(chars)} != {vocab_size}"

tokens = []
scores = []
for i, ch in enumerate(chars):
    tokens.append(ch)
    scores.append(-i)  # arbitrary descending scores

tokenizer_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False,
         "special": True}
    ],
    "normalizer": None,
    "pre_tokenizer": {
        "type": "PreTokenizer",
        "pre_tokenizers": [{"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False}]
    },
    "post_processor": {
        "type": "TemplateProcessing",
        "special_tokens": [["<unk>", 0]],
        "template": "[0]",
    },
    "decoder": {"type": "ByteLevel"},
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": {t: i for i, t in enumerate(tokens)},
        "merges": [],
        "ignore_merges": True,
    },
}
with open(os.path.join(HF, "tokenizer.json"), "w") as f:
    json.dump(tokenizer_json, f, indent=2)

with open(os.path.join(HF, "special_tokens_map.json"), "w") as f:
    json.dump({"unk_token": "<unk>"}, f)
with open(os.path.join(HF, "tokenizer_config.json"), "w") as f:
    json.dump({"unk_token": "<unk>", "model_max_length": block_size,
               "tokenizer_class": "GPT2Tokenizer"}, f, indent=2)

print("WROTE HF-format model to:", HF)
print("  config.json  (gpt2, n_layer=%d n_embd=%d n_head=%d vocab=%d)" % (n_layer, n_embd, n_head, vocab_size))
print("  pytorch_model.bin  (%d tensors)" % len(hf_sd))
print("  tokenizer.json  (%d char tokens)" % len(tokens))
print("\nNEXT STEP (run from llama.cpp src dir):")
print('  python3 /Users/venkataswaraswamy/Desktop/srinivasa_llm/llama_cpp_src/convert_hf_to_gguf.py \\')
print('    --outtype f16 --outfile %s/model.gguf %s' % (HF, HF))
print("THEN:")
print("  cd %s && ollama create venkatesh-bommeparthi -f Modelfile_gguf" % os.path.join(ROOT, "ollama"))
