import os
import json
import torch

def main():
    checkpoint_path = "/Users/venkataswaraswamy/Desktop/venkatesh_llm/checkpoints/checkpoint.pt"
    output_dir = "/Users/venkataswaraswamy/Desktop/venkatesh_llm/ollama"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]

    # Try to save as safetensors if library is available, else fall back to torch.save
    try:
        import safetensors.torch
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        safetensors.torch.save_file(model_state_dict, safetensors_path)
        print(f"Saved weights to safetensors: {safetensors_path}")
    except ImportError:
        pt_path = os.path.join(output_dir, "model.pt")
        torch.save(model_state_dict, pt_path)
        print(f"Safetensors library not found. Saved weights to torch checkpoint: {pt_path}")

    # Save config.json
    config_data = {
        "n_embd": 128,
        "n_layer": 5,
        "n_head": 4,
        "block_size": 128,
        "vocab_size": 182,
        "model_type": "gpt2"
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to {config_path}")

    # Save tokenizer.json (the 182-char vocab from vocab.json)
    vocab_src_path = "/Users/venkataswaraswamy/Desktop/venkatesh_llm/vocab.json"
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(vocab_src_path):
        with open(vocab_src_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        with open(tokenizer_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Saved tokenizer to {tokenizer_path}")
    else:
        print(f"Warning: vocab.json not found at {vocab_src_path}")

    # Save metadata.json
    metadata_data = {
        "model_name": "Venkatesh Bommeparthi",
        "owner": "Venkatesh Bommeparthi"
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_data, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()
