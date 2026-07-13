import sys
import os
import json
import torch
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add parent directory to sys.path to allow importing from model.py and config.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from model import TransformerLM
from tokenizer import CharacterTokenizer
from config import DEVICE

# Paths
dir_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(dir_path, "config.json")
tokenizer_path = os.path.join(dir_path, "tokenizer.json")
safetensors_path = os.path.join(dir_path, "model.safetensors")
pt_path = os.path.join(dir_path, "model.pt")

# Global instances loaded at startup
tokenizer = None
model = None

def init_server():
    global tokenizer, model
    
    # Load tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        
    # Build model (dropout=0.0 during inference)
    model = TransformerLM(
        vocab_size=config_data["vocab_size"],
        n_embd=config_data["n_embd"],
        block_size=config_data["block_size"],
        n_layer=config_data["n_layer"],
        n_head=config_data["n_head"],
        dropout=0.0
    )
    
    # Load state dict
    if os.path.exists(safetensors_path):
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(safetensors_path)
            print("Loaded model from safetensors.")
        except ImportError:
            print("Warning: model.safetensors exists but safetensors library is not installed. Falling back to .pt")
            state_dict = torch.load(pt_path, map_location=DEVICE)
    else:
        state_dict = torch.load(pt_path, map_location=DEVICE)
        print("Loaded model from PyTorch checkpoint (.pt).")
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

class ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
            except Exception as e:
                self.send_error(400, f"Invalid JSON: {str(e)}")
                return
            
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 50)
            temperature = data.get("temperature", 0.8)
            
            try:
                # Generate
                prompt_ids = tokenizer.encode(prompt)
                x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                
                # Check sequence length constraint
                if x.size(1) > model.block_size:
                    x = x[:, -model.block_size:]
                
                y = model.generate(
                    idx=x,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0.0 else False
                )
                generated_text = tokenizer.decode(y[0].tolist())
                
                response_data = {"text": generated_text}
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Error generating text: {str(e)}")
        else:
            self.send_error(404, "Not Found")

def run(port=11743):
    init_server()
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, ModelHandler)
    print("READY", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 11743
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port=port)
