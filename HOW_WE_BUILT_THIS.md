# Venkatesh Bommeparthi: How We Built a 1M-Parameter LLM

Venkatesh Bommeparthi is a custom 1,031,478-parameter decoder-only transformer Large Language Model (LLM) owned and created by Venkatesh Bommeparthi, trained from scratch on a personal iCloud text corpus, and run on a Mac laptop using Apple Silicon hardware acceleration.

---

## The Big Picture

Building a Large Language Model involves a sequence of stages to gather text, map vocabulary, define neural network layers, optimize weights, evaluate learning, and generate sequences. This project implements an end-to-end pipeline consisting of 6 core stages: Data Prep → Tokenization → Model Design → Training → Loss Visualization → Inference. An additional step generates a visual architecture diagram. Together, this pipeline demonstrates how a raw corpus is transformed into a functioning generative model.

---

## Step-by-Step Walkthrough

### Step 1: Data Preparation
* **Concept**: Gathering, cleaning, and splitting raw text data.
* **Why it's needed**: Models require structured text to learn vocabulary and character transitions.
* **Responsible File**: [data_prep.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/data_prep.py)
* **Key Details**: Scans the iCloud directory `~/Library/Mobile Documents/com~apple~CloudDocs/` for text-based extensions (`.txt`, `.md`, `.markdown`, `.json`, `.csv`, `.pdf`), skipping files >500 KB and backup folders. The text is compiled into a single [corpus.txt](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/corpus.txt) capped at 1 MB (our run scanned 90 files, yielding 1,058,152 bytes). The text is split 90/10 into a training set (948,455 characters) and a validation set (105,384 characters), with run statistics saved to [meta.json](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/meta.json).
* **Run Command**: `python3 data_prep.py`

### Step 2: Tokenization
* **Concept**: Mapping textual symbols to integers.
* **Why it's needed**: Neural networks operate on numerical tensors rather than character strings.
* **Responsible File**: [tokenizer.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/tokenizer.py)
* **Key Details**: Uses the [CharacterTokenizer](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/tokenizer.py#L4) class to index unique characters, yielding a vocabulary size of 182 tokens. Index 0 is reserved for `<unk>` (unknown characters), and the mapping is saved to [vocab.json](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/vocab.json). Character tokenization keeps the vocabulary size tiny (182 vs 32,000+ for subwords), preventing the embedding layer from bloating (e.g., 23,296 vs 4.1M parameters), which would exceed the entire budget of our 1M model.
* **Run Command**: Fitted and run inside training: `python3 train.py`

### Step 3: Model Design
* **Concept**: Defining network layers and forward-pass logic.
* **Why it's needed**: The architecture determines how inputs process through attention to predict next tokens.
* **Responsible File**: [model.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/model.py)
* **Key Details**: The [TransformerLM](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/model.py#L75) class defines a GPT-style decoder-only transformer configured in [config.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/config.py) (`N_EMBD=128`, `BLOCK_SIZE=128`, `N_HEAD=4`, `N_LAYER=5`, `DROPOUT=0.1`). Weights of the embedding (`wte`) and language model head (`lm_head`) layers are tied to conserve memory. The exact parameters total 1,031,478: Token embeddings (`wte`) contribute 23,296; Position embeddings (`wpe`) add 16,384; the 5 decoder blocks ([Block](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/model.py#L62) modules containing [CausalSelfAttention](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/model.py#L5) and feed-forward [MLP](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/model.py#L47) blocks) yield 991,360; final normalization (`ln_f`) adds 256; and the tied `lm_head` bias supplies 182.
* **Run Command**: Check parameters via: `python3 -c "import config, model; print(model.build(config).param_count)"`

### Step 4: Training
* **Concept**: Optimizing weights via backpropagation.
* **Why it's needed**: Adjusts parameters to learn patterns and structure from the corpus.
* **Responsible File**: [train.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/train.py)
* **Key Details**: Trains on Apple Silicon `mps` or `cpu` for 1,000 steps (`MAX_STEPS=1000`) using the AdamW optimizer (learning rate 1e-3). It optimizes cross-entropy loss, which measures the surprise of predicted probability distributions versus true targets. Evaluations are appended to [loss_log.jsonl](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/loss_log.jsonl) every 100 steps, and model weights, states, and metadata are checkpointed to [checkpoints/checkpoint.pt](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/checkpoints/checkpoint.pt) to support resuming.
* **Run Command**: `python3 train.py`

### Step 5: Loss Visualization
* **Concept**: Plotting validation and training curves.
* **Why it's needed**: Diagnoses learning speed, convergence, and overfitting.
* **Responsible File**: [plot_loss.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/plot_loss.py)
* **Key Details**: Parses [loss_log.jsonl](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/loss_log.jsonl) to render [loss_curve.svg](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/loss_curve.svg) and [loss_curve.html](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/loss_curve.html). Our logs demonstrate overfitting: while train loss drops from 2.5432 (step 0) to 1.4924 (step 999), validation loss reaches a minimum of 1.7543 (step 200) and then climbs back to 2.9084 (step 999) as the model memorizes the training data.
* **Run Command**: `python3 plot_loss.py`

### Step 6: Inference
* **Concept**: Generating sequences from seed prompts.
* **Why it's needed**: Assesses generation capability by decoding probabilities.
* **Responsible File**: [infer.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/infer.py)
* **Key Details**: Loads [checkpoints/checkpoint.pt](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/checkpoints/checkpoint.pt) and [vocab.json](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/vocab.json) to generate text from prompts. It supports greedy search, temperature scaling, and top-k filtering. The output is gibberish: while the model forms basic word structures, its ~1.03M parameters and small dataset are insufficient to model vocabulary syntax, sentence logic, or semantic coherency.
* **Run Command**: `python3 infer.py --prompt "Venkatesh Bommeparthi is " --max-tokens 200 --temperature 0.8`

### Step 7: Architecture Diagram
* **Concept**: Visualizing the pipeline flow.
* **Why it's needed**: Provides a blueprint of the codebase and data flow.
* **Responsible File**: [architecture_diagram.py](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/architecture_diagram.py)
* **Key Details**: Generates [architecture_diagram.svg](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/architecture_diagram.svg) and [architecture_diagram.html](file:///Users/venkataswaraswamy/Desktop/venkatesh_llm/architecture_diagram.html) to map scripts, artifacts, training checkpoints, and the model's internal tensor flow.
* **Run Command**: `python3 architecture_diagram.py`

---

## What We Learned & Limitations

Building the Venkatesh Bommeparthi model demonstrates the operational pipeline of modern generative AI, but highlights severe constraints compared to commercial models:
1. **Scale Requirements**: A 1.03M-parameter model trained on ~1 MB of text has extremely low capacity. It learns spelling patterns and word clusters but cannot form logical sentences. Commercial LLMs require billions of parameters and trillions of tokens to achieve semantic understanding.
2. **Overfitting Spikes**: Due to the tiny corpus, validation loss begins rising after step 200 while training loss drops. Real-world models mitigate memorization by training on diverse, massive datasets.
3. **Tokenization Limitations**: Character-level tokenizers are simple and parameter-efficient for tiny architectures but yield long token sequence lengths. Commercial systems utilize subword BPE tokenization to process sequences more densely.

---

## Full Run Commands

Follow this sequence to run the entire pipeline from scratch:

```bash
# 1. Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (PyTorch)
pip install -r requirements.txt

# 3. Gather and split text assets from iCloud
python3 data_prep.py

# 4. Train the model for 1000 steps
python3 train.py

# 5. Render the loss evaluation plots
python3 plot_loss.py

# 6. Generate text starting with a prompt
python3 infer.py --prompt "Venkatesh Bommeparthi is " --max-tokens 200 --temperature 0.8

# 7. Generate the interactive architecture diagram files
python3 architecture_diagram.py
```
