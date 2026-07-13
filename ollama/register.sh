#!/bin/bash

# Configuration
PORT=11743
DIR="/Users/venkataswaraswamy/Desktop/venkatesh_llm/ollama"
VENV_PYTHON="/Users/venkataswaraswamy/Desktop/venkatesh_llm/venv/bin/python"

echo "=== Registering Venkatesh Bommeparthi LLM ==="

# 1. Kill any existing server running on the target port
echo "Checking for existing processes on port $PORT..."
PID=$(lsof -t -i:$PORT)
if [ ! -z "$PID" ]; then
    echo "Killing existing process on port $PORT (PID: $PID)..."
    kill -9 $PID
fi

# 2. Start serve.py in the background using the project virtual environment
echo "Starting serve.py in the background..."
nohup "$VENV_PYTHON" "$DIR/serve.py" $PORT > "$DIR/serve.log" 2>&1 &

# Give it a moment to initialize
sleep 2

# Check if the server is up and listening
if lsof -i:$PORT -t >/dev/null; then
    echo "serve.py wrapper started successfully on port $PORT."
else
    echo "Error: serve.py failed to start. Log output:"
    cat "$DIR/serve.log"
    exit 1
fi

# 3. Create Ollama model entry
echo "Registering 'venkatesh-bommeparthi' model in Ollama..."
/usr/local/bin/ollama create venkatesh-bommeparthi -f "$DIR/Modelfile"

echo "=== Registration Completed Successfully ==="
echo "The model is registered in Ollama and can be listed with 'ollama list'."
echo "To run inference, send requests to the wrapper server:"
echo "curl -X POST http://127.0.0.1:$PORT/api/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Venkatesh Bommeparthi is\", \"max_tokens\": 50}'"
