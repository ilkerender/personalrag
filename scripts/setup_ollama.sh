#!/bin/bash


set -e

if ! command -v ollama &> /dev/null
then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "Ollama is already installed."
fi

MODEL="deepseek-r1:1.5b"
echo "Pulling model: $MODEL..."
ollama pull $MODEL

echo "Starting Ollama server..."
ollama serve &

echo "Ollama setup complete. Running at http://localhost:11434"
