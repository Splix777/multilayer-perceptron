#!/bin/sh
# Start the Ollama server
ollama serve &

# Wait for the server to initialize
sleep 5

# Pull the required model
ollama pull llama3
ollama pull mxbai-embed-large

# Keep the container running
wait
