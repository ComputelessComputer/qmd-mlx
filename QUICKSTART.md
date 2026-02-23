# QMD-MLX Quick Start Guide

## Setup Complete ✓

Your QMD-MLX installation is now working with the MLX backend for Apple Silicon!

## What Was Fixed

1. **TypeScript Compilation**: Fixed LLM interface and session management
2. **MLX Backend**: Fixed Python tokenizer and model API compatibility
3. **Path Resolution**: Fixed script path detection for built distribution

## Quick Usage

### 1. Index Your Documents

```bash
# Add a collection of markdown files
qmd collection add ~/Documents/notes --name notes

# Index the files
qmd update
```

### 2. Generate Embeddings

```bash
# Generate vector embeddings (uses MLX backend automatically)
qmd embed
```

### 3. Search

```bash
# Fast keyword search (BM25)
qmd search "your search query"

# Vector semantic search
qmd vsearch "your search query"

# Hybrid search with LLM reranking (best results)
qmd query "your search query"
```

### 4. Add Context

```bash
# Add human-readable context to help with search
qmd context add qmd://notes "Personal notes and ideas"
qmd context add qmd://notes/work "Work-related documentation"
```

## Backend Selection

QMD automatically detects your platform:
- **Apple Silicon (M1/M2/M3/M4)**: Uses MLX backend (2-4x faster)
- **Other platforms**: Uses GGUF backend (node-llama-cpp)

You can see which backend is active:
```bash
qmd status
```

Look for the message: `[QMD] Using MLX backend (Apple Silicon detected)`

## MLX Models Used

When using MLX backend, these models are downloaded on first use (~2GB total):
- **Embedding**: `mlx-community/embeddinggemma-300m-4bit` (~300MB)
- **Reranking**: `mlx-community/Qwen3-Reranker-0.6B-mxfp8` (~640MB)
- **Query Expansion**: `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` (~1GB)

Models are cached in `~/.cache/huggingface/`

## Testing

Run the test suite:
```bash
# Unit tests
bun test

# Test MLX backend directly
python3 test_mlx_backend.py
```

## MCP Server (for AI Agents)

QMD includes an MCP (Model Context Protocol) server for integration with AI agents like Claude:

### Stdio Transport (Default)
```json
// ~/.claude/settings.json or Claude Desktop config
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
```

### HTTP Transport (Shared Server)
```bash
# Start HTTP server
qmd mcp --http          # localhost:8181
qmd mcp --http --port 8080  # custom port

# Background daemon
qmd mcp --http --daemon
qmd mcp stop           # stop daemon
```

Point your MCP client to: `http://localhost:8181/mcp`

## Troubleshooting

### MLX Backend Not Working

If you see errors about MLX backend:

1. **Verify Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the backend directly**:
   ```bash
   python3 test_mlx_backend.py
   ```

3. **Check if on Apple Silicon**:
   ```bash
   uname -m  # should show "arm64"
   ```

### SQLite Extension Issues

If you see sqlite-vec errors, ensure you have Homebrew SQLite:
```bash
brew install sqlite
```

The system uses Homebrew's SQLite which supports extension loading.

## Development

### Build from Source
```bash
# Install dependencies
bun install

# Build
bun run build

# Run locally
bun run qmd status
```

### Link Globally
```bash
npm link
# or
bun link

# Now 'qmd' command is available globally
qmd --help
```

## Next Steps

1. **Index your documents**: Start with a small collection to test
2. **Add context**: Use `qmd context add` to improve search relevance
3. **Try different search modes**: Compare `search`, `vsearch`, and `query`
4. **Configure MCP**: Set up with your AI agent for seamless integration

## Resources

- Full documentation: See [README.md](README.md)
- MLX backend details: See [MLX_BACKEND.md](MLX_BACKEND.md)
- Changelog: See [CHANGELOG.md](CHANGELOG.md)

## Performance

With MLX backend on Apple Silicon, you can expect:
- **Embedding**: ~2-4x faster than GGUF
- **Reranking**: ~2-3x faster than GGUF
- **Query expansion**: ~2x faster than GGUF

The speedup is most noticeable with batch operations and large document sets.
