#!/usr/bin/env python3
"""
MLX Backend for qmd-mlx

This script provides a JSON-based stdin/stdout protocol for running MLX models on Apple Silicon.
It loads embedding, reranking, and text generation models and handles commands from Node.js.

Protocol:
- Input: JSON objects on stdin, one per line
- Output: JSON responses on stdout
- Errors: Logged to stderr, error responses on stdout

Commands:
- embed: Generate embedding for a single text
- embedBatch: Generate embeddings for multiple texts
- rerank: Rerank documents by relevance to a query
- expandQuery: Generate query variations for search
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('mlx_backend')

# Model configurations
EMBED_MODEL = "mlx-community/embeddinggemma-300m-4bit"
RERANK_MODEL = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"
GENERATE_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "qmd" / "mlx-models"


class MLXBackend:
    """Manages MLX models and handles commands."""
    
    def __init__(self):
        self.embed_model = None
        self.embed_tokenizer = None
        self.rerank_model = None
        self.rerank_tokenizer = None
        self.generate_model = None
        self.generate_tokenizer = None
    
    def _tokenize(self, tokenizer, texts):
        """Tokenize texts, handling different tokenizer types."""
        # Try to get the underlying tokenizer if wrapped
        if hasattr(tokenizer, 'tokenizer'):
            tok = tokenizer.tokenizer
        elif hasattr(tokenizer, '_tokenizer'):
            tok = tokenizer._tokenizer
        else:
            tok = tokenizer
        
        # Try different tokenization methods
        try:
            # Standard HuggingFace tokenizer API
            if hasattr(tok, '__call__'):
                return tok(texts, return_tensors="np", padding=True, truncation=True)
        except Exception:
            pass
        
        try:
            # Try encode method
            if hasattr(tok, 'encode'):
                if isinstance(texts, list):
                    encoded = [tok.encode(t) for t in texts]
                    # Create attention mask and convert to dict format
                    import numpy as np
                    max_len = max(len(e) for e in encoded)
                    padded = [e + [0] * (max_len - len(e)) for e in encoded]
                    attention_mask = [[1] * len(e) + [0] * (max_len - len(e)) for e in encoded]
                    return {
                        'input_ids': np.array(padded, dtype=np.int32),
                        'attention_mask': np.array(attention_mask, dtype=np.float16)
                    }
                else:
                    encoded = tok.encode(texts)
                    import numpy as np
                    return {
                        'input_ids': np.array([encoded], dtype=np.int32),
                        'attention_mask': np.array([[1] * len(encoded)], dtype=np.float16)
                    }
        except Exception as e:
            raise ValueError(f"Could not tokenize with available methods: {e}")
        
    def initialize(self) -> Dict[str, Any]:
        """Load all models. Returns status."""
        try:
            logger.info("Initializing MLX backend...")
            
            # Import MLX libraries (lazy import to fail fast if not available)
            try:
                import mlx.core as mx
                from mlx_embeddings import load as load_embedding_model
                from mlx_lm import load as load_lm_model
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"MLX libraries not available: {e}. Install with: pip install -r requirements.txt"
                }
            
            # Store for later use
            self.mx = mx
            
            # Load embedding model
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            try:
                self.embed_model, self.embed_tokenizer = load_embedding_model(EMBED_MODEL)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return {"success": False, "error": f"Failed to load embedding model: {e}"}
            
            # Load reranking model (uses same mlx-embeddings library)
            logger.info(f"Loading reranking model: {RERANK_MODEL}")
            try:
                self.rerank_model, self.rerank_tokenizer = load_embedding_model(RERANK_MODEL)
                logger.info("Reranking model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranking model: {e}")
                return {"success": False, "error": f"Failed to load reranking model: {e}"}
            
            # Load generation model for query expansion
            logger.info(f"Loading generation model: {GENERATE_MODEL}")
            try:
                self.generate_model, self.generate_tokenizer = load_lm_model(GENERATE_MODEL)
                logger.info("Generation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load generation model: {e}")
                return {"success": False, "error": f"Failed to load generation model: {e}"}
            
            logger.info("All models loaded successfully")
            return {
                "success": True,
                "models": {
                    "embed": EMBED_MODEL,
                    "rerank": RERANK_MODEL,
                    "generate": GENERATE_MODEL
                }
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def embed(self, text: str, is_query: bool = False, title: Optional[str] = None) -> Dict[str, Any]:
        """Generate embedding for a single text."""
        try:
            import mlx.core as mx
            
            # Format text with appropriate prefix (matching EmbeddingGemma format)
            if is_query:
                formatted_text = f"task: search result | query: {text}"
            else:
                formatted_text = f"title: {title or 'none'} | text: {text}"
            
            # Tokenize
            inputs = self._tokenize(self.embed_tokenizer, [formatted_text])
            
            # Generate embedding - MLX models expect 'inputs' parameter
            if 'input_ids' in inputs:
                input_ids = mx.array(inputs['input_ids'])
                # Pass attention_mask if available, ensuring it's float16
                if 'attention_mask' in inputs:
                    attention_mask = mx.array(inputs['attention_mask']).astype(mx.float16)
                    output = self.embed_model(inputs=input_ids, attention_mask=attention_mask)
                else:
                    output = self.embed_model(inputs=input_ids)
            else:
                output = self.embed_model(**{k: mx.array(v) for k, v in inputs.items()})
            
            # Get embedding (mean pooling over sequence)
            if hasattr(output, 'last_hidden_state'):
                embeddings = output.last_hidden_state
            elif hasattr(output, 'pooler_output'):
                embeddings = output.pooler_output
            else:
                embeddings = output[0] if isinstance(output, tuple) else output
            
            # Mean pool and normalize
            embedding = mx.mean(embeddings[0], axis=0)
            embedding = embedding / mx.linalg.norm(embedding)
            embedding_list = embedding.tolist()
            
            return {
                "success": True,
                "embedding": embedding_list,
                "model": EMBED_MODEL
            }
        except Exception as e:
            logger.error(f"Embed error: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def embed_batch(self, texts: List[str], is_query: bool = False, titles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate embeddings for multiple texts."""
        try:
            import mlx.core as mx
            
            # Format texts with appropriate prefixes
            formatted_texts = []
            for i, text in enumerate(texts):
                if is_query:
                    formatted_texts.append(f"task: search result | query: {text}")
                else:
                    title = titles[i] if titles and i < len(titles) else "none"
                    formatted_texts.append(f"title: {title} | text: {text}")
            
            # Tokenize all texts
            inputs = self._tokenize(self.embed_tokenizer, formatted_texts)
            
            # Generate embeddings - MLX models expect 'inputs' parameter
            if 'input_ids' in inputs:
                input_ids = mx.array(inputs['input_ids'])
                # Pass attention_mask if available, ensuring it's float16
                if 'attention_mask' in inputs:
                    attention_mask = mx.array(inputs['attention_mask']).astype(mx.float16)
                    output = self.embed_model(inputs=input_ids, attention_mask=attention_mask)
                else:
                    output = self.embed_model(inputs=input_ids)
            else:
                output = self.embed_model(**{k: mx.array(v) for k, v in inputs.items()})
            
            # Get embeddings
            if hasattr(output, 'last_hidden_state'):
                embeddings_mx = output.last_hidden_state
            elif hasattr(output, 'pooler_output'):
                embeddings_mx = output.pooler_output
            else:
                embeddings_mx = output[0] if isinstance(output, tuple) else output
            
            # Mean pool and normalize each embedding
            embeddings = []
            for i in range(len(formatted_texts)):
                emb = mx.mean(embeddings_mx[i], axis=0)
                emb = emb / mx.linalg.norm(emb)
                embeddings.append(emb.tolist())
            
            return {
                "success": True,
                "embeddings": embeddings,
                "model": EMBED_MODEL
            }
        except Exception as e:
            logger.error(f"EmbedBatch error: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def rerank(self, query: str, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Rerank documents by relevance to query."""
        try:
            import mlx.core as mx
            
            # Embed query
            query_formatted = f"task: search result | query: {query}"
            query_inputs = self._tokenize(self.embed_tokenizer, [query_formatted])
            if 'input_ids' in query_inputs:
                input_ids = mx.array(query_inputs['input_ids'])
                # Pass attention_mask if available, ensuring it's float16
                if 'attention_mask' in query_inputs:
                    attention_mask = mx.array(query_inputs['attention_mask']).astype(mx.float16)
                    query_output = self.embed_model(inputs=input_ids, attention_mask=attention_mask)
                else:
                    query_output = self.embed_model(inputs=input_ids)
            else:
                query_output = self.embed_model(**{k: mx.array(v) for k, v in query_inputs.items()})
            
            if hasattr(query_output, 'last_hidden_state'):
                query_embeddings = query_output.last_hidden_state
            elif hasattr(query_output, 'pooler_output'):
                query_embeddings = query_output.pooler_output
            else:
                query_embeddings = query_output[0] if isinstance(query_output, tuple) else query_output
            
            query_emb = mx.mean(query_embeddings[0], axis=0)
            query_emb = query_emb / mx.linalg.norm(query_emb)
            
            # Embed documents
            doc_formatted = [f"title: {doc.get('title', 'none')} | text: {doc.get('text', '')}" for doc in documents]
            doc_inputs = self._tokenize(self.embed_tokenizer, doc_formatted)
            if 'input_ids' in doc_inputs:
                input_ids = mx.array(doc_inputs['input_ids'])
                # Pass attention_mask if available, ensuring it's float16
                if 'attention_mask' in doc_inputs:
                    attention_mask = mx.array(doc_inputs['attention_mask']).astype(mx.float16)
                    doc_output = self.embed_model(inputs=input_ids, attention_mask=attention_mask)
                else:
                    doc_output = self.embed_model(inputs=input_ids)
            else:
                doc_output = self.embed_model(**{k: mx.array(v) for k, v in doc_inputs.items()})
            
            if hasattr(doc_output, 'last_hidden_state'):
                doc_embeddings = doc_output.last_hidden_state
            elif hasattr(doc_output, 'pooler_output'):
                doc_embeddings = doc_output.pooler_output
            else:
                doc_embeddings = doc_output[0] if isinstance(doc_output, tuple) else doc_output
            
            # Compute cosine similarities
            scores = []
            for i in range(len(documents)):
                doc_emb = mx.mean(doc_embeddings[i], axis=0)
                doc_emb = doc_emb / mx.linalg.norm(doc_emb)
                # Normalized dot product (cosine similarity)
                similarity = mx.sum(query_emb * doc_emb).item()
                scores.append(float(similarity))
            
            # Create results with original indices
            results = [
                {
                    "file": documents[i].get("file", f"doc_{i}"),
                    "score": scores[i],
                    "index": i
                }
                for i in range(len(documents))
            ]
            
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "success": True,
                "results": results,
                "model": RERANK_MODEL
            }
        except Exception as e:
            logger.error(f"Rerank error: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def expand_query(self, query: str, context: Optional[str] = None, include_lexical: bool = True) -> Dict[str, Any]:
        """Generate query variations for search."""
        try:
            from mlx_lm import generate
            
            # Create prompt for query expansion
            prompt = f"Expand this search query into variations: {query}\n\nGenerate 2-3 alternative search queries. Format each as:\ntype: text\n\nWhere type is 'lex' for lexical/keyword search, 'vec' for semantic search, or 'hyde' for hypothetical document."
            
            # Generate variations
            response = generate(
                self.generate_model,
                self.generate_tokenizer,
                prompt=prompt,
                max_tokens=200,
                temp=0.7
            )
            
            # Parse response into queryables
            # Expected format: "type: text\n"
            lines = response.strip().split('\n')
            queryables = []
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        qtype = parts[0].strip()
                        qtext = parts[1].strip()
                        
                        # Validate type
                        if qtype in ['lex', 'vec', 'hyde']:
                            # Filter out if lexical not requested
                            if not include_lexical and qtype == 'lex':
                                continue
                            
                            queryables.append({
                                "type": qtype,
                                "text": qtext
                            })
            
            # Fallback if parsing failed
            if not queryables:
                queryables = [
                    {"type": "vec", "text": query},
                    {"type": "hyde", "text": f"Information about {query}"}
                ]
                if include_lexical:
                    queryables.insert(0, {"type": "lex", "text": query})
            
            return {
                "success": True,
                "queryables": queryables,
                "model": GENERATE_MODEL
            }
        except Exception as e:
            logger.error(f"ExpandQuery error: {e}")
            # Fallback on error
            queryables = [
                {"type": "vec", "text": query}
            ]
            if include_lexical:
                queryables.insert(0, {"type": "lex", "text": query})
            
            return {
                "success": True,
                "queryables": queryables,
                "model": GENERATE_MODEL,
                "note": f"Fallback used due to error: {str(e)}"
            }
    
    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single command and return response."""
        cmd_type = command.get("command")
        
        if cmd_type == "initialize":
            return self.initialize()
        
        elif cmd_type == "embed":
            text = command.get("text", "")
            is_query = command.get("isQuery", False)
            title = command.get("title")
            return self.embed(text, is_query, title)
        
        elif cmd_type == "embedBatch":
            texts = command.get("texts", [])
            is_query = command.get("isQuery", False)
            titles = command.get("titles")
            return self.embed_batch(texts, is_query, titles)
        
        elif cmd_type == "rerank":
            query = command.get("query", "")
            documents = command.get("documents", [])
            return self.rerank(query, documents)
        
        elif cmd_type == "expandQuery":
            query = command.get("query", "")
            context = command.get("context")
            include_lexical = command.get("includeLexical", True)
            return self.expand_query(query, context, include_lexical)
        
        elif cmd_type == "ping":
            return {"success": True, "message": "pong"}
        
        elif cmd_type == "shutdown":
            return {"success": True, "message": "shutting down"}
        
        else:
            return {"success": False, "error": f"Unknown command: {cmd_type}"}


def main():
    """Main loop: read commands from stdin, write responses to stdout."""
    backend = MLXBackend()
    logger.info("MLX Backend started, waiting for commands...")
    
    # Send ready signal
    print(json.dumps({"ready": True}), flush=True)
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse command
                command = json.loads(line)
                
                # Handle command
                response = backend.handle_command(command)
                
                # Send response
                print(json.dumps(response), flush=True)
                
                # Check for shutdown
                if command.get("command") == "shutdown":
                    logger.info("Shutdown requested, exiting...")
                    break
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                response = {"success": False, "error": f"Invalid JSON: {str(e)}"}
                print(json.dumps(response), flush=True)
            
            except Exception as e:
                logger.error(f"Error handling command: {e}")
                logger.error(traceback.format_exc())
                response = {"success": False, "error": str(e)}
                print(json.dumps(response), flush=True)
    
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting...")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
