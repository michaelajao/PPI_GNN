# ESM embeddings for protein sequences
import torch
import numpy as np
from typing import List, Union, Tuple, Dict

class ESMEmbedder:
    def __init__(self, model_name="esm2_t6_8M_UR50D"):
        """
        Initialize the ESM embedder with the specified model.
        
        Args:
            model_name: Name of the ESM model to use. Default is "esm2_t6_8M_UR50D" (smallest ESM-2 model).
        """
        try:
            import esm
        except ImportError:
            raise ImportError(
                "ESM is not installed. Please install it with: pip install fair-esm"
            )
        
        print(f"Loading ESM model: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # Set to evaluation mode
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Get embedding dimension from model
        # Different ESM versions have different ways to access the embedding dimension
        try:
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'embed_dim'):
                self.embedding_dim = self.model.args.embed_dim
            elif hasattr(self.model, 'embed_dim'):
                self.embedding_dim = self.model.embed_dim
            else:
                # Default for ESM-2 t6 model
                self.embedding_dim = 320
        except:
            # Fallback to default
            self.embedding_dim = 320
            
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def embed(self, sequence: str) -> np.ndarray:
        """
        Generate embeddings for a single protein sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            numpy array of embeddings with shape [1024]
        """
        # Prepare data
        data = [(0, sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # Generate embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            token_embeddings = results["representations"][self.model.num_layers]
        
        # Remove cls and eos tokens
        embeddings = token_embeddings[0, 1:-1]
        
        # Sum across sequence length for fixed size representation
        protein_embedding = embeddings.sum(dim=0)
        
        return protein_embedding.cpu().numpy()

    def embed_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of protein sequences.
        
        Args:
            sequences: List of amino acid sequence strings
            
        Returns:
            numpy array of embeddings with shape [batch_size, 1024]
        """
        # Prepare batch data
        data = [(i, seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # Generate embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            token_embeddings = results["representations"][self.model.num_layers]
        
        # Process each sequence in batch
        embeddings = []
        for i in range(len(sequences)):
            seq_emb = token_embeddings[i, 1:-1]  # Remove cls and eos
            protein_emb = seq_emb.sum(dim=0)
            embeddings.append(protein_emb)
            
        return torch.stack(embeddings).cpu().numpy()

    def embed_features(self, node_features: torch.Tensor) -> np.ndarray:
        """
        Generate embeddings from existing node features.
        
        Args:
            node_features: Tensor of shape [num_nodes, num_features] containing protein features
            
        Returns:
            numpy array of embeddings with shape [embedding_dim]
        """
        with torch.no_grad():
            # Use mean pooling over node features
            embedding = node_features.mean(dim=0)
            
            # Project to embedding dimension if needed
            if embedding.shape[0] != self.embedding_dim:
                projection = torch.nn.Linear(embedding.shape[0], self.embedding_dim)
                embedding = projection(embedding)
        
        return embedding.cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Example sequence
    seq = 'MVTYDFGSDEMHD'
    
    # Initialize embedder
    embedder = ESMEmbedder()
    
    # Generate embedding
    embedding = embedder.embed(seq)
    print(f"Embedding shape: {embedding.shape}")
    
    # Convert to tensor if needed
    protein_embd = torch.tensor(embedding)
    print(f"Tensor shape: {protein_embd.shape}")
