# Install bio_embeddings using the command: pip install bio-embeddings[all]

from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
import numpy as np
import torch 

seq = 'MVTYDFGSDEMHD' # A protein sequence of length L

embedder = SeqVecEmbedder()
embedding = embedder.embed(seq)
protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
np_arr = protein_embd.cpu().detach().numpy()





# """
# SeqVec embeddings for protein sequences using ProtBERT model from transformers.
# This provides similar contextualized embeddings as the original SeqVec.
# """

# import torch
# import numpy as np
# from typing import List, Optional
# import os
# from tqdm import tqdm
# from transformers import BertModel, BertTokenizer
# import torch_geometric

# # Legacy support for older PyG data objects
# class LegacyData:
#     def __init__(self, data_dict):
#         """Load legacy PyG data from dictionary format"""
#         self.x = torch.tensor(data_dict['x']) if 'x' in data_dict else None
#         self.edge_index = torch.tensor(data_dict['edge_index']) if 'edge_index' in data_dict else None
#         self.edge_attr = torch.tensor(data_dict['edge_attr']) if 'edge_attr' in data_dict else None
#         self.pos = torch.tensor(data_dict['pos']) if 'pos' in data_dict else None
#         self.batch = torch.tensor(data_dict['batch']) if 'batch' in data_dict else None

# def load_legacy_data(filepath):
#     """Load PyG data with legacy support"""
#     try:
#         # First try loading normally
#         data = torch.load(filepath)
#         if hasattr(data, 'x'):
#             return data
#     except:
#         pass
        
#     try:
#         # Try loading as dictionary
#         data_dict = torch.load(filepath, map_location='cpu')
#         if isinstance(data_dict, dict):
#             return LegacyData(data_dict)
#     except:
#         pass
        
#     raise ValueError(f"Could not load data from {filepath}")

# class SeqVecEmbedder:
#     def __init__(self, model_name: str = "Rostlab/prot_bert"):
#         """
#         Initialize the SeqVec embedder using ProtBERT.
        
#         Args:
#             model_name: Name of the pretrained model to use
#         """
#         print(f"Loading model: {model_name}")
#         self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
#         self.model = BertModel.from_pretrained(model_name)
#         self.model.eval()
        
#         # Move to GPU if available
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()
            
#         print("Model loaded successfully")
        
#     def embed(self, sequence: str) -> np.ndarray:
#         """
#         Generate embeddings for a single protein sequence.
        
#         Args:
#             sequence: Amino acid sequence string
            
#         Returns:
#             numpy array of embeddings with shape [L x 1024] where L is sequence length
#         """
#         # Add spaces between amino acids as required by ProtBert
#         sequence = " ".join(sequence)
        
#         # Tokenize
#         encoded = self.tokenizer.encode_plus(
#             sequence,
#             add_special_tokens=True,
#             padding=True,
#             return_tensors='pt'
#         )
        
#         # Move to same device as model
#         if torch.cuda.is_available():
#             encoded = {k: v.cuda() for k, v in encoded.items()}
        
#         # Generate embeddings
#         with torch.no_grad():
#             outputs = self.model(**encoded)
#             embeddings = outputs.last_hidden_state
            
#         # Remove special tokens and convert to numpy
#         embeddings = embeddings[:, 1:-1].cpu().numpy()
#         return embeddings[0]  # Remove batch dimension
    
#     def embed_batch(self, sequences: List[str]) -> List[np.ndarray]:
#         """
#         Generate embeddings for a batch of sequences.
        
#         Args:
#             sequences: List of amino acid sequence strings
            
#         Returns:
#             List of numpy arrays, each with shape [L x 1024] where L is sequence length
#         """
#         # Add spaces between amino acids
#         sequences = [" ".join(seq) for seq in sequences]
        
#         # Tokenize
#         encoded = self.tokenizer.batch_encode_plus(
#             sequences,
#             add_special_tokens=True,
#             padding=True,
#             return_tensors='pt'
#         )
        
#         if torch.cuda.is_available():
#             encoded = {k: v.cuda() for k, v in encoded.items()}
            
#         # Generate embeddings
#         embeddings_list = []
#         with torch.no_grad():
#             outputs = self.model(**encoded)
#             embeddings = outputs.last_hidden_state
            
#             # Process each sequence
#             for i, seq in enumerate(sequences):
#                 # Get sequence length (excluding special tokens)
#                 seq_len = len(seq.split())
                
#                 # Extract embeddings for this sequence (removing special tokens)
#                 seq_emb = embeddings[i, 1:seq_len+1].cpu().numpy()
#                 embeddings_list.append(seq_emb)
                
#         return embeddings_list
    
#     def embed_and_pool(self, sequence: str) -> np.ndarray:
#         """
#         Generate embeddings and pool across sequence length for fixed-size representation.
        
#         Args:
#             sequence: Amino acid sequence string
            
#         Returns:
#             numpy array of embeddings with shape [1024]
#         """
#         # Get embeddings
#         embeddings = self.embed(sequence)
        
#         # Pool across sequence length (mean pooling)
#         pooled = np.mean(embeddings, axis=0)
#         return pooled

# def process_dataset(dataset_path: str, output_file: str):
#     """Process all proteins in a dataset and generate embeddings"""
#     print(f"\nProcessing {dataset_path} dataset...")
    
#     # Get list of processed protein files
#     processed_dir = os.path.join(dataset_path, "processed")
#     protein_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
#     protein_ids = [os.path.splitext(f)[0] for f in protein_files]
    
#     print(f"Found {len(protein_ids)} proteins in {processed_dir}")
    
#     # Load existing embeddings if available
#     try:
#         embeddings = dict(np.load(output_file, allow_pickle=True).item())
#         print(f"Loaded {len(embeddings)} existing embeddings from {output_file}")
#     except:
#         embeddings = {}
#         print("Starting fresh with new embeddings dictionary")
    
#     # Initialize embedder
#     embedder = SeqVecEmbedder()
    
#     # Process each protein file that doesn't have embeddings yet
#     new_count = 0
#     for protein_id in tqdm(protein_ids):
#         if protein_id in embeddings:
#             continue
            
#         pt_path = os.path.join(processed_dir, f"{protein_id}.pt")
#         try:
#             # Load the processed protein data with legacy support
#             protein_data = load_legacy_data(pt_path)
            
#             if protein_data.x is None:
#                 print(f"Warning: No node features found for {protein_id}")
#                 continue
                
#             # Extract sequence from node features
#             # Assuming one-hot encoded sequence in first 20 dimensions
#             node_features = protein_data.x.cpu().numpy()
#             sequence = ""
#             amino_acids = "ACDEFGHIKLMNPQRSTVWY"
#             for node in node_features:
#                 aa_index = np.argmax(node[:20])  # First 20 features are one-hot amino acids
#                 sequence += amino_acids[aa_index]
            
#             # Generate embedding
#             embedding = embedder.embed_and_pool(sequence)
#             embeddings[protein_id] = embedding
#             new_count += 1
            
#             # Save periodically
#             if new_count % 10 == 0:
#                 np.save(output_file, embeddings, allow_pickle=True)
#                 print(f"Progress: Generated {new_count} new embeddings")
                
#         except Exception as e:
#             print(f"Error generating embedding for {protein_id}: {str(e)}")
    
#     # Final save
#     if new_count > 0:
#         np.save(output_file, embeddings, allow_pickle=True)
#         print(f"\nCompleted. Generated {new_count} new embeddings")
#     else:
#         print("\nNo new embeddings needed to be generated")
    
#     print(f"Total embeddings available: {len(embeddings)}")
#     return embeddings

# def main():
#     # Process Human dataset
#     human_embeddings = process_dataset(
#         "Human_features",
#         "Human_features/seqvec_embeddings.npy"
#     )
    
#     # Process Yeast dataset
#     yeast_embeddings = process_dataset(
#         "S. cerevisiae",
#         "S. cerevisiae/seqvec_embeddings.npy"
#     )

# if __name__ == "__main__":
#     main()
