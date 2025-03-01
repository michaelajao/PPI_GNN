"""
Script to generate ESM embeddings from existing processed protein files.
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from esm_embedding import ESMEmbedder

class CompatData:
    """Compatibility wrapper for older PyG data format"""
    def __init__(self, data_obj):
        # If the data object is already a modern PyG Data object with stores, return as is
        if hasattr(data_obj, 'stores'):
            self.x = data_obj.x
            return
            
        # Otherwise, create a compatibility wrapper for older format
        self.x = data_obj.x if hasattr(data_obj, 'x') else None
        self.edge_index = data_obj.edge_index if hasattr(data_obj, 'edge_index') else None
        self.batch = data_obj.batch if hasattr(data_obj, 'batch') else None

def process_dataset(dataset_path, embedder, output_file):
    """Process all protein files in a dataset and generate embeddings"""
    print(f"\nProcessing {dataset_path} dataset...")
    
    # Get list of processed protein files
    processed_dir = os.path.join(dataset_path, "processed")
    protein_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    protein_ids = [os.path.splitext(f)[0] for f in protein_files]
    
    print(f"Found {len(protein_ids)} proteins in {processed_dir}")
    
    # Load existing embeddings if available
    try:
        embeddings = dict(np.load(output_file, allow_pickle=True).item())
        print(f"Loaded {len(embeddings)} existing embeddings from {output_file}")
    except:
        embeddings = {}
        print("Starting fresh with new embeddings dictionary")
    
    # Process each protein file that doesn't have embeddings yet
    new_count = 0
    for protein_id in tqdm(protein_ids):
        if protein_id in embeddings:
            continue
            
        pt_path = os.path.join(processed_dir, f"{protein_id}.pt")
        try:
            # Load the processed protein data with compatibility wrapper
            protein_data = CompatData(torch.load(pt_path))
            
            if protein_data.x is None:
                print(f"Warning: No node features found for {protein_id}")
                continue
                
            # Generate embedding from node features
            # We'll use all node features to create a rich sequence representation
            embedding = embedder.embed_features(protein_data.x)
            embeddings[protein_id] = embedding
            new_count += 1
            
            # Save periodically
            if new_count % 10 == 0:
                np.save(output_file, embeddings, allow_pickle=True)
                print(f"Progress: Generated {new_count} new embeddings")
                
        except Exception as e:
            print(f"Error generating embedding for {protein_id}: {str(e)}")
    
    # Final save
    if new_count > 0:
        np.save(output_file, embeddings, allow_pickle=True)
        print(f"\nCompleted. Generated {new_count} new embeddings")
    else:
        print("\nNo new embeddings needed to be generated")
    
    print(f"Total embeddings available: {len(embeddings)}")
    return embeddings

def main():
    # Initialize ESM embedder
    print("Initializing ESM embedder...")
    embedder = ESMEmbedder()
    
    # Process Human dataset
    human_embeddings = process_dataset(
        "Human_features",
        embedder,
        "Human_features/pdb_to_esm_dict.npy"
    )
    
    # Process Yeast dataset
    yeast_embeddings = process_dataset(
        "S. cerevisiae",
        embedder,
        "S. cerevisiae/pdb_to_esm_dict.npy"
    )

if __name__ == "__main__":
    main()
