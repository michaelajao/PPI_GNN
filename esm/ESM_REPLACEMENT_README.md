# ESM Embeddings for Protein-Protein Interaction Prediction

This document explains the changes made to replace the `bio_embeddings` package with ESM (Evolutionary Scale Modeling) for protein sequence embeddings.

## Overview of Changes

1. **Replaced `bio_embeddings` with ESM**:
   - Created `esm_embedding.py` to provide a similar interface to the original `SeqVecEmbedder`
   - Updated model dimensions from 1024 to 320 (ESM-2 t6 model's embedding dimension)
   - Modified `proteins_to_graphs.py` to use ESM embeddings

2. **Added new functionality**:
   - Created `generate_esm_embeddings.py` for batch processing of protein sequences
   - Updated environment file (`ppi_env_updated.yml`) with required dependencies

## Installation

1. If the conda environment already exists, update it with the required dependencies:
   ```bash
   conda activate ppi_env
   pip install fair-esm
   ```

2. Alternatively, you can create a new environment if needed:
   ```bash
   conda env create -f ppi_env_updated.yml
   conda activate ppi_env
   ```

## Usage

### Generating Embeddings

You can generate ESM embeddings for protein sequences using the `generate_esm_embeddings.py` script:

1. For PDB files:
   ```bash
   python generate_esm_embeddings.py --pdb_dir Human_features/raw --output Human_features/pdb_to_esm_dict.npy
   ```

2. For FASTA files:
   ```bash
   python generate_esm_embeddings.py --fasta /path/to/sequences.fasta --output Human_features/pdb_to_esm_dict.npy
   ```

### Training the Model

The training process remains the same:

1. Prepare the data:
   ```bash
   python data_prepare.py
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Evaluate on test set:
   ```bash
   python test.py
   ```

## Technical Details

### ESM vs. SeqVec

ESM (Evolutionary Scale Modeling) is a more modern protein language model compared to SeqVec:

- **Better performance**: ESM models have shown superior performance on various protein prediction tasks
- **Active development**: ESM is actively maintained by Meta AI Research
- **Direct PyTorch integration**: Seamless integration with PyTorch
- **Multiple model sizes**: Options from small (8M parameters) to large (3B parameters)

### Implementation Details

1. **ESMEmbedder class**:
   - Provides `embed()` and `embed_batch()` methods similar to the original SeqVecEmbedder
   - Automatically handles tokenization and model inference
   - Returns fixed-size embeddings by summing across sequence length

2. **Model Architecture Updates**:
   - Updated input dimensions in GCNN and AttGNN models to match ESM embedding size
   - Rest of the architecture remains unchanged

3. **Protein Graph Generation**:
   - Modified to use ESM embeddings while maintaining the same graph structure
   - Fallback to one-hot encoding if embeddings are not available

## Troubleshooting

If you encounter issues:

1. **Memory errors**: Try using a smaller ESM model by changing the model name in `esm_embedding.py`
2. **Missing embeddings**: Run the `generate_esm_embeddings.py` script to pre-generate embeddings
3. **Dimension mismatch**: Ensure the model's `num_features_pro` parameter matches the ESM model's embedding dimension

## References

- [ESM GitHub Repository](https://github.com/facebookresearch/esm)
- [ESM Paper](https://www.pnas.org/content/118/15/e2016239118)
