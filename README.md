# PPI_GNN

## Replication Steps

1. Download required feature files:

    - For human features: Place files in `../human_features/processed/` (See `PPI_GNN/Human_features/README.md` for link)
    - For S. cerevisiae PPI dataset: Place files in `../S. cerevisiae/processed/` (See `PPI_GNN/S. cerevisiae/README.md` for link)

1. Train the model:

    ```bash
    python train.py
    ```

## Using the Model with New Dataset

1. Generate node features:

- Use SeqVec method (`seqvec_embedding.py`) for protein sequences
- Build protein graph (`proteins_to_graphs.py`)

2. Prepare input features:

    ```bash
    python data_prepare.py
    ```

1. Train the model:

    ```bash
    python train.py
    ```

1. Evaluate on test set:

    ```bash
    python test.py
    ```

## Environment Setup

Create the environment using:
```bash
conda env create -f ppi_env.yml
```