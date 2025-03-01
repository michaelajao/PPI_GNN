# note that this custom dataset is not prepared on the top of geometric Dataset(pytorch's inbuilt)
import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

processed_dir="Human_features/processed/"
npy_file = "Human_features/npy_file_new(human_dataset).npy"
npy_ar = np.load(npy_file)
print(npy_ar.shape)

class LabelledDataset(Dataset):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.load(npy_file)
      self.processed_dir = processed_dir
      # Filter out pairs where either protein file doesn't exist
      valid_pairs = []
      for i in range(len(self.npy_ar)):
          prot_1_id = self.npy_ar[i,2]
          prot_2_id = self.npy_ar[i,5]
          prot_1_exists = any(f.lower() == (prot_1_id + ".pt").lower() for f in os.listdir(processed_dir))
          prot_2_exists = any(f.lower() == (prot_2_id + ".pt").lower() for f in os.listdir(processed_dir))
          if prot_1_exists and prot_2_exists:
              valid_pairs.append(i)
      
      self.valid_indices = valid_pairs
      self.protein_1 = self.npy_ar[valid_pairs, 2]
      self.protein_2 = self.npy_ar[valid_pairs, 5]
      self.label = self.npy_ar[valid_pairs, 6].astype(float)
      self.n_samples = len(valid_pairs)
      print(f"Found {self.n_samples} valid protein pairs out of {len(self.npy_ar)}")

    def __len__(self):
      return self.n_samples

    def __getitem__(self, index):
      prot_1_id = self.protein_1[index]
      prot_2_id = self.protein_2[index]
      
      # Find the correct case-sensitive filename
      prot_1_file = next(f for f in os.listdir(self.processed_dir) if f.lower() == (prot_1_id + ".pt").lower())
      prot_2_file = next(f for f in os.listdir(self.processed_dir) if f.lower() == (prot_2_id + ".pt").lower())
      
      prot_1_path = os.path.join(self.processed_dir, prot_1_file)
      prot_2_path = os.path.join(self.processed_dir, prot_2_file)

      # Load raw tensors and bypass PyG version check
      try:
          def extract_tensors(filepath):
              data_obj = torch.load(filepath, map_location='cpu')
              # Try to access _store directly for newer PyG versions
              if hasattr(data_obj, '_store'):
                  store = data_obj._store
                  if 'x' in store and 'edge_index' in store:
                      return store['x'], store['edge_index']
              
              # For older PyG versions, data might be stored differently
              if hasattr(data_obj, 'x') and hasattr(data_obj, 'edge_index'):
                  # Access attributes via __dict__ to bypass version check
                  return data_obj.__dict__['x'], data_obj.__dict__['edge_index']
              
              # Handle raw dictionary format
              if isinstance(data_obj, dict):
                  return data_obj['x'], data_obj['edge_index']
                  
              # Last resort: try to access raw attributes
              raw_dict = vars(data_obj)
              for x_key in ['x', '_x', 'features']:
                  for edge_key in ['edge_index', '_edge_index', 'edges']:
                      if x_key in raw_dict and edge_key in raw_dict:
                          return raw_dict[x_key], raw_dict[edge_key]
                          
              raise ValueError(f"Could not extract tensors from {filepath}")

          x1, edge_index1 = extract_tensors(prot_1_path)
          x2, edge_index2 = extract_tensors(prot_2_path)
              
      except Exception as e:
          print(f"Error loading proteins {prot_1_id} and {prot_2_id}: {str(e)}")
          raise
          
      # Ensure we have tensors
      if not isinstance(x1, torch.Tensor):
          x1 = torch.tensor(x1)
      if not isinstance(edge_index1, torch.Tensor):
          edge_index1 = torch.tensor(edge_index1)
      if not isinstance(x2, torch.Tensor):
          x2 = torch.tensor(x2)
      if not isinstance(edge_index2, torch.Tensor):
          edge_index2 = torch.tensor(edge_index2)
          
      return (x1.clone().detach(), edge_index1.clone().detach()), (x2.clone().detach(), edge_index2.clone().detach()), torch.tensor(self.label[index])

def collate_fn(batch):
    proteins_1, proteins_2, labels = zip(*batch)
    x1s, edge_index1s = zip(*proteins_1)
    x2s, edge_index2s = zip(*proteins_2)
    
    # Convert to lists of tensors
    x1s = list(x1s)
    edge_index1s = list(edge_index1s)
    x2s = list(x2s)
    edge_index2s = list(edge_index2s)
    
    return (x1s, edge_index1s), (x2s, edge_index2s), torch.stack(labels)

dataset = LabelledDataset(npy_file=npy_file, processed_dir=processed_dir)

# Use the filtered dataset size for splitting
filtered_size = len(dataset)
print("Size after filtering: ")
print(filtered_size)
seed = 42
torch.manual_seed(seed)
trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * filtered_size), filtered_size - math.floor(0.8 * filtered_size)])

trainloader = DataLoader(dataset=trainset, batch_size=4, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(dataset=testset, batch_size=4, num_workers=0, collate_fn=collate_fn)
print("Length")
print(len(trainloader))
print(len(testloader))
