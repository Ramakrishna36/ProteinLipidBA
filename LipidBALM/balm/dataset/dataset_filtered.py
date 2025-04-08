import pandas as pd
import numpy as np
from collections import defaultdict
from random import Random
from typing import Dict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

class ProteinLipidInteractionDataset:
    def __init__(self, csv_path):
        """
        Initialize the dataset from a CSV file.
        
        Args:
            csv_path (str): Path to the combined CSV file
        """
        # Read the CSV file
        self.data = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['LipidSMILES', 'ProteinSequence', 'BindingAffinityValue', 'AffinityType']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset")
        
        # Reset index to ensure clean indexing
        self.data = self.data.reset_index(drop=True)
        
        # Store binding affinity values
        self.y = self.data['BindingAffinityValue'].values

    def _create_scaffold_split(self, seed, frac):
        """
        Create scaffold split based on lipid SMILES scaffolds.
        
        Args:
            seed (int): Random seed for reproducibility
            frac (list): Fractions for train/valid/test splits
        
        Returns:
            dict: Splits of the dataset
        """
        random = Random(seed)

        # Generate scaffolds for each lipid molecule
        scaffolds = defaultdict(set)
        for idx, row in self.data.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['LipidSMILES'])
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
                scaffolds[scaffold].add(idx)
            except:
                continue

        # Split scaffolds into train, valid, test sets
        scaffolds = list(scaffolds.values())
        random.shuffle(scaffolds)

        train_size = int(len(self.data) * frac[0])
        valid_size = int(len(self.data) * frac[1])

        train, valid, test = set(), set(), set()
        for scaffold_set in scaffolds:
            if len(train) + len(scaffold_set) <= train_size:
                train.update(scaffold_set)
            elif len(valid) + len(scaffold_set) <= valid_size:
                valid.update(scaffold_set)
            else:
                test.update(scaffold_set)

        # Create DataFrame subsets for each split
        train_df = self.data.iloc[list(train)].reset_index(drop=True)
        valid_df = self.data.iloc[list(valid)].reset_index(drop=True)
        test_df = self.data.iloc[list(test)].reset_index(drop=True)

        return {
            "train": train_df,
            "valid": valid_df,
            "test": test_df,
        }

    def _create_random_split(self, seed, frac):
        """
        Create a random split of the dataset.
        
        Args:
            seed (int): Random seed for reproducibility
            frac (list): Fractions for train/valid/test splits
        
        Returns:
            dict: Splits of the dataset
        """
        _, val_frac, test_frac = frac
        
        # Set random seed
        np.random.seed(seed)
        
        # Sample test set
        test = self.data.sample(frac=test_frac, replace=False)
        
        # Remaining data for train and validation
        train_val = self.data[~self.data.index.isin(test.index)]
        
        # Sample validation set from remaining data
        val = train_val.sample(
            frac=val_frac / (1 - test_frac), replace=False
        )
        
        # Remaining data is train set
        train = train_val[~train_val.index.isin(val.index)]

        return {
            "train": train.reset_index(drop=True),
            "valid": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }

    def _create_cold_split(self, seed, frac, entity='LipidSMILES'):
        """
        Create a cold split where specific entities are exclusive to one set.
        
        Args:
            seed (int): Random seed for reproducibility
            frac (list): Fractions for train/valid/test splits
            entity (str): Column to base cold split on
        
        Returns:
            dict: Splits of the dataset
        """
        train_frac, val_frac, test_frac = frac

        # Set random seed
        np.random.seed(seed)

        # Get unique entities
        unique_entities = self.data[entity].unique()
        
        # Sample test entities
        test_entities = np.random.choice(
            unique_entities, 
            size=int(len(unique_entities) * test_frac), 
            replace=False
        )

        # Select test data
        test = self.data[self.data[entity].isin(test_entities)]

        # Remaining data for train and validation
        train_val = self.data[~self.data[entity].isin(test_entities)]

        # Sample validation entities from remaining entities
        val_entities = np.random.choice(
            train_val[entity].unique(), 
            size=int(len(train_val[entity].unique()) * val_frac / (1 - test_frac)), 
            replace=False
        )

        # Select validation data
        val = train_val[train_val[entity].isin(val_entities)]

        # Remaining data is train set
        train = train_val[~train_val[entity].isin(val_entities)]

        return {
            "train": train.reset_index(drop=True),
            "valid": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }

    def get_split(
        self, 
        method="random", 
        frac=[0.7, 0.2, 0.1], 
        seed=42, 
        entity='LipidSMILES'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get a dataset split based on the specified method.

        Args:
            method (str): The split method ('random', 'scaffold', 'cold').
            frac (list): A list of train/valid/test fractions, e.g., [0.7, 0.2, 0.1].
            seed (int): The random seed.
            entity (str): The column to base the split on for cold split.

        Returns:
            dict: A dictionary of split dataframes with keys 'train', 'valid', and 'test'.
        """
        if method == "random":
            return self._create_random_split(seed, frac)
        elif method == "scaffold":
            return self._create_scaffold_split(seed, frac)
        elif method == "cold":
            return self._create_cold_split(seed, frac, entity)
        else:
            raise ValueError(f"Unknown split method: {method}")

# Example usage
if __name__ == "__main__":
    # Specify the path to your combined CSV file
    csv_path = r"C:\Users\91967\Documents\Protein-Lipid interactions\ProteinLipidBA\LipidBALM\Dataset\combined_binding_data.csv"
    
    # Create dataset instance
    dataset = ProteinLipidInteractionDataset(csv_path)
    
    # Get different types of splits
    random_split = dataset.get_split(method="random")
    scaffold_split = dataset.get_split(method="scaffold")
    cold_split = dataset.get_split(method="cold", entity='LipidSMILES')
    
    # Print split sizes
    for split_name, split_data in random_split.items():
        print(f"{split_name} split size: {len(split_data)}")