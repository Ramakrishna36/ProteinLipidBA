import pandas as pd
from transformers import PreTrainedTokenizer


def pre_tokenize_unique_entities(
    dataset: pd.DataFrame,
    protein_tokenizer: PreTrainedTokenizer,
    lipid_tokenizer: PreTrainedTokenizer,
):
    """
    Pre-tokenizes unique protein sequences and lipid SMILES in the dataset.
    """
    unique_proteins = dataset["ProteinSequence"].unique().tolist()
    unique_lipids = dataset["LipidSMILES"].unique().tolist()

    print(f"Tokenizing {len(unique_proteins)} unique proteins...")
    
    # Add more aggressive truncation and handling
    valid_proteins = []
    for protein in unique_proteins:
        try:
            # Convert to string and strip
            protein = str(protein).strip()
            
            # Truncate to max length, leaving room for special tokens
            max_length = protein_tokenizer.model_max_length - 2
            if len(protein) > max_length:
                protein = protein[:max_length]
            
            valid_proteins.append(protein)
        except Exception as e:
            print(f"Error processing protein sequence: {protein}")
            print(f"Error details: {e}")
    
    # Similar handling for lipids
    valid_lipids = []
    for lipid in unique_lipids:
        try:
            # Convert to string and strip
            lipid = str(lipid).strip()
            
            # Truncate to max length, leaving room for special tokens
            max_length = lipid_tokenizer.model_max_length - 2
            if len(lipid) > max_length:
                lipid = lipid[:max_length]
            
            valid_lipids.append(lipid)
        except Exception as e:
            print(f"Error processing lipid SMILES: {lipid}")
            print(f"Error details: {e}")

    # Tokenize with more flexible padding
    tokenized_proteins = protein_tokenizer(
        valid_proteins, 
        padding=True,
        truncation=True,
        max_length=protein_tokenizer.model_max_length
    )
    
    tokenized_lipids = lipid_tokenizer(
        valid_lipids,
        padding=True,
        truncation=True,
        max_length=lipid_tokenizer.model_max_length
    )
    
    protein_tokenized_dict = {
        protein: {
            "input_ids": tokenized_protein_input_ids,
            "attention_mask": tokenized_protein_attention_mask,
        }
        for protein, tokenized_protein_input_ids, tokenized_protein_attention_mask in zip(
            valid_proteins,
            tokenized_proteins["input_ids"],
            tokenized_proteins["attention_mask"],
        )
    }

    lipid_tokenized_dict = {
        lipid: {
            "input_ids": tokenized_lipid_input_ids,
            "attention_mask": tokenized_lipid_attention_mask,
        }
        for lipid, tokenized_lipid_input_ids, tokenized_lipid_attention_mask in zip(
            valid_lipids,
            tokenized_lipids["input_ids"],
            tokenized_lipids["attention_mask"],
        )
    }

    return protein_tokenized_dict, lipid_tokenized_dict


def tokenize_with_lookup(examples, protein_tokenized_dict, lipid_tokenized_dict):
    """
    Retrieves pre-tokenized protein and lipid sequences using lookup dictionaries.

    Args:
        examples (dict): A dictionary containing the 'ProteinSequence' and 'LipidSMILES' keys.
        protein_tokenized_dict (dict): Dictionary with pre-tokenized protein sequences.
        lipid_tokenized_dict (dict): Dictionary with pre-tokenized lipid SMILES.

    Returns:
        dict: A dictionary with the original sequences, tokenized input IDs, and attention masks.
    """
    protein_input = protein_tokenized_dict[examples["ProteinSequence"]]
    lipid_input = lipid_tokenized_dict[examples["LipidSMILES"]]

    return {
        "protein_ori_sequences": examples["ProteinSequence"],
        "lipid_ori_sequences": examples["LipidSMILES"],
        "protein_input_ids": protein_input["input_ids"],
        "lipid_input_ids": lipid_input["input_ids"],
        "protein_attention_mask": protein_input["attention_mask"],
        "lipid_attention_mask": lipid_input["attention_mask"],
    }


def batch_tokenize_dataset(
    dataset: pd.DataFrame,
    protein_tokenizer: PreTrainedTokenizer,
    lipid_tokenizer: PreTrainedTokenizer,
    batch_size: int = 32
):
    """
    Tokenizes an entire dataset in batches to prevent memory issues with large datasets.
    
    Args:
        dataset (pd.DataFrame): DataFrame containing protein and lipid data
        protein_tokenizer (PreTrainedTokenizer): Tokenizer for protein sequences
        lipid_tokenizer (PreTrainedTokenizer): Tokenizer for lipid SMILES
        batch_size (int): Number of samples to process in each batch
        
    Returns:
        dict: Dictionary with all tokenized inputs for the dataset
    """
    result = {
        "protein_ori_sequences": [],
        "lipid_ori_sequences": [],
        "protein_input_ids": [],
        "lipid_input_ids": [],
        "protein_attention_mask": [],
        "lipid_attention_mask": [],
        "binding_affinity_values": [],
        "affinity_types": []
    }
    
    # Pre-tokenize unique entities to avoid redundant tokenization
    protein_tokenized_dict, lipid_tokenized_dict = pre_tokenize_unique_entities(
        dataset, protein_tokenizer, lipid_tokenizer
    )
    
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Processing dataset in {total_batches} batches...")
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            # Get tokenized inputs from lookup dictionaries
            tokenized = tokenize_with_lookup(
                {
                    "ProteinSequence": row["ProteinSequence"],
                    "LipidSMILES": row["LipidSMILES"]
                },
                protein_tokenized_dict,
                lipid_tokenized_dict
            )
            
            # Add to result dictionary
            result["protein_ori_sequences"].append(tokenized["protein_ori_sequences"])
            result["lipid_ori_sequences"].append(tokenized["lipid_ori_sequences"])
            result["protein_input_ids"].append(tokenized["protein_input_ids"])
            result["lipid_input_ids"].append(tokenized["lipid_input_ids"])
            result["protein_attention_mask"].append(tokenized["protein_attention_mask"])
            result["lipid_attention_mask"].append(tokenized["lipid_attention_mask"])
            result["binding_affinity_values"].append(row["BindingAffinityValue"])
            result["affinity_types"].append(row["AffinityType"])
            
        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i // batch_size + 1}/{total_batches}")
    
    return result