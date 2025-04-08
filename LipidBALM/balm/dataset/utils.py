import torch
from transformers import PreTrainedTokenizer

class DataCollatorWithPadding:
    """
    Data collator that handles padding for protein and lipid sequences.
    
    Args:
        protein_tokenizer (PreTrainedTokenizer): Tokenizer for protein sequences
        lipid_tokenizer (PreTrainedTokenizer): Tokenizer for lipid SMILES
        padding (str, optional): Padding strategy. Defaults to "max_length"
        protein_max_length (int, optional): Maximum length for protein sequences
        lipid_max_length (int, optional): Maximum length for lipid sequences
        return_tensors (str, optional): Type of tensors to return. Defaults to "pt"
    """
    def __init__(
        self, 
        protein_tokenizer: PreTrainedTokenizer, 
        lipid_tokenizer: PreTrainedTokenizer, 
        padding: str = "max_length",
        protein_max_length: int = 1024,
        lipid_max_length: int = 512,
        return_tensors: str = "pt"
    ):
        self.protein_tokenizer = protein_tokenizer
        self.lipid_tokenizer = lipid_tokenizer
        self.padding = padding
        self.protein_max_length = protein_max_length
        self.lipid_max_length = lipid_max_length
        self.return_tensors = return_tensors

    def __call__(self, features):
        """
        Collate and pad a batch of features.
        
        Args:
            features (list): List of dictionary features from the dataset
        
        Returns:
            dict: Batched and padded features
        """
        # Separate different types of features
        protein_sequences = [feature["protein_ori_sequences"] for feature in features]
        lipid_sequences = [feature["lipid_ori_sequences"] for feature in features]
        
        # Tokenize protein sequences
        protein_inputs = self.protein_tokenizer(
            protein_sequences, 
            padding=self.padding, 
            truncation=True, 
            max_length=self.protein_max_length,
            return_tensors=self.return_tensors
        )
        
        # Tokenize lipid sequences
        lipid_inputs = self.lipid_tokenizer(
            lipid_sequences, 
            padding=self.padding, 
            truncation=True, 
            max_length=self.lipid_max_length,
            return_tensors=self.return_tensors
        )
        
        # Prepare the batch
        batch = {
            "protein_input_ids": protein_inputs["input_ids"],
            "protein_attention_mask": protein_inputs["attention_mask"],
            "lipid_input_ids": lipid_inputs["input_ids"],
            "lipid_attention_mask": lipid_inputs["attention_mask"],
            "protein_ori_sequences": protein_sequences,
            "lipid_ori_sequences": lipid_sequences
        }
        
        # Add labels if present in the original features
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([feature["labels"] for feature in features])
        
        return batch