from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class FineTuningType(str, Enum):
    """
    baseline: Only for baseline model (Concatenated Embedding + linear projection)
    projection: Common fine tuning technique: only tuning Linear projection
    """
    baseline = "baseline"
    projection = "projection"
    lora = "lora"
    lokr = "lokr"
    loha = "loha"
    ia3 = "ia3"


class ModelHyperparameters(BaseModel):
    learning_rate: float = 0.001
    protein_max_seq_len: int = 1024
    lipid_max_seq_len: int = 512    # Changed from drug_max_seq_len to lipid_max_seq_len
    warmup_steps_ratio: float = 0.06
    gradient_accumulation_steps: int = 32
    projected_size: int = 256
    projected_dropout: float = 0.5
    relu_before_cosine: bool = False
    init_noise_sigma: float = 1
    sigma_lr: float = 0.01


class ModelConfigs(BaseModel):
    protein_model_name_or_path: str
    lipid_model_name_or_path: str   # Changed from drug_model_name_or_path to lipid_model_name_or_path
    checkpoint_path: Optional[str] = None
    model_hyperparameters: ModelHyperparameters
    protein_fine_tuning_type: Optional[FineTuningType]
    lipid_fine_tuning_type: Optional[FineTuningType]    # Changed from drug_fine_tuning_type to lipid_fine_tuning_type
    protein_peft_hyperparameters: Optional[dict]
    lipid_peft_hyperparameters: Optional[dict]    # Changed from drug_peft_hyperparameters to lipid_peft_hyperparameters
    loss_function: str


class DatasetConfigs(BaseModel):
    dataset_name: str = "ProteinLipid"    # Default to your dataset
    csv_path: str                         # Added to specify the path to your combined CSV file
    split_method: str = "random"          # Default split method
    frac: List[float] = [0.7, 0.2, 0.1]   # Split fractions for train/valid/test
    seed: int = 42                        # Random seed for splits
    entity: Optional[str] = "LipidSMILES" # Entity for cold split


class TrainingConfigs(BaseModel):
    random_seed: int = 1234
    device: int = 0
    epochs: int = 30                      # Increased from 1 to a more realistic number
    batch_size: int = 32                  # Increased for potentially faster training
    patience: int = 10                    # Reduced for earlier stopping
    min_delta: float = 0.005              # Changed to float to match typical usage
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    model_configs: ModelConfigs
    dataset_configs: DatasetConfigs
    training_configs: TrainingConfigs