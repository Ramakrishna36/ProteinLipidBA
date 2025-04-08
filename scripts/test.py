import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import time
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from balm import common_utils
from balm.configs import Configs
from balm.datasets.utils import DataCollatorWithPadding
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman, evaluate_predictions
from balm.models import LipidBALM, LipidBaselineModel
from balm.models.utils import load_trained_model, load_pretrained_pkd_bounds
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup


def argument_parser():
    """
    Parses command-line arguments for the LipidBALM testing script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="LipidBALM Testing")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--pkd_upper_bound", type=float, default=10.0, help="Upper bound for binding affinity scaling.")
    parser.add_argument("--pkd_lower_bound", type=float, default=2.0, help="Lower bound for binding affinity scaling.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/", help="Directory containing model checkpoints.")
    args = parser.parse_args()
    return args


def get_checkpoint_name(configs: Configs):
    """
    Constructs the checkpoint name based on the configuration hyperparameters.

    Args:
        configs (Configs): Configuration object containing model settings.

    Returns:
        str: The constructed checkpoint name.
    """
    protein_peft_hyperparameters = configs.model_configs.protein_peft_hyperparameters
    lipid_peft_hyperparameters = configs.model_configs.lipid_peft_hyperparameters

    # Build the run name based on the fine-tuning types and hyperparameters
    hyperparams = []
    hyperparams += [f"protein_{configs.model_configs.protein_fine_tuning_type}"]
    if protein_peft_hyperparameters:
        for key, value in protein_peft_hyperparameters.items():
            if key not in ["target_modules", "feedforward_modules"]:
                hyperparams += [f"{key}_{value}"]
    hyperparams += [f"lipid_{configs.model_configs.lipid_fine_tuning_type}"]
    if lipid_peft_hyperparameters:
        for key, value in lipid_peft_hyperparameters.items():
            if key not in ["target_modules", "feedforward_modules"]:
                hyperparams += [f"{key}_{value}"]
    hyperparams += [
        f"lr_{configs.model_configs.model_hyperparameters.learning_rate}",
        f"dropout_{configs.model_configs.model_hyperparameters.projected_dropout}",
        f"dim_{configs.model_configs.model_hyperparameters.projected_size}",
    ]
    run_name = "_".join(hyperparams)
    return run_name


def load_model(configs, checkpoint_dir):
    """
    Loads the model from the specified checkpoint directory.

    Args:
        configs (Configs): Configuration object containing model settings.
        checkpoint_dir (str): Directory where model checkpoints are stored.

    Returns:
        LipidBALM or LipidBaselineModel: The loaded and prepared model.
    """
    # Initialize the model based on configuration
    if (
        configs.model_configs.protein_fine_tuning_type == "baseline"
        and configs.model_configs.lipid_fine_tuning_type == "baseline"
    ):
        model = LipidBaselineModel(configs.model_configs)
    else:
        model = LipidBALM(configs.model_configs)
        
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_name = get_checkpoint_name(configs)
    print(f"Loading checkpoint from {os.path.join(checkpoint_dir, checkpoint_name)}")
    
    try:
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, checkpoint_name, "pytorch_model.bin"),
            map_location=device,
        )
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Checkpoint not found at {os.path.join(checkpoint_dir, checkpoint_name, 'pytorch_model.bin')}")
        print("Trying to load from HF hub using the load_trained_model function...")
        model = load_trained_model(model, configs.model_configs, is_training=False)

    model = model.eval()

    # Merge PEFT and base model if applicable
    if configs.model_configs.protein_fine_tuning_type in ["lora", "lokr", "loha", "ia3"]:
        print("Merging protein model with adapter")
        model.protein_model.merge_and_unload()
    if configs.model_configs.lipid_fine_tuning_type in ["lora", "lokr", "loha", "ia3"]:
        print("Merging lipid model with adapter")
        model.lipid_model.merge_and_unload()

    return model


def load_tokenizers(configs):
    """
    Loads the tokenizers for protein and lipid sequences.

    Args:
        configs (Configs): Configuration object containing model settings.

    Returns:
        tuple: A tuple containing the protein and lipid tokenizers.
    """
    protein_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.protein_model_name_or_path)
    lipid_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.lipid_model_name_or_path)

    return protein_tokenizer, lipid_tokenizer


def load_data(
    test_data,
    batch_size,
    protein_tokenizer,
    lipid_tokenizer,
    protein_max_seq_len,
    lipid_max_seq_len,
):
    """
    Loads and prepares the test dataset for evaluation.

    Args:
        test_data (str): Path to the test data CSV file.
        batch_size (int): Batch size for data loading.
        protein_tokenizer (PreTrainedTokenizer): Tokenizer for protein sequences.
        lipid_tokenizer (PreTrainedTokenizer): Tokenizer for lipid sequences.
        protein_max_seq_len (int): Maximum sequence length for protein tokens.
        lipid_max_seq_len (int): Maximum sequence length for lipid tokens.

    Returns:
        DataLoader: DataLoader for the prepared test dataset.
    """
    print(f"Loading test data from {test_data}")
    df = pd.read_csv(test_data)
    
    # Rename columns if needed to match expected format
    if "Target" in df.columns and "ProteinSequence" not in df.columns:
        df = df.rename(columns={"Target": "ProteinSequence"})
    if "Drug" in df.columns and "LipidSMILES" not in df.columns:
        df = df.rename(columns={"Drug": "LipidSMILES"})
    if "Y" in df.columns and "BindingAffinityValue" not in df.columns:
        df = df.rename(columns={"Y": "BindingAffinityValue"})
    
    # Ensure required columns exist
    required_columns = ["ProteinSequence", "LipidSMILES", "BindingAffinityValue"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the test dataset")
            
    print(f"Test dataset loaded with {len(df)} samples")
    
    # Pre-tokenize unique proteins and lipids
    protein_tokenized_dict, lipid_tokenized_dict = pre_tokenize_unique_entities(
        df,
        protein_tokenizer,
        lipid_tokenizer,
    )

    # Convert to Dataset format and apply tokenization
    dataset = Dataset.from_pandas(df).map(
        lambda x: tokenize_with_lookup(
            {"ProteinSequence": x["ProteinSequence"], "LipidSMILES": x["LipidSMILES"]},
            protein_tokenized_dict, 
            lipid_tokenized_dict
        ),
    )
    
    # Add binding affinity values as labels
    dataset = dataset.add_column("labels", df["BindingAffinityValue"].values)

    # Create data collator for batching
    data_collator = DataCollatorWithPadding(
        protein_tokenizer=protein_tokenizer,
        lipid_tokenizer=lipid_tokenizer,
        padding="max_length",
        protein_max_length=protein_max_seq_len,
        lipid_max_length=lipid_max_seq_len,
        return_tensors="pt",
    )

    print(f"Creating test DataLoader with batch size {batch_size}")
    dataloader = DataLoader(
        dataset,
        shuffle=False,  # No shuffling for test data
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    return dataloader


def compute_metrics(labels, predictions, pkd_upper_bound, pkd_lower_bound):
    """
    Computes performance metrics for binding affinity prediction.

    Args:
        labels (Tensor): Ground truth binding affinity values.
        predictions (Tensor): Model predictions.
        pkd_upper_bound (float): Upper bound for binding affinity scaling.
        pkd_lower_bound (float): Lower bound for binding affinity scaling.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Rescale predictions and labels back to the original binding affinity range
    pkd_range = pkd_upper_bound - pkd_lower_bound
    labels = (labels + 1) / 2 * pkd_range + pkd_lower_bound
    predictions = (predictions + 1) / 2 * pkd_range + pkd_lower_bound

    # Use the comprehensive metrics function
    return evaluate_predictions(labels, predictions)


def main():
    """
    Main function to execute the testing process for LipidBALM.

    It performs the following steps:
    1. Parses command-line arguments.
    2. Loads configuration settings and model checkpoints.
    3. Prepares the test dataset.
    4. Evaluates the model on the test data and computes performance metrics.
    """
    args = argument_parser()
    config_filepath = args.config_filepath
    configs = Configs(**common_utils.load_yaml(config_filepath))

    # Get maximum sequence lengths from config
    protein_max_seq_len = configs.model_configs.model_hyperparameters.protein_max_seq_len
    lipid_max_seq_len = configs.model_configs.model_hyperparameters.lipid_max_seq_len
    
    # Load tokenizers and model
    protein_tokenizer, lipid_tokenizer = load_tokenizers(configs)
    model = load_model(configs, args.checkpoint_dir)
    
    # Load and prepare test data
    dataloader = load_data(
        args.test_data,
        configs.training_configs.batch_size,
        protein_tokenizer,
        lipid_tokenizer,
        protein_max_seq_len,
        lipid_max_seq_len,
    )

    # Get binding affinity scaling bounds
    pkd_upper_bound = args.pkd_upper_bound
    pkd_lower_bound = args.pkd_lower_bound
    print(f"Using binding affinity bounds: [{pkd_lower_bound}, {pkd_upper_bound}]")

    # Evaluation loop
    start = time.time()
    all_proteins = []
    all_lipids = []
    all_labels = []
    all_predictions = []
    
    print("Starting evaluation...")
    device = next(model.parameters()).device
    
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            # Move batch to the same device as the model
            batch = {key: value.to(device) for key, value in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Collect results
            all_proteins.extend(batch["protein_ori_sequences"])
            all_lipids.extend(batch["lipid_ori_sequences"])
            all_labels.append(batch["labels"])
            
            if configs.model_configs.loss_function == "cosine_mse":
                all_predictions.append(outputs["cosine_similarity"])
            elif configs.model_configs.loss_function == "baseline_mse":
                all_predictions.append(outputs["logits"])
                
            # Print progress
            if step % 10 == 0:
                print(
                    f"Time elapsed: {time.time()-start:.2f}s | "
                    f"Processed: {step * configs.training_configs.batch_size}/{len(dataloader) * configs.training_configs.batch_size} samples"
                )
                
    end = time.time()
    print(f"Evaluation completed in {end - start:.2f} seconds")

    # Concatenate all tensors
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Compute performance metrics
    performance_metrics = compute_metrics(
        all_labels, all_predictions, pkd_upper_bound, pkd_lower_bound
    )
    
    # Print metrics
    print("\nPerformance Metrics:")
    for metric_name, metric_value in performance_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
        
    # Save predictions to CSV
    result_df = pd.DataFrame({
        "protein": all_proteins,
        "lipid": all_lipids,
        "label": all_labels.cpu().numpy(),
        "prediction": all_predictions.cpu().numpy()
    })
    
    # Apply scaling if using cosine_mse
    if configs.model_configs.loss_function == "cosine_mse":
        pkd_range = pkd_upper_bound - pkd_lower_bound
        result_df["label"] = (result_df["label"] + 1) / 2 * pkd_range + pkd_lower_bound
        result_df["prediction"] = (result_df["prediction"] + 1) / 2 * pkd_range + pkd_lower_bound
    
    # Add absolute error column
    result_df["abs_error"] = abs(result_df["label"] - result_df["prediction"])
    
    # Save results
    output_file = "test_predictions.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    main()