import os
from typing import Union
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb

from balm.configs import Configs
#from balm.datasets import ProteinLipidInteractionDataset
#from balm.datasets.utils import DataCollatorWithPadding
from balm.models import LipidBaselineModel, LipidBALM
#from balm.models.utils import load_trained_model, load_pretrained_pkd_bounds
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman, evaluate_predictions
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup, batch_tokenize_dataset
from balm.common_utils import setup_experiment_folder, setup_device, setup_random_seed, save_dataset_split_info
# Replace the current import
# from balm.datasets import ProteinLipidInteractionDataset
from balm.dataset.dataset_filtered import ProteinLipidInteractionDataset
from balm.dataset.utils import DataCollatorWithPadding  # Changed 'datasets' to 'dataset'
from balm.utils import load_trained_model, load_pretrained_pkd_bounds


class LipidTrainer:
    """
    The LipidTrainer class handles the training, validation, and testing processes for protein-lipid models.
    It supports setting up datasets, initializing models, and managing the training loop with
    early stopping and learning rate scheduling.

    Attributes:
        configs (Configs): Configuration object with all necessary hyperparameters and settings.
        wandb_entity (str): Weights & Biases entity name.
        wandb_project (str): Weights & Biases project name.
        outputs_dir (str): Directory where output files such as checkpoints and logs are saved.
    """

    def __init__(
        self, configs: Configs, wandb_entity: str, wandb_project: str, outputs_dir: str
    ):
        """
        Initialize the Trainer with the provided configurations, Weights & Biases settings, 
        and output directory.

        Args:
            configs (Configs): Configuration object.
            wandb_entity (str): Weights & Biases entity name.
            wandb_project (str): Weights & Biases project name.
            outputs_dir (str): Directory where outputs are saved.
        """
        self.configs = configs

        self.dataset_configs = self.configs.dataset_configs
        self.training_configs = self.configs.training_configs
        self.model_configs = self.configs.model_configs

        self.gradient_accumulation_steps = (
            self.model_configs.model_hyperparameters.gradient_accumulation_steps
        )
        self.protein_max_seq_len = (
            self.model_configs.model_hyperparameters.protein_max_seq_len
        )
        self.lipid_max_seq_len = (
            self.model_configs.model_hyperparameters.lipid_max_seq_len
        )

        self.outputs_dir = outputs_dir
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Set random seed for reproducibility
        setup_random_seed(self.training_configs.random_seed)
        
        # Setup device for training
        self.device = setup_device(self.training_configs.device)

        # Load the tokenizers for protein and lipid sequences
        self.protein_tokenizer, self.lipid_tokenizer = self._load_tokenizers()

        # Determine which model to use based on fine-tuning type
        if (
            self.model_configs.protein_fine_tuning_type == "baseline"
            and self.model_configs.lipid_fine_tuning_type == "baseline"
        ):
            self.model = LipidBaselineModel(self.model_configs)
        else:
            self.model = LipidBALM(self.model_configs)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self._setup_run_name()

    def _load_tokenizers(self):
        """
        Load the tokenizers for protein and lipid sequences based on the model paths.

        Returns:
            Tuple: (protein_tokenizer, lipid_tokenizer)
        """
        protein_tokenizer = AutoTokenizer.from_pretrained(
            self.model_configs.protein_model_name_or_path
        )
        lipid_tokenizer = AutoTokenizer.from_pretrained(
            self.model_configs.lipid_model_name_or_path
        )

        return protein_tokenizer, lipid_tokenizer

    def set_pkd_bounds(self, dataset):
        """
        Set the pKd bounds for scaling the labels in the dataset. If a checkpoint is loaded 
        for a zero-shot experiment, the bounds are loaded from the checkpoint.

        Args:
            dataset (ProteinLipidInteractionDataset): The dataset containing the binding affinity values.
        """
        # Extract all binding affinity values
        binding_values = dataset.data['BindingAffinityValue'].values
        
        self.pkd_lower_bound = min(binding_values)
        self.pkd_upper_bound = max(binding_values)

        # Load pKd bounds from a trained model if performing a zero-shot experiment
        if self.model_configs.checkpoint_path:
            if self.dataset_configs.frac[0] == 0.0:  # No training data
                self.pkd_lower_bound, self.pkd_upper_bound = load_pretrained_pkd_bounds(
                    self.model_configs.checkpoint_path
                )
        
        print(
            f"Scaling binding affinity values: from {self.pkd_lower_bound} - {self.pkd_upper_bound} to -1 to 1"
        )

    def set_dataset(self):
        """
        Prepare and set up the dataset for training, validation, and testing. This includes
        loading the dataset, creating splits, pre-tokenization, filtering based on sequence length, 
        and setting up DataLoaders.

        Returns:
            dict: Dictionary containing the dataset splits (train, valid, test).
        """
        # Load the dataset
        print(f"Loading dataset from {self.dataset_configs.csv_path}")
        dataset = ProteinLipidInteractionDataset(self.dataset_configs.csv_path)

        print(
            f"Training with {self.model_configs.loss_function} loss function."
        )

        # Apply binding affinity scaling if using cosine MSE loss
        if self.model_configs.loss_function == "cosine_mse":
            self.set_pkd_bounds(dataset)

            if self.pkd_upper_bound == self.pkd_lower_bound:
                # Handle case where all labels are the same
                dataset.data['BindingAffinityValue'] = 0
            else:
                # Scale binding affinity values to [-1, 1] range
                dataset.data['BindingAffinityValue'] = dataset.data['BindingAffinityValue'].apply(
                    lambda x: (x - self.pkd_lower_bound) / (self.pkd_upper_bound - self.pkd_lower_bound) * 2 - 1
                )
        elif self.model_configs.loss_function in ["baseline_mse"]:
            print("Using original binding affinity values")

        # Create dataset splits
        print(f"Creating dataset splits using {self.dataset_configs.split_method} method")
        dataset_splits = dataset.get_split(
            method=self.dataset_configs.split_method,
            frac=self.dataset_configs.frac,
            seed=self.dataset_configs.seed,
            entity=self.dataset_configs.entity
        )
        
        # Save split information for reproducibility
        save_dataset_split_info(dataset_splits, self.outputs_dir)

        # Filter the dataset by sequence length
        print("Preparing dataset splits")
        print(f"Protein max length: {self.protein_max_seq_len}")
        print(f"Lipid max length: {self.lipid_max_seq_len}")

        # Process each split
        processed_splits = {}
        for split_name, split_df in dataset_splits.items():
            print(f"Processing {split_name} split with {len(split_df)} samples")
            
            # Pre-tokenize unique lipids and proteins for this split
            print(f"Pre-tokenizing unique proteins and lipids for {split_name}")
            protein_tokenized_dict, lipid_tokenized_dict = pre_tokenize_unique_entities(
                split_df,
                self.protein_tokenizer,
                self.lipid_tokenizer,
            )

            # Tokenize the dataset and filter by sequence length
            split_dataset = Dataset.from_pandas(split_df).map(
                lambda x: tokenize_with_lookup(
                    {
                        "ProteinSequence": x["ProteinSequence"],
                        "LipidSMILES": x["LipidSMILES"]
                    },
                    protein_tokenized_dict, 
                    lipid_tokenized_dict
                ),
            )
            
            num_original_dataset = len(split_dataset)
            
            # Increase max lengths or remove strict filtering
            split_dataset = split_dataset.filter(
                lambda example: (
                    len(example.get("protein_input_ids", [])) <= self.protein_max_seq_len and
                    len(example.get("lipid_input_ids", [])) <= self.lipid_max_seq_len
                 )
            )

            num_filtered_dataset = len(split_dataset)
            
            # Add binding affinity values to the dataset
            split_dataset = split_dataset.add_column(
                "labels", 
                split_df["BindingAffinityValue"].values[:num_filtered_dataset]
            )
            
            print(
                f"Number of filtered pairs: "
                f"{num_filtered_dataset}/{num_original_dataset} "
                f"({float(num_filtered_dataset)/num_original_dataset*100:.2f}%)"
                
            )
            
            # If no samples remain, raise an informative error
            if num_filtered_dataset == 0:
                raise ValueError(
                    "No samples remain after filtering. "
                    "Check your sequence length constraints and input data. "
                    f"Protein max length: {self.protein_max_seq_len}, "
                    f"Lipid max length: {self.lipid_max_seq_len}"
            )

            processed_splits[split_name] = split_dataset

        # Create data collator to handle padding during batching
        data_collator = DataCollatorWithPadding(
            protein_tokenizer=self.protein_tokenizer,
            lipid_tokenizer=self.lipid_tokenizer,
            padding="max_length",
            protein_max_length=self.protein_max_seq_len,
            lipid_max_length=self.lipid_max_seq_len,
            return_tensors="pt",
        )

        # Setup DataLoaders for train, valid, and test splits
        if "train" in processed_splits:
            print(f"Setting up Train DataLoader")
            self.train_dataloader = DataLoader(
                processed_splits["train"],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=self.training_configs.batch_size,
                pin_memory=True,
            )
        if "valid" in processed_splits:
            print(f"Setting up Valid DataLoader")
            self.valid_dataloader = DataLoader(
                processed_splits["valid"],
                shuffle=False,
                collate_fn=data_collator,
                batch_size=self.training_configs.batch_size,
                pin_memory=True,
            )
        print(f"Setting up Test DataLoader")
        self.test_dataloader = DataLoader(
            processed_splits["test"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.training_configs.batch_size,
            pin_memory=True,
        )
        
        return dataset_splits

    def _setup_run_name(self):
        """
        Setup the run name and group name for the Weights & Biases tracker based on
        the dataset, split method, and model hyperparameters.
        """
        protein_peft_hyperparameters = (
            self.model_configs.protein_peft_hyperparameters
        )
        lipid_peft_hyperparameters = self.model_configs.lipid_peft_hyperparameters

        # Group name depends on the dataset and split method
        self.group_name = f"ProteinLipid_{self.dataset_configs.split_method}"

        # Run name depends on the fine-tuning type and other relevant hyperparameters
        hyperparams = []
        hyperparams += [f"protein_{self.model_configs.protein_fine_tuning_type}"]
        if protein_peft_hyperparameters:
            for key, value in protein_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        hyperparams += [f"lipid_{self.model_configs.lipid_fine_tuning_type}"]
        if lipid_peft_hyperparameters:
            for key, value in lipid_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        self.run_name = "_".join(hyperparams)

    def setup_training(self):
        """
        Setup the training environment, including initializing the Accelerator, WandB tracker, 
        optimizer, and learning rate scheduler. Prepares the model and dataloaders for training.
        """
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="wandb",
        )
        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.wandb_entity,
                        "name": self.run_name,
                        "group": self.group_name,
                    }
                },
                config=self.configs.dict(),
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

        if self.train_dataloader is not None:
            # Initialize optimizer with parameters that require gradients
            self.optimizer = AdamW(
                params=[
                    param
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                    and "noise_sigma" not in name  # Handle Balanced MSE loss
                ],
                lr=self.model_configs.model_hyperparameters.learning_rate,
            )

            # Setup learning rate scheduler
            num_training_steps = (
                len(self.train_dataloader) * self.training_configs.epochs
            )
            warmup_steps = int(
                num_training_steps * self.model_configs.model_hyperparameters.warmup_steps_ratio
            )
            print(f"Total training steps: {num_training_steps}, Warmup steps: {warmup_steps}")
            
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

            # Prepare model, dataloaders, optimizer, and scheduler for training
            (
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            # If only testing, prepare the model and test dataloader
            (
                self.model,
                self.test_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.test_dataloader,
            )

        # Load a trained model from checkpoint if specified
        if self.model_configs.checkpoint_path:
            load_trained_model(self.model, self.model_configs, is_training=self.train_dataloader is not None)

    def compute_metrics(self, labels, predictions):
        """
        Compute evaluation metrics including RMSE, Pearson, Spearman, and CI.

        Args:
            labels (Tensor): True labels.
            predictions (Tensor): Predicted values.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        if self.model_configs.loss_function in [
            "cosine_mse"
        ]:
            # Rescale predictions and labels back to the original binding affinity range
            pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
            labels = (labels + 1) / 2 * pkd_range + self.pkd_lower_bound
            predictions = (predictions + 1) / 2 * pkd_range + self.pkd_lower_bound

        # Compute comprehensive metrics
        metrics = evaluate_predictions(labels, predictions)
        
        return metrics

    def train(self):
        """
        Execute the training loop, handling early stopping, checkpoint saving, and logging metrics.
        """
        if self.train_dataloader is None:
            epoch = 0
            best_checkpoint_dir = None
            print("No training data provided. Proceeding to evaluation only.")
        else:
            best_loss = float('inf')
            patience = self.training_configs.patience
            min_delta = self.training_configs.min_delta
            eval_train_every_n_epochs = max(1, self.training_configs.epochs // 4)
            epochs_no_improve = 0  # Initialize early stopping counter
            best_checkpoint_dir = ""

            print("Trainable parameters:")
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.numel()} parameters")
                    trainable_count += param.numel()
            print(f"Total trainable parameters: {trainable_count:,}")

            for epoch in range(self.training_configs.epochs):
                self.model.train()

                num_train_steps = len(self.train_dataloader)
                progress_bar = tqdm(
                    total=int(num_train_steps // self.gradient_accumulation_steps),
                    position=0,
                    leave=True,
                    disable=not self.accelerator.is_local_main_process,
                )
                total_train_loss = 0
                for train_step, batch in enumerate(self.train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(batch)
                        loss = outputs["loss"]

                        # Backpropagation
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.model.zero_grad()
                        self.optimizer.zero_grad()

                        progress_bar.set_description(f"Epoch {epoch+1}/{self.training_configs.epochs}; Loss: {loss:.4f}")
                        total_train_loss += loss.detach().float()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)

                # Evaluate training performance periodically
                if (epoch + 1) % eval_train_every_n_epochs == 0:
                    train_metrics = self.evaluate("train")
                else:
                    train_metrics = {
                        "train/loss": total_train_loss / len(self.train_dataloader)
                    }
                
                # Evaluate validation performance
                valid_metrics = self.evaluate("valid")

                if valid_metrics:
                    current_loss = valid_metrics["valid/loss"]
                    print(f"Epoch {epoch+1}: valid_loss = {current_loss:.4f}, best_loss = {best_loss:.4f}")
                    
                    # Check for improvement
                    if current_loss < best_loss - min_delta:
                        best_loss = current_loss
                        epochs_no_improve = 0
                        # Save the model
                        best_checkpoint_dir = f"epoch_{epoch+1}"
                        checkpoint_path = os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
                        print(f"Saving checkpoint to {checkpoint_path}")
                        self.accelerator.save_state(checkpoint_path)
                    else:
                        epochs_no_improve += 1
                        print(f"No improvement for {epochs_no_improve} epochs")
                else:
                    # Just train until the last epoch
                    print("No validation metrics available, continuing training")
                    epochs_no_improve = 0

                # Log metrics to WandB
                self.accelerator.log({**train_metrics, **valid_metrics}, step=epoch)

                # Early stopping check
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Reload the best model checkpoint
            if best_checkpoint_dir:
                checkpoint_path = os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
                print(f"Loading best model from {checkpoint_path}")
                self.accelerator.load_state(checkpoint_path)
                self.accelerator.wait_for_everyone()

        # Compute test metrics and log results
        test_metrics = self.evaluate("test", save_prediction=True)
        self.accelerator.log(test_metrics, step=epoch)

        # For completeness, also evaluate on train and validation
        if self.train_dataloader is not None:
            train_metrics = self.evaluate("train", save_prediction=True)
            self.accelerator.log(train_metrics, step=epoch)
        
        if self.valid_dataloader is not None:
            valid_metrics = self.evaluate("valid", save_prediction=True)
            self.accelerator.log(valid_metrics, step=epoch)

        if best_checkpoint_dir:
            print(f"Creating a WandB artifact from {best_checkpoint_dir}")
            artifact = wandb.Artifact(best_checkpoint_dir, type="model")
            artifact.add_dir(
                os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
            )
            wandb.log_artifact(artifact)

    def evaluate(self, split: str, save_prediction=False):
        """
        Evaluate the model on the specified dataset split and optionally save predictions.

        Args:
            split (str): The dataset split to evaluate on ('train', 'valid', 'test').
            save_prediction (bool): Whether to save the predictions as a CSV file.

        Returns:
            dict: Dictionary containing the evaluation metrics for the specified split.
        """
        if split == "train":
            dataloader = self.train_dataloader
        elif split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        if dataloader is None:
            return {}

        total_loss = 0
        all_proteins = []
        all_lipids = []
        all_labels = []
        all_predictions = []

        self.model.eval()

        num_steps = len(dataloader)
        progress_bar = tqdm(
            total=num_steps,
            position=0,
            leave=True,
            disable=not self.accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = self.model(batch)
                loss = outputs["loss"]
                total_loss += loss.detach().float()

                # Collect predictions and labels for metric computation
                all_proteins += batch["protein_ori_sequences"]
                all_lipids += batch["lipid_ori_sequences"]
                if self.model_configs.loss_function == "cosine_mse":
                    all_labels.append(batch["labels"])
                    all_predictions.append(outputs["cosine_similarity"])
                elif self.model_configs.loss_function == "baseline_mse":
                    all_labels.append(batch["labels"])
                    all_predictions.append(outputs["logits"])

            progress_bar.set_description(f"Eval: {split} split")
            progress_bar.update(1)

        # Concatenate all predictions and labels across batches
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Calculate performance metrics
        performance_metrics = self.compute_metrics(all_labels, all_predictions)
        
        # Prepare metrics dictionary
        metrics = {
            f"{split}/loss": total_loss / len(dataloader),
        }
        for metric_name, metric_value in performance_metrics.items():
            metrics[f"{split}/{metric_name}"] = metric_value

        if save_prediction:
            # Save predictions and labels to a CSV file
            df = pd.DataFrame()
            df["protein"] = all_proteins
            df["lipid"] = all_lipids

            if self.model_configs.loss_function == "cosine_mse":
                # Rescale predictions and labels back to the original binding affinity range
                pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
                scaled_labels = (all_labels + 1) / 2 * pkd_range + self.pkd_lower_bound
                scaled_predictions = (all_predictions + 1) / 2 * pkd_range + self.pkd_lower_bound
                
                df["label"] = scaled_labels.cpu().numpy().tolist()
                df["prediction"] = scaled_predictions.cpu().numpy().tolist()
            else:
                df["label"] = all_labels.cpu().numpy().tolist()
                df["prediction"] = all_predictions.cpu().numpy().tolist()
            
            # Add the absolute error column
            df["abs_error"] = abs(df["label"] - df["prediction"])
            
            # Save to CSV
            csv_path = os.path.join(self.outputs_dir, f"{split}_prediction.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved predictions to {csv_path}")

            # Log the predictions as a WandB artifact
            if self.accelerator.is_main_process and self.wandb_tracker:
                artifact = wandb.Artifact(f"{split}_prediction", type="prediction")
                artifact.add_file(csv_path)
                wandb.log_artifact(artifact)

        return metrics