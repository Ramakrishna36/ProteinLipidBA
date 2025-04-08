import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv
import wandb
import pandas as pd
import numpy as np
import torch
import gpytorch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from balm import common_utils
from balm.models.utils import load_trained_model, load_pretrained_pkd_bounds
from balm.configs import Configs
from balm.models import LipidBALM


def argument_parser():
    """
    Parse the command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments with options such as dataset path, embedding type,
                            test size, learning rate, and number of epochs.
    """
    parser = argparse.ArgumentParser(
        description="Train Gaussian Process models on ECFP8 and ChemBERTa embeddings for protein-lipid interactions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/combined_binding_data.csv",
        help="Path to the protein-lipid dataset file.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["ECFP8", "ChemBERTa", "LipidBALM-lipid", "LipidBALM-concat", "LipidBALM-sum", "LipidBALM-subtract", "LipidBALM-cosine"],
        help="Type of embedding on which the GP is trained on",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="The ratio of the test set."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for data splitting."
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training iterations."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the script in debug mode."
    )

    return parser.parse_args()


def smiles_to_ecfp8_fingerprint(smiles_list):
    """
    Convert SMILES strings to ECFP8 Morgan Fingerprints.

    Args:
        smiles_list (list of str): A list of SMILES strings.

    Returns:
        np.ndarray: A NumPy array containing ECFP8 fingerprints for each SMILES.
    """
    fingerprints = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fingerprints.append(
                morgan_gen.GetFingerprint(mol)
            )  # ECFP8
        else:
            fingerprints.append(np.zeros(2048))  # Handle invalid molecules
            print(f"Warning: Could not parse SMILES: {smi}")
    return np.array(fingerprints)


def smiles_to_chemberta_embedding(smiles_list, tokenizer, model):
    """
    Convert SMILES strings into ChemBERTa embeddings using a transformer model.

    Args:
        smiles_list (list of str): A list of SMILES strings.
        tokenizer: Pre-trained ChemBERTa tokenizer.
        model: Pre-trained ChemBERTa model.

    Returns:
        np.ndarray: A NumPy array of ChemBERTa embeddings for each SMILES.
    """
    embeddings = []
    for smi in smiles_list:
        inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)


def get_lipidbalm_embeddings(proteins_list, lipids_list, protein_tokenizer, lipid_tokenizer, model, batch_size=128):
    """
    Compute LipidBALM embeddings for protein targets and lipids in batches.

    Args:
        proteins_list (list of str): List of protein sequences.
        lipids_list (list of str): List of lipid SMILES strings.
        protein_tokenizer: Tokenizer for the protein sequences.
        lipid_tokenizer: Tokenizer for the lipid SMILES.
        model: LipidBALM model used for generating embeddings.
        batch_size (int): The size of batches to process.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Protein embeddings, lipid embeddings, and cosine similarities.
    """
    # Prepare for batching
    protein_embeddings = []
    lipid_embeddings = []
    cosine_similarities = []
    
    # Ensure model is in evaluation mode
    model.eval()
    
    print("Computing embeddings...")
    # Create batches for more efficient processing
    for i in range(0, len(proteins_list), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{len(proteins_list) // batch_size + 1}...")
        # Get the batch
        batch_proteins = proteins_list[i:i + batch_size]
        batch_lipids = lipids_list[i:i + batch_size]
        
        # Tokenize the batch
        device = next(model.parameters()).device
        protein_inputs = protein_tokenizer(batch_proteins, return_tensors="pt", padding=True, truncation=True).to(device)
        lipid_inputs = lipid_tokenizer(batch_lipids, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Prepare inputs for the model
        inputs = {
            "protein_input_ids": protein_inputs["input_ids"],
            "protein_attention_mask": protein_inputs["attention_mask"],
            "lipid_input_ids": lipid_inputs["input_ids"],
            "lipid_attention_mask": lipid_inputs["attention_mask"],
        }
        
        # Run the model in batches
        with torch.no_grad():  # Disable gradient calculations for efficiency
            predictions = model(inputs)
        
        # Collect results
        protein_embeddings.extend([embedding.squeeze().detach().cpu().numpy() for embedding in predictions["protein_embedding"]])
        lipid_embeddings.extend([embedding.squeeze().detach().cpu().numpy() for embedding in predictions["lipid_embedding"]])
        if "cosine_similarity" in predictions:
            cosine_similarities.extend([cos_sim.squeeze().detach().cpu().numpy() for cos_sim in predictions["cosine_similarity"]])

    protein_embeddings = np.array(protein_embeddings)
    lipid_embeddings = np.array(lipid_embeddings)
    cosine_similarities = np.array(cosine_similarities) if "cosine_similarity" in predictions else None
    
    return protein_embeddings, lipid_embeddings, cosine_similarities


def split_data(X, y, smiles, proteins=None, test_size=0.2, seed=42):
    """
    Split data into training and testing sets, along with SMILES strings and proteins.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target values.
        smiles (list): SMILES strings for lipids.
        proteins (list, optional): Protein sequences.
        test_size (float): Proportion of the data to include in the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple: Training and testing feature matrices, target values, SMILES strings, and protein sequences.
    """
    if proteins is not None:
        X_train, X_test, y_train, y_test, smiles_train, smiles_test, proteins_train, proteins_test = train_test_split(
            X, y, smiles, proteins, test_size=test_size, random_state=seed
        )
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                smiles_train, smiles_test,
                proteins_train, proteins_test)
    else:
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
            X, y, smiles, test_size=test_size, random_state=seed
        )
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                smiles_train, smiles_test,
                None, None)


# GP Models
class ExactGPModelECFP8(gpytorch.models.ExactGP):
    """
    Gaussian Process model for ECFP8 fingerprints using the Tanimoto kernel.

    Args:
        train_x (torch.Tensor): Training feature data.
        train_y (torch.Tensor): Training target values.
        likelihood: Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelECFP8, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelChemBERTa(gpytorch.models.ExactGP):
    """
    Gaussian Process model for ChemBERTa embeddings using the RBF kernel.

    Args:
        train_x (torch.Tensor): Training feature data.
        train_y (torch.Tensor): Training target values.
        likelihood: Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelChemBERTa, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())

    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelLipidBALM(gpytorch.models.ExactGP):
    """
    Gaussian Process model for LipidBALM embeddings using the RBF kernel.

    Args:
        train_x (torch.Tensor): Training feature data.
        train_y (torch.Tensor): Training target values.
        likelihood: Gaussian likelihood for the GP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelLipidBALM, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())

    def forward(self, x):
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(
    X_train, y_train, X_test, y_test, smiles_test, proteins_test=None, model_class=None, learning_rate=0.1, epochs=50
):
    """
    Train a Gaussian Process model and log training progress to Weights & Biases.

    Args:
        X_train (torch.Tensor): Training feature matrix.
        y_train (torch.Tensor): Training target values.
        X_test (torch.Tensor): Test feature matrix.
        y_test (torch.Tensor): Test target values.
        smiles_test (list): List of SMILES strings for the test set.
        proteins_test (list, optional): List of protein sequences for the test set.
        model_class (class): The class of the GP model to train.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.

    Returns:
        Tuple: RMSE, R-squared, Spearman correlation, and Pearson correlation.
    """
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_class(X_train, y_train, likelihood)

    # Training mode
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize training logging for W&B
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        # Log the loss to W&B
        wandb.log({"epoch": i+1, "train/loss": loss.item()})
        if (i+1) % 5 == 0:
            print(f"Epoch {i+1}/{epochs} - Loss: {loss.item():.4f}")

    # Evaluation mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test)).mean

    preds = preds.detach().numpy()
    y_test = y_test.detach().numpy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    spearman_corr, _ = spearmanr(y_test, preds)
    pearson_corr, _ = pearsonr(y_test, preds)

    # Log evaluation metrics to W&B
    wandb.log({
        "test/rmse": rmse, 
        "test/r2": r2, 
        "test/spearman": spearman_corr, 
        "test/pearson": pearson_corr
    })

    # Save predictions and actual values to a CSV
    predictions_df = pd.DataFrame({
        "protein": proteins_test if proteins_test else [None] * len(y_test),
        "lipid": smiles_test,
        "label": y_test,
        "prediction": preds,
        "abs_error": np.abs(y_test - preds)
    })
    predictions_filename = "test_prediction.csv"
    predictions_df.to_csv(predictions_filename, index=False)

    # Create W&B artifact for the predictions
    artifact = wandb.Artifact("test_prediction", type="prediction")
    artifact.add_file(predictions_filename)
    wandb.log_artifact(artifact)

    print(f"Results: RMSE={rmse:.4f}, R2={r2:.4f}, Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")
    return rmse, r2, spearman_corr, pearson_corr


def main():
    """
    Main function to load the dataset, train the model, and log results.
    """
    args = argument_parser()

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    data = pd.read_csv(args.dataset)
    
    # Check and rename columns if necessary
    if "Target" in data.columns and "ProteinSequence" not in data.columns:
        data = data.rename(columns={"Target": "ProteinSequence"})
    if "Drug" in data.columns and "LipidSMILES" not in data.columns:
        data = data.rename(columns={"Drug": "LipidSMILES"})
    if "Y" in data.columns and "BindingAffinityValue" not in data.columns:
        data = data.rename(columns={"Y": "BindingAffinityValue"})
    
    # Ensure required columns exist
    required_columns = ["ProteinSequence", "LipidSMILES", "BindingAffinityValue"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")

    # Initialize Weights & Biases logging
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    run_name = f"gp__{args.embedding_type}__test_{args.test_size}__lr_{args.lr}__data_{os.path.basename(args.dataset)}__seed_{args.random_seed}"
    
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        config=vars(args),
    )

    # Extract protein sequences, SMILES, and binding affinity values
    proteins = data["ProteinSequence"].tolist()
    smiles = data["LipidSMILES"].tolist()
    y = data["BindingAffinityValue"].to_numpy()

    if args.debug:
        print("Running in debug mode with reduced dataset")
        proteins = proteins[:100]
        smiles = smiles[:100]
        y = y[:100]

    if args.embedding_type == "ECFP8":
        # Convert SMILES to ECFP8 fingerprints
        print("Generating ECFP8 fingerprints...")
        X = smiles_to_ecfp8_fingerprint(smiles)

        # Split data (proteins not needed for this embedding)
        X_train, X_test, y_train, y_test, smiles_train, smiles_test, _, _ = split_data(
            X, y, smiles, None, test_size=args.test_size, seed=args.random_seed
        )

        model_class = ExactGPModelECFP8
        
        # Train and evaluate ECFP8-based GP
        rmse, r2, spearman, pearson = train_gp_model(
            X_train,
            y_train,
            X_test,
            y_test,
            smiles_test,
            None,
            ExactGPModelECFP8,
            args.lr,
            args.epochs,
        )
        
    elif args.embedding_type == "ChemBERTa":
        # Load ChemBERTa tokenizer and model
        print("Loading ChemBERTa model...")
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

        # Convert SMILES into ChemBERTa embeddings
        print("Generating ChemBERTa embeddings...")
        X = smiles_to_chemberta_embedding(smiles, tokenizer, model)

        # Split data (proteins not needed for this embedding)
        X_train, X_test, y_train, y_test, smiles_train, smiles_test, _, _ = split_data(
            X, y, smiles, None, test_size=args.test_size, seed=args.random_seed
        )

        model_class = ExactGPModelChemBERTa
        
        # Train and evaluate ChemBERTa-based GP
        rmse, r2, spearman, pearson = train_gp_model(
            X_train,
            y_train,
            X_test,
            y_test,
            smiles_test,
            None,
            model_class,
            args.lr,
            args.epochs,
        )
        
    elif args.embedding_type.startswith("LipidBALM"):
        # Load configuration for LipidBALM
        config_filepath = "configs/lipidbalm_peft.yaml"  # Adjust path as needed
        configs = Configs(**common_utils.load_yaml(config_filepath))

        # Load the LipidBALM model
        print("Loading LipidBALM model...")
        model = LipidBALM(configs.model_configs)
        model = load_trained_model(model, configs.model_configs, is_training=False)
        
        # Detect and use CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the tokenizers
        protein_tokenizer = AutoTokenizer.from_pretrained(
            configs.model_configs.protein_model_name_or_path
        )
        lipid_tokenizer = AutoTokenizer.from_pretrained(
            configs.model_configs.lipid_model_name_or_path
        )

        # Generate embeddings
        print(f"Generating embeddings using {args.embedding_type}...")
        protein_embeddings, lipid_embeddings, cosine_similarities = get_lipidbalm_embeddings(
            proteins, smiles, protein_tokenizer, lipid_tokenizer, model
        )
        
        # Choose embedding type
        if args.embedding_type == "LipidBALM-lipid":
            X = lipid_embeddings
            print(f"Using lipid embeddings with shape {X.shape}")
        elif args.embedding_type == "LipidBALM-concat":
            X = np.concatenate((protein_embeddings, lipid_embeddings), axis=1)
            print(f"Using concatenated embeddings with shape {X.shape}")
        elif args.embedding_type == "LipidBALM-sum":
            X = protein_embeddings + lipid_embeddings
            print(f"Using summed embeddings with shape {X.shape}")
        elif args.embedding_type == "LipidBALM-subtract":
            X = protein_embeddings - lipid_embeddings
            print(f"Using difference embeddings with shape {X.shape}")
        elif args.embedding_type == "LipidBALM-cosine":
            X = cosine_similarities.reshape(-1, 1)  # Reshape to 2D array
            print(f"Using cosine similarity with shape {X.shape}")

        # Split data including protein sequences
        X_train, X_test, y_train, y_test, smiles_train, smiles_test, proteins_train, proteins_test = split_data(
            X, y, smiles, proteins, test_size=args.test_size, seed=args.random_seed
        )

        model_class = ExactGPModelLipidBALM

        # Train and evaluate the GP model
        rmse, r2, spearman, pearson = train_gp_model(
            X_train,
            y_train,
            X_test,
            y_test,
            smiles_test,
            proteins_test,
            model_class,
            args.lr,
            args.epochs,
        )

    # Print final results
    print(
        f"{args.embedding_type} GP Model - RMSE: {rmse:.4f}, "
        f"R^2: {r2:.4f}, "
        f"Spearman: {spearman:.4f}, "
        f"Pearson: {pearson:.4f}"
    )

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()