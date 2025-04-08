import os
import sys
import importlib
import argparse


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from balm import common_utils
from balm.configs import Configs
from balm.trainer import LipidTrainer

def argument_parser():
    parser = argparse.ArgumentParser(description="LipidBALM Training")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()



# Debugging imports
try:
    import balm
    print("balm package imported successfully")
    
    # Try to import specific modules
    import balm.dataset
    print("balm.dataset imported successfully")
    
    from balm import common_utils
    print("common_utils imported successfully")
    
    from balm.configs import Configs
    print("Configs imported successfully")
    
    from balm.trainer import LipidTrainer
    print("LipidTrainer imported successfully")

except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Python path:", sys.path)
    
    # Additional debugging - list contents of balm directory
    import os
    print("\nContents of balm directory:")
    print(os.listdir(os.path.join(project_root, 'balm')))
    
    # Try to manually import
    print("\nAttempting manual import:")
    spec = importlib.util.spec_from_file_location(
        "balm.dataset_filtered", 
        os.path.join(project_root, 'balm', 'dataset', 'dataset_filtered.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    # Set the random seed for reproducibility
    common_utils.setup_random_seed(configs.training_configs.random_seed)
    
    # Setup the output directory for the experiment
    outputs_dir, checkpoint_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    
    # Save the training configuration to the output directory
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to Weights & Biases (WandB)
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    # Initialize the Trainer and start training
    trainer = LipidTrainer(configs, wandb_entity, wandb_project, outputs_dir)
    
    # Set up the protein-lipid dataset
    dataset_splits = trainer.set_dataset()
    
    # Setup the training environment
    trainer.setup_training()
    
    # Start the training loop
    trainer.train()

if __name__ == "__main__":
    main()