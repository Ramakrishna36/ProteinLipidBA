from Dataset import ProteinLipidInteractionDataset


DATASET_MAPPING = {
    "ProteinLipid": ProteinLipidInteractionDataset,
}


def get_dataset(dataset_name, csv_path=None, *args, **kwargs):
    """
    Factory function to get dataset instances.
    
    Args:
        dataset_name (str): Name of the dataset to load
        csv_path (str): Path to the CSV file for ProteinLipid dataset
        *args, **kwargs: Additional arguments to pass to dataset constructor
        
    Returns:
        Dataset instance
    """
    if dataset_name == "ProteinLipid":
        if csv_path is None:
            raise ValueError("csv_path must be provided for ProteinLipid dataset")
        return ProteinLipidInteractionDataset(csv_path=csv_path, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")