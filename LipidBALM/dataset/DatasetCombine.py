import pandas as pd

# Read CSV files with full paths
ec50_df = pd.read_csv(r"C:\Users\91967\Documents\Protein-Lipid interactions\ProteinLipidBA\LipidBALM\Dataset\EC50.csv")
ic50_df = pd.read_csv(r"C:\Users\91967\Documents\Protein-Lipid interactions\ProteinLipidBA\LipidBALM\Dataset\IC50.csv")
kd_df = pd.read_csv(r"C:\Users\91967\Documents\Protein-Lipid interactions\ProteinLipidBA\LipidBALM\Dataset\Kd.csv")
ki_df = pd.read_csv(r"C:\Users\91967\Documents\Protein-Lipid interactions\ProteinLipidBA\LipidBALM\Dataset\Ki.csv")

# Add affinity type column based on file name
ec50_df['AffinityType'] = 'EC50'
ic50_df['AffinityType'] = 'IC50'
kd_df['AffinityType'] = 'Kd'
ki_df['AffinityType'] = 'Ki'

# Rename columns for consistency
for df in [ec50_df, ic50_df, kd_df, ki_df]:
    df.rename(columns={
        'Drug': 'LipidSMILES', 
        'Target': 'ProteinSequence', 
        'Y': 'BindingAffinityValue'
    }, inplace=True)

# Combine dataframes
combined_df = pd.concat([ec50_df, ic50_df, kd_df, ki_df], ignore_index=True)

# Reorder columns
column_order = [
    'LipidSMILES', 
    'ProteinSequence', 
    'BindingAffinityValue', 
    'AffinityType'
]
combined_df = combined_df[column_order]

# Save to CSV in the same directory
output_path = r"C:\Users\91967\Documents\Protein-Lipid interactions\ProteinLipidBA\LipidBALM\Dataset\combined_binding_data.csv"
combined_df.to_csv(output_path, index=False)

print(f"CSV file created with {len(combined_df)} rows.")
print("Columns:", list(combined_df.columns))
print(f"Saved to: {output_path}")
print("\nFirst few rows:")
print(combined_df.head())

# Additional data analysis
print("\nSummary by Affinity Type:")
print(combined_df.groupby('AffinityType').agg({
    'BindingAffinityValue': ['count', 'mean', 'min', 'max']
}))