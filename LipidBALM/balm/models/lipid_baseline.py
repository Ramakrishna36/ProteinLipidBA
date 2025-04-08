import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from balm.configs import ModelConfigs
from balm.models.base_model import BaseModel


class LipidBaselineModel(BaseModel):
    """
    LipidBaselineModel model extends BaseModel to concatenate protein and lipid encodings.
    This model takes the embeddings from both the protein and lipid models, concatenates them, and processes them further.

    Attributes:
        model_configs (ModelConfigs): The configuration object for the model.
        protein_model (AutoModel): The pre-trained protein model.
        lipid_model (AutoModel): The pre-trained lipid model.
        protein_embedding_size (int): The size of the protein model embeddings.
        lipid_embedding_size (int): The size of the lipid model embeddings.
    """

    def __init__(
        self,
        model_configs: ModelConfigs,
        protein_embedding_size=640,
        lipid_embedding_size=384,
    ):
        super(LipidBaselineModel, self).__init__(
            model_configs, protein_embedding_size, lipid_embedding_size
        )

        # concatenating layers
        self.linear_projection = nn.Linear(
            self.protein_embedding_size + self.lipid_embedding_size,
            model_configs.model_hyperparameters.projected_size,
        )
        self.dropout = nn.Dropout(model_configs.model_hyperparameters.projected_dropout)
        self.out = nn.Linear(model_configs.model_hyperparameters.projected_size, 1)

        self.print_trainable_params()

        self.loss_fn = nn.MSELoss()

    def forward(self, batch_input, **kwargs):
        """
        Forward pass for the LipidBaselineModel.

        This method takes the input for both protein and lipid models, obtains their embeddings, concatenates them, and processes them further.

        Args:
            protein_input (torch.Tensor): The input tensor for the protein model.
            lipid_input (torch.Tensor): The input tensor for the lipid model.

        Returns:
            torch.Tensor: The output tensor after processing the concatenated embeddings.
        """
        forward_output = {}

        protein_embedding = self.protein_model(
            input_ids=batch_input["protein_input_ids"],
            attention_mask=batch_input["protein_attention_mask"],
        )["pooler_output"]

        lipid_embedding = self.lipid_model(
            input_ids=batch_input["lipid_input_ids"],
            attention_mask=batch_input["lipid_attention_mask"],
        )["pooler_output"]

        # concat
        concatenated_embedding = torch.cat((protein_embedding, lipid_embedding), 1)

        # add some dense layers
        projected_embedding = F.relu(self.linear_projection(concatenated_embedding))
        projected_embedding = self.dropout(projected_embedding)
        logits = self.out(projected_embedding)

        if batch_input["labels"] is not None:
            forward_output["loss"] = self.loss_fn(logits, batch_input["labels"])

        forward_output["protein_embedding"] = protein_embedding
        forward_output["lipid_embedding"] = lipid_embedding
        forward_output["logits"] = logits.squeeze(-1)

        return forward_output