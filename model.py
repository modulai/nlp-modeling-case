import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, output_dim=1, pad_index=1):
        """
        """
        
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        # Fully connected layer
        self.fc = nn.Linear(embedding_dim, output_dim)
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids):
        # embed input
        embedded = self.embedding(ids)
        # pooling layer
        pooled = embedded.mean(dim=1)
        # prediction using a fully connected layer followed by a sigmoid
        prediction = self.sigmoid(self.fc(pooled))

        return prediction
    


def set_seed(seed):
    """
    Set seed to allow experiment replication

    Input:
        - seed (int): seed to use
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_model(model_name, vocab_size, embedding_dim, pad_index, seed=1234, exp_id=-1):
    """
    Setup the model

    Inputs:
        model_name (str): name of model to use
        vocab_size (int): vocabulary size
        embedding_dim (int): dimension of embedding layer
        pad_index (int): index of "padding" token 
    """

    # Set seed
    set_seed(seed)
    
    # Load model
    if model_name == 'mlp':
        model = MLP(vocab_size, embedding_dim, pad_index)
    else:
        raise ValueError("Model is not implemented.")

    # Using an Adam optimizer
    optimizer = optim.Adam(model.parameters())
    # Since we have two classes: positive and non-positive sentiment we use a binary cross entropy loss
    criterion = nn.BCELoss()
    # Allow GPU to be used if existing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = criterion.to(device)

    if exp_id != -1:
        model.load_state_dict(torch.load("experiments/{}/weights.pt".format(exp_id)))

    return model, criterion, optimizer, device
