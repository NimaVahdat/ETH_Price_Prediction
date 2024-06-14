import torch
import pandas as pd
import numpy as np


def create_collate_fn(time_step):
    """
    Create a collate function for DataLoader.

    Parameters:
        time_step (int): Number of time steps to consider for each sequence.

    Returns:
        collate_fn: Collate function for DataLoader.
    """

    def collate_fn(batch):
        X, y = [], []
        for i in range(len(batch) - time_step):
            seq = batch[i : (i + time_step)]  # Extract sequence of length 'time_step'
            X.append(seq)
            y.append(batch[i + time_step])  # Target is the value after the sequence
        return torch.tensor(np.array(X)), torch.tensor(np.array(y))

    return collate_fn


def get_data_loader(dataset, batch_size, time_step=15, shuffle=False):
    """
    Create a DataLoader for time series dataset.

    Parameters:
        dataset (numpy.ndarray or torch.Tensor): Time series dataset.
        batch_size (int): Number of samples per batch.
        time_step (int): Number of time steps to consider for each sequence.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        torch.utils.data.DataLoader: DataLoader object.
    """
    # Convert dataset to tensor if it's a numpy array
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_numpy()

    # Create collate function
    collate_fn = create_collate_fn(time_step)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )

    return data_loader
