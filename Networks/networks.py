import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_rate):
        """
        LSTM Model Initialization.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output units.
            n_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout rate.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim).
        """
        # Pass through LSTM layer
        output, (hidden, cell) = self.lstm(x)

        # Apply dropout to the last hidden state
        hidden = self.dropout(hidden[-1])

        # Pass through fully connected layer
        prediction = self.fc(hidden)

        return prediction


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_rate):
        """
        GRU Model Initialization.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output units.
            n_layers (int): Number of GRU layers.
            dropout_rate (float): Dropout rate.
        """
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim).
        """
        # Pass through GRU layer
        output, hidden = self.gru(x)

        # Apply dropout to the last hidden state
        hidden = self.dropout(hidden[-1])

        # Pass through fully connected layer
        prediction = self.fc(hidden)

        return prediction
