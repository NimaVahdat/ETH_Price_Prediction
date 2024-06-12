import collections
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Data import ETHData, get_data_loader
from Networks.networks import GRUModel, LSTMModel


class ETHPricePredictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_data()
        self._setup_model()
        self._initialize_weights()
        self._setup_training_tools()
        self.writer = SummaryWriter()

    def _setup_model(self):
        """Set up the model based on the configuration."""
        model_name = self.config["model"]
        model_config = self.config["model_config"]

        # Initialize the models based on the specified name
        model_cls = (
            LSTMModel
            if model_name == "LSTM"
            else GRUModel if model_name == "GRU" else None
        )
        if model_cls is None:
            raise ValueError("Model should be either 'LSTM' or 'GRU'!")

        self.model_open = model_cls(**model_config).to(self.device)
        self.model_high = model_cls(**model_config).to(self.device)
        self.model_low = model_cls(**model_config).to(self.device)
        self.model_close = model_cls(**model_config).to(self.device)

    def _setup_data(self):
        """Set up data loaders based on the configuration."""
        data_config = self.config["data_config"]
        batch_size = self.config["batch_size"]
        data = ETHData(**data_config)
        self.train_data, self.test_data = data.get_data()

        self.train_loader = get_data_loader(self.train_data, batch_size, shuffle=True)
        self.valid_loader = get_data_loader(self.test_data, batch_size)

    def _initialize_weights(self):
        """Initialize model weights if specified in the configuration."""
        if self.config.get("initialize_weights", False):
            for model in [
                self.model_open,
                self.model_high,
                self.model_low,
                self.model_close,
            ]:
                model.apply(self._init_weights)

    def _setup_training_tools(self):
        """Set up optimizer and loss criterion for training."""
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizers = {
            "open": optim.Adam(self.model_open.parameters(), lr=self.config["lr"]),
            "high": optim.Adam(self.model_high.parameters(), lr=self.config["lr"]),
            "low": optim.Adam(self.model_low.parameters(), lr=self.config["lr"]),
            "close": optim.Adam(self.model_close.parameters(), lr=self.config["lr"]),
        }

    def _init_weights(self, m):
        """Initialize weights of the model."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    def train_epoch(self, dataloader, epoch):
        """Train the model for one epoch."""
        self.model_open.train()
        self.model_high.train()
        self.model_low.train()
        self.model_close.train()
        epoch_losses = [[], [], [], []]
        for X, y in tqdm.tqdm(dataloader, desc=f"Training epoch {epoch+1}"):
            X, y = X.to(self.device), y.to(self.device)
            open_X, high_X, low_X, close_X = (
                X[:, :, 0],
                X[:, :, 1],
                X[:, :, 2],
                X[:, :, 3],
            )
            open_y, high_y, low_y, close_y = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

            predictions_open = self.model_open(open_X)
            predictions_high = self.model_high(high_X)
            predictions_low = self.model_low(low_X)
            predictions_close = self.model_close(close_X)

            loss_open = self.criterion(predictions_open, open_y)
            loss_high = self.criterion(predictions_high, high_y)
            loss_low = self.criterion(predictions_low, low_y)
            loss_close = self.criterion(predictions_close, close_y)

            self.optimizers["open"].zero_grad()
            self.optimizers["high"].zero_grad()
            self.optimizers["low"].zero_grad()
            self.optimizers["close"].zero_grad()
            loss_open.backward()
            loss_high.backward()
            loss_low.backward()
            loss_close.backward()
            self.optimizers["open"].step()
            self.optimizers["high"].step()
            self.optimizers["low"].step()
            self.optimizers["close"].step()

            epoch_losses[0].append(loss_open.item())
            epoch_losses[1].append(loss_high.item())
            epoch_losses[2].append(loss_low.item())
            epoch_losses[3].append(loss_close.item())

        return torch.mean(torch.tensor(epoch_losses).T, 1)

    def evaluate_epoch(self, dataloader):
        """Evaluate the model for one epoch."""
        self.model_open.eval()
        self.model_high.eval()
        self.model_low.eval()
        self.model_close.eval()
        epoch_losses = [[], [], [], []]
        all_open_y, all_open_pred = [], []
        all_high_y, all_high_pred = [], []
        all_low_y, all_low_pred = [], []
        all_close_y, all_close_pred = [], []
        with torch.no_grad():
            for X, y in tqdm.tqdm(dataloader, desc="Evaluating "):
                X, y = X.to(self.device), y.to(self.device)
                open_X, high_X, low_X, close_X = (
                    X[:, :, 0],
                    X[:, :, 1],
                    X[:, :, 2],
                    X[:, :, 3],
                )
                open_y, high_y, low_y, close_y = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

                predictions_open = self.model_open(open_X)
                predictions_high = self.model_high(high_X)
                predictions_low = self.model_low(low_X)
                predictions_close = self.model_close(close_X)

                loss_open = self.criterion(predictions_open, open_y)
                loss_high = self.criterion(predictions_high, high_y)
                loss_low = self.criterion(predictions_low, low_y)
                loss_close = self.criterion(predictions_close, close_y)

                epoch_losses[0].append(loss_open.item())
                epoch_losses[1].append(loss_high.item())
                epoch_losses[2].append(loss_low.item())
                epoch_losses[3].append(loss_close.item())

                all_open_y.extend(open_y.cpu().numpy())
                all_open_pred.extend(predictions_open.cpu().numpy())
                all_high_y.extend(high_y.cpu().numpy())
                all_high_pred.extend(predictions_high.cpu().numpy())
                all_low_y.extend(low_y.cpu().numpy())
                all_low_pred.extend(predictions_low.cpu().numpy())
                all_close_y.extend(close_y.cpu().numpy())
                all_close_pred.extend(predictions_close.cpu().numpy())

        metrics = {
            "open": self._calculate_metrics(all_open_y, all_open_pred),
            "high": self._calculate_metrics(all_high_y, all_high_pred),
            "low": self._calculate_metrics(all_low_y, all_low_pred),
            "close": self._calculate_metrics(all_close_y, all_close_pred),
        }

        return torch.mean(torch.tensor(epoch_losses).T, 1), metrics

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    def train_and_evaluate(self):
        """Train and evaluate the model for a specified number of epochs."""
        num_epochs = self.config["num_epochs"]
        best_valid_losses = [float("inf")] * 4
        metrics = collections.defaultdict(list)
        model_name = self.config["model"]
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(self.train_loader, epoch)
            valid_loss, valid_metrics = self.evaluate_epoch(self.valid_loader)
            metrics["train_losses_open"].append(train_loss[0])
            metrics["valid_losses_open"].append(valid_loss[0])
            metrics["train_losses_high"].append(train_loss[1])
            metrics["valid_losses_high"].append(valid_loss[1])
            metrics["train_losses_low"].append(train_loss[2])
            metrics["valid_losses_low"].append(valid_loss[2])
            metrics["train_losses_close"].append(train_loss[3])
            metrics["valid_losses_close"].append(valid_loss[3])

            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/Train/Open", train_loss[0], epoch)
            self.writer.add_scalar("Loss/Valid/Open", valid_loss[0], epoch)
            self.writer.add_scalar("Loss/Train/High", train_loss[1], epoch)
            self.writer.add_scalar("Loss/Valid/High", valid_loss[1], epoch)
            self.writer.add_scalar("Loss/Train/Low", train_loss[2], epoch)
            self.writer.add_scalar("Loss/Valid/Low", valid_loss[2], epoch)
            self.writer.add_scalar("Loss/Train/Close", train_loss[3], epoch)
            self.writer.add_scalar("Loss/Valid/Close", valid_loss[3], epoch)

            for key in ["open", "high", "low", "close"]:
                for metric, value in valid_metrics[key].items():
                    self.writer.add_scalar(
                        f"Metrics/Valid/{key.capitalize()}/{metric}", value, epoch
                    )

            # Save best models
            for i, (train, valid, model) in enumerate(
                zip(
                    train_loss,
                    valid_loss,
                    [
                        self.model_open,
                        self.model_high,
                        self.model_low,
                        self.model_close,
                    ],
                )
            ):
                if valid < best_valid_losses[i]:
                    best_valid_losses[i] = valid
                    torch.save(
                        model.state_dict(),
                        f"{model_name}_{['Open', 'High', 'Low', 'Close'][i]}.pt",
                    )

            print(f"Epoch: {epoch + 1}")
            print(f"Train Loss (Open): {train_loss[0]:.3f}")
            print(
                f"Valid Loss (Open): {valid_loss[0]:.3f} - MAE: {valid_metrics['open']['MAE']:.3f}, MSE: {valid_metrics['open']['MSE']:.3f}, RMSE: {valid_metrics['open']['RMSE']:.3f}, R2: {valid_metrics['open']['R2']:.3f}"
            )
            print(f"Train Loss (High): {train_loss[1]:.3f}")
            print(
                f"Valid Loss (High): {valid_loss[1]:.3f} - MAE: {valid_metrics['high']['MAE']:.3f}, MSE: {valid_metrics['high']['MSE']:.3f}, RMSE: {valid_metrics['high']['RMSE']:.3f}, R2: {valid_metrics['high']['R2']:.3f}"
            )
            print(f"Train Loss (Low): {train_loss[2]:.3f}")
            print(
                f"Valid Loss (Low): {valid_loss[2]:.3f} - MAE: {valid_metrics['low']['MAE']:.3f}, MSE: {valid_metrics['low']['MSE']:.3f}, RMSE: {valid_metrics['low']['RMSE']:.3f}, R2: {valid_metrics['low']['R2']:.3f}"
            )
            print(f"Train Loss (Close): {train_loss[3]:.3f}")
            print(
                f"Valid Loss (Close): {valid_loss[3]:.3f} - MAE: {valid_metrics['close']['MAE']:.3f}, MSE: {valid_metrics['close']['MSE']:.3f}, RMSE: {valid_metrics['close']['RMSE']:.3f}, R2: {valid_metrics['close']['R2']:.3f}"
            )

    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        total_params = 0
        for model in [
            self.model_open,
            self.model_high,
            self.model_low,
            self.model_close,
        ]:
            total_params += sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        return total_params
