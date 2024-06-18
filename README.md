# <img src="https://github.com/NimaVahdat/ETH_Price_Prediction/blob/main/ETH.png" alt="Logo" style="width: 50px; height: auto;"> ETH Price Prediction

This repository contains a simple and basic project for predicting Ethereum (ETH) prices using time series analysis and machine learning models. The project includes data preprocessing, feature engineering, and model training using LSTM and GRU neural networks. The models are evaluated using various regression metrics to assess their performance.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Evaluation](#Model-Evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Visualization**: Plot historical ETH price data.
- **Model Training**: Train a machine learning model to predict future ETH prices.
- **Model Evaluation**: Evaluate the model performance.
- **Parameter Counting**: Display the number of trainable parameters in the model.

## Installation

#### Clone the repository:

```bash
git clone https://github.com/NimaVahdat/ETH_Price_Prediction.git
cd ETH_Price_Prediction
```


## Usage

1. Prepare your configuration file `config.yml` with the necessary parameters (refer to the [Configuration](#configuration) section). Look at the example LSTM_config.yaml.

2. Run the main script:
```sh
python main.py --config config.yml
```

## Configuration

1. Create a `config.yml` file with the following structure:

```yaml
# Configuration file

# Model configuration
model_config:
  input_dim: 16                 # Number of features (time steps)
  hidden_dim: 10                # Increased hidden dimension for more learning capacity
  output_dim: 1                 # Predicting one value (e.g., closing price)
  n_layers: 2                   # Increased number of layers for more abstract representations
  dropout_rate: 0               # Introduce dropout to prevent overfitting

# Data configuration
data_config:
  test_size: 0.25               # 25% of data for testing
  data_dir: ./Data/ETH-USD.csv  # Path to the data file

# Training configuration
batch_size: 32                  # Typical batch size for stable training
time_step: 16                   # Number of time steps to look back for prediction
lr: 0.00001                     # Adjusted learning rate for faster convergence
num_epochs: 1000                # Increased number of epochs for thorough training
initialize_weights: True        # Initialize weights at the start

# Model to use (LSTM, GRU)
model: LSTM                     # Type of model to use
start: 2023-06-09               # Start date for training
end: 2024-06-09                 # End date for training
```
2. Script Breakdown:

 * main.py: The entry point for training and evaluating the ETH price prediction model.
 * eth_price_predictor.py: Contains the ETHPricePredictor class, which handles data loading, model setup, training, and evaluation.
 * utils.py: Contains utility functions, including load_config_file to read the configuration file.

## Model Evaluation
The model performance is evaluated using the following metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R-squared (R²)

## Future Work
This project provides a simple and basic ETH price prediction model. In the future, we plan to enhance the model by incorporating sentiment analysis on news data and tweets to improve its predictive capabilities and better handle the complexities of time series data.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
