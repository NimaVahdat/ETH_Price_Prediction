import yaml
import torch
import argparse
from eth_price_prediction import ETHPricePredictor
from utils import load_config_file


def main(config_path):
    """Main function to train and evaluate the ETH price prediction model."""
    # Load the configuration
    config = load_config_file(config_path)

    # Initialize the ETHPricePredictor with the configuration
    predictor = ETHPricePredictor(config)

    # Plot Stock
    predictor.plot_stock(start="2020-01-11", end="2024-06-09")

    # Print the number of parameters
    print(f"Number of trainable parameters: {predictor.count_parameters()}")

    # Train and evaluate the model
    predictor.train_and_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETH Price Prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    main(args.config)
