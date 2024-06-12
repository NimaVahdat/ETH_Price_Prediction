import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class ETHData:
    def __init__(self, test_size: float = 0.25, data_dir: str = './Data/ETH-USD.csv') -> None:
        """
        Initialize ETHData object.

        Parameters:
            test_size (float): The proportion of data to include in the test split.
            data_dir (str): Directory of the CSV file containing the data.
        """
        self.test_size = test_size
        self.data_dir = data_dir
        self.data = None
        self.load_data()  # Load data automatically when object is created

    def load_data(self) -> None:
        """
        Load data from CSV file.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"File '{self.data_dir}' not found.")
        self.data = pd.read_csv(self.data_dir)

    def get_data_info(self) -> None:
        """
        Display information and summary statistics of the loaded data.
        """
        print(self.data.info())
        print(self.data.describe())

    def plot_stock(self, start='2020-01-11', end='2024-06-09') -> None:
        """
        Plot stock prices within the specified date range.

        Parameters:
            start (str): Start date of the plot.
            end (str): End date of the plot.
        """
        data = self.data.loc[(self.data['Date'] >= start) & (self.data['Date'] <= end)]
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Open'], marker='o', linestyle='-', color='blue', label='Open')
        plt.plot(data['Date'], data['High'], marker='o', linestyle='-', color='green', label='High')
        plt.plot(data['Date'], data['Low'], marker='o', linestyle='-', color='red', label='Low')
        plt.plot(data['Date'], data['Close'], marker='o', linestyle='-', color='purple', label='Close')
        plt.title('Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_data(self, start='2020-01-11', end='2024-06-09') -> tuple:
        """
        Get preprocessed and scaled data for modeling.

        Parameters:
            start (str): Start date of the data.
            end (str): End date of the data.

        Returns:
            tuple: Tuple containing scaled training and test data (X_train_scaled, X_test_scaled,
            y_train_scaled, y_test_scaled).
        """
        data = self.data.loc[(self.data['Date'] >= start) & (self.data['Date'] <= end)]
        X = data.drop(columns=['Date', 'Close'])
        y = data['Close']
        split_index = int(len(data) * (1 - self.test_size))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # Initialize the MinMaxScaler for X and y
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Fit and transform the training data
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

        # Save the scalers to disk
        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open('scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
