from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import os
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from interface import implements, Interface
import xgboost as xgb


class TradingStrategy(
    Interface
):  ### Our Trading Strategy interface, which all of the other concrete strategies will implement.
    @abstractmethod
    def execute(self, data: pd.DataFrame):
        pass


class MomentumStrategy(
    implements(TradingStrategy)
):  # Concrete momentum strategy class that implements the TradingStrategy interface
    def __init__(self, look_back_period=3, threshold=0.01):
        self.look_back_period = look_back_period
        self.threshold = threshold

    def execute(self, data: pd.DataFrame):
        if len(data) < self.look_back_period + 1:
            return "Hold"

        return_pct = self.calculate_return(data)

        if return_pct > self.threshold:
            return "Buy"
        elif return_pct < -self.threshold:
            return "Sell"
        else:
            return "Hold"

    def calculate_return(self, data: pd.DataFrame):
        close_prices = data["Close"]
        start_price = close_prices.iloc[-(self.look_back_period + 1)]
        end_price = close_prices.iloc[-1]
        return_pct = (end_price - start_price) / start_price
        return return_pct


class MovingAveragesStrategy(
    implements(TradingStrategy)
):  # Concrete moving averages strategy class that implements the TradingStrategy interface
    def __init__(self, window_size):
        self.window_size = window_size

    def execute(self, data: pd.DataFrame):
        if len(data) < self.window_size:
            return "Hold"

        moving_average = self.calculate_moving_average(data)
        latest_price = data["Close"].iloc[-1]

        if latest_price > moving_average:
            return "Buy"
        else:
            return "Sell"

    def calculate_moving_average(self, data: pd.DataFrame):
        window = data["Close"].tail(self.window_size)
        window_average = window.mean()
        return window_average


class AnalyzePatternStrategy(
    implements(TradingStrategy)
):  # Concrete pattern analysis strategy class that implements the TradingStrategy interface
    def execute(self, data: pd.DataFrame):
        if len(data) < 3:
            return "Hold"

        pattern = self.identify_pattern(data.tail(3))
        if pattern == "Bullish Engulfing":
            return "Buy"
        elif pattern == "Hammer":
            return "Buy"
        elif pattern == "Morning Star":
            return "Buy"
        elif pattern == "Bearish Engulfing":
            return "Sell"
        elif pattern == "Inverse Hammer":
            return "Sell"
        else:
            return "Hold"

    def identify_pattern(self, candles: pd.DataFrame):
        # Convert string data to floats
        candles = candles.astype(float)

        def is_bullish(candle):
            return candle["Close"] > candle["Open"]

        def is_bearish(candle):
            return candle["Close"] < candle["Open"]

        def body_size(candle):
            return abs(candle["Close"] - candle["Open"])

        def upper_shadow(candle):
            return candle["High"] - max(candle["Open"], candle["Close"])

        def lower_shadow(candle):
            return min(candle["Open"], candle["Close"]) - candle["Low"]

        recent_candles = candles.tail(10) if len(candles) >= 10 else candles
        avg_body_size = recent_candles.apply(body_size, axis=1).mean()
        c1, c2, c3 = (
            candles.iloc[-3],
            candles.iloc[-2],
            candles.iloc[-1],
        )

        if (
            is_bearish(c1)
            and is_bullish(c2)
            and body_size(c2) > body_size(c1)
            and c2["Open"] < c1["Close"]
            and c2["Close"] > c1["Open"]
        ):
            return "Bullish Engulfing"

        if (
            is_bullish(c1)
            and is_bearish(c2)
            and body_size(c2) > body_size(c1)
            and c2["Open"] > c1["Close"]
            and c2["Close"] < c1["Open"]
        ):
            return "Bearish Engulfing"

        if lower_shadow(c2) > 2 * body_size(c2) and upper_shadow(c2) < body_size(c2):
            return "Hammer"

        if upper_shadow(c2) > 2 * body_size(c2) and lower_shadow(c2) < body_size(c2):
            return "Inverse Hammer"

        if (
            is_bearish(c1)
            and body_size(c2) < avg_body_size * 0.5
            and is_bullish(c3)
            and c3["Close"] > c1["Open"]
        ):
            return "Morning Star"

        return None


class RSIStrategy(
    implements(TradingStrategy)
):  # Concrete RSI strategy class that implements the TradingStrategy interface
    def __init__(self, window_size=14, overbought=70, oversold=30):
        self.window_size = window_size
        self.overbought = overbought
        self.oversold = oversold

    def execute(self, data: pd.DataFrame):
        if len(data) < self.window_size:
            return "Hold"

        rsi = self.calculate_rsi(data)
        latest_rsi = rsi.iloc[-1]

        if latest_rsi > self.overbought:
            return "Sell"
        elif latest_rsi < self.oversold:
            return "Buy"
        else:
            return "Hold"

    def calculate_rsi(self, data: pd.DataFrame):
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window_size).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window_size).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class XGBStrategy(
    implements(TradingStrategy)
):  # Concrete XGB strategy class that implements the TradingStrategy interface
    def __init__(
        self,
        seq_length=10,
        model_dir="XGBModel",
        model_filename="xgb_model.pkl",
        scaler_filename="scaler.pkl",
    ):
        self.seq_length = seq_length
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, model_filename)
        self.scaler_path = os.path.join(self.model_dir, scaler_filename)
        self.model = None
        self.scaler = None

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.load_model()
        else:
            self.train_model()

    def label_data(
        self, data
    ):  # We shift the data and label it as buy, sell or hold based on the future price --> If we make a profit
        data["Future Price"] = data["Close"].shift(-1)
        conditions = [
            (data["Future Price"] > data["Close"]),
            (data["Future Price"] < data["Close"]),
        ]
        choices = ["Buy", "Sell"]
        data["Decision"] = np.select(conditions, choices, default="Hold")

        return data

    def preprocess_data(self, data):  # We preprocess the data, scale it and label it
        data = self.label_data(data)
        label_mapping = {"Buy": 0, "Hold": 1, "Sell": 2}
        data["Decision"] = data["Decision"].map(label_mapping)
        return data

    def train_model(self):  # Actual training of the model
        data = pd.read_csv("all_time_data.csv")
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data = self.preprocess_data(data)

        features = data[["Open", "High", "Low", "Close", "Volume"]]
        labels = data["Decision"]

        # Check data balance
        print(labels.value_counts())

        self.scaler = (
            MinMaxScaler()
        )  # We scale the features with sklearn's MinMaxScaler
        features_scaled = self.scaler.fit_transform(features)

        X, y = [], []
        for i in range(
            self.seq_length, len(features_scaled)
        ):  # We create the sequences for the model
            X.append(features_scaled[i - self.seq_length : i])
            y.append(labels.iloc[i])
        X, y = np.array(X), np.array(y)

        X = X.reshape(X.shape[0], -1)

        self.model = xgb.XGBClassifier(  # This is the XGB model we are using,
            # Allowing us to tweak the hyperparameters
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.model.fit(
            X, y
        )  # We fit the independent and dependent variables to the model
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)  # We save the model to disk via pickle
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self):  # If pretrained, load it
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def execute(
        self, data: pd.DataFrame
    ):  # Execute method from the TradingStrategy interface where we return the next prediction
        if self.model is None or self.scaler is None:
            self.load_model()

        data = data[["Open", "High", "Low", "Close", "Volume"]]

        features = data[["Open", "High", "Low", "Close", "Volume"]]
        features_scaled = self.scaler.transform(features)

        if len(features_scaled) < self.seq_length:
            return "Hold"

        X_input = features_scaled[-self.seq_length :]
        X_input = X_input.reshape(1, -1).astype(np.float32)

        prediction = self.model.predict(X_input)[0]
        decision_mapping = {0: "Buy", 1: "Hold", 2: "Sell"}
        decision = decision_mapping.get(prediction, "Hold")

        return decision
