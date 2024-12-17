import time
import pandas as pd
from binance.client import Client
from Utility import write_last_n_minutes_data, read_csv
import os
from PyQt5.QtCore import QObject, pyqtSignal
from StrategyFactory import StrategyFactory
import abc
from dotenv import load_dotenv

load_dotenv()
symbol = os.getenv("SYMBOL")
csv_file = os.getenv("CSV_FILE")
api_key = os.getenv("API_KEY")
api_secret = os.getenv("SECRET_KEY")
client = Client(api_key, api_secret)


class QABCMeta(type(QObject), abc.ABCMeta):
    pass


class AbstractTradingBot(
    QObject, metaclass=QABCMeta
):  ## This is the abstract class we have defined
    update_progress = pyqtSignal(
        int
    )  ##These signals are related to the PyQT ui element
    fetched_data = (
        pyqtSignal()
    )  # Please ignore QObject, and QABCMeta implies that this class is an abstract class
    update_log = pyqtSignal(str)
    fetching_data = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(
        self, sleep_time=25
    ):  # Our constructor, we have defined some variables here with default values
        super().__init__()
        self.strategy = None
        self.symbol = symbol
        self.csv_file = csv_file
        self.client = client
        self.running = True
        self.data = pd.DataFrame()
        self.observers = []
        self.sleep_time = sleep_time

    def register_observer(
        self, observer
    ):  # Methods related to observer pattern are defined here
        self.observers.append(observer)  # To add observers

    def remove_observer(self, observer):  # To remove observers
        self.observers.remove(observer)

    def notify_observers(self, signal):  # To notify observers
        for observer in self.observers:
            observer.update(signal)  # Via SMTP or console,

    @abc.abstractmethod
    def create_strategy(self, metric):  # Abstract method to create strategy
        pass

    def calculate_strategy_metric(  # This is a template method, where we calculate the metric since it does not change among other trading bots.
        self, data
    ):  # Our metric, standard deviation of last 10 closing prices and the difference between the highest and lowest price in the last 10 minutes
        if len(data) < 10:
            self.update_log.emit("Not enough data to calculate standard deviation.")
            return None
        else:
            metric1 = data["Close"].tail(10).std()
            metric2 = max(data["High"].tail(10)) - min(data["Low"].tail(10))
            return (metric1 + metric2) / 2

    @abc.abstractmethod
    def predict_next_move(
        self,
    ):  # Abstract method to predict next move according to our current strategy
        pass

    @abc.abstractmethod
    def run(
        self,
    ):  # Abstract method to run the bot, if we have different types of bots, we will implement this method in the child class
        pass

    def stop(
        self,
    ):  # Method to stop the bot, no logic to change here so not abstract, template method
        self.running = False  # Default implementation, when set to False, bot's main loop will halt, effectively ending the program.


class TradingBot(
    AbstractTradingBot
):  # This is our concrete class that extends the AbstractTradingBot
    def create_strategy(
        self, metric
    ):  # Method to create strategy according to the metric we selected
        strategy = StrategyFactory.create_strategy(
            metric
        )  # Only thing strategy factory does is to create a strategy object, according to the metric we have selected
        strategy_name = strategy.__class__.__name__  # To get the strategy name
        self.update_log.emit(
            f"Creating {strategy_name} based on metric {metric:.2f}."
        )  # To log the strategy name and metric to the UI
        return strategy  # Return the strategy object created by the factory

    def predict_next_move(
        self,
    ):  # The method to predict the next signal,(buy, sell or hold)
        if (
            self.strategy is None
        ):  # According to the current strategy we have selected or altered at runtime
            self.update_log.emit("No strategy selected.")
            return

        signal = self.strategy.execute(self.data)  # Execute the strategy
        self.update_log.emit(f"Predicted Signal: {signal}")
        self.notify_observers(
            signal
        )  # Notify all observers, their update method will be called
        # Depending on which type of observer they are

    def run(
        self,
    ):  # This is the entry point of the program, where the bot will start running
        while self.running:  # The main loop of the bot
            write_last_n_minutes_data(  # The method to fetch the last n minutes of data(set to 15 here), with any interval
                self.client,  # We use the binance python API to fetch the data and write it to the CSV file.
                self.symbol,
                self.csv_file,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                minutes=15,
            )

            self.data = read_csv(self.csv_file)  # We read the data from the CSV file

            metric = self.calculate_strategy_metric(
                self.data
            )  # We calculate the metric
            self.update_log.emit(
                f"Metric result: {metric}"
            )  # We log the metric to the UI(from the latest data)
            last_data = self.data.tail(1)  # We get the last row of the data
            self.update_log.emit(
                f"Open: {str(last_data['Open'].tail().values[0])}"
            )  # We log the open, high, low, close prices to the UI
            self.update_log.emit(f"Close: {str(last_data['Close'].tail().values[0])} ")
            self.strategy = self.create_strategy(
                metric
            )  # We create the strategy according to the metric
            # And as you can see it is changed during RUNTIME...

            self.predict_next_move()  # We predict the next move according to the strategy we have just altered above
            self.update_log.emit(
                "Notified all observers via email."
            )  # We log that we have notified all observers via email
            self.update_log.emit(
                "-----------------------------" * 3
            )  # We log some dashes to separate the logs

            for i in range(
                self.sleep_time
            ):  # This is just for UI purposes, where we update the progress bar
                self.update_progress.emit(int((i + 1) / self.sleep_time * 100))
                time.sleep(1)

            self.fetched_data.emit()  # We emit the fetched data signal to the UI
