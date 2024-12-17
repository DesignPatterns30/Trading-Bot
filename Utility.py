import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta


# These are utility methods used in various parts of program
def get_spot_market(
    client, symbol, interval, minutes
):  # This method is used to get the spot market data
    now = datetime.utcnow()  # Getting the current time
    past = now - timedelta(minutes=minutes)  # Getting the time in the past
    candles = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=past.strftime("%d %b, %Y %H:%M:%S"),
        end_str=now.strftime("%d %b, %Y %H:%M:%S"),
    )
    return candles  # We return the candles (klines)


def write_last_n_minutes_data(
    client, symbol, csv_file, interval, minutes
):  # This method is used to write the last n minutes of data to a CSV file
    last_n_minutes_data = get_spot_market(
        client, symbol=symbol, interval=interval, minutes=minutes
    )
    df = pd.DataFrame(
        last_n_minutes_data,
        columns=[
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
            "Ignore",
        ],
    )
    df.to_csv(csv_file, index=False)


def write_last_3_years_Data(  # This method is used only once to train the XGB model, it writes the last 3 years of data to a CSV file
    client,
    symbol,
):
    last_5_years_data = get_spot_market(
        client,
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_5MINUTE,
        minutes=60 * 24 * 365 * 3,
    )
    df = pd.DataFrame(
        last_5_years_data,
        columns=[
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
            "Ignore",
        ],
    )
    df.to_csv("all_time_data.csv", index=False)


def read_csv(
    file_path,
):  # This method is used to read the CSV file, just to override the pandas read_csv method
    return pd.read_csv(file_path)
