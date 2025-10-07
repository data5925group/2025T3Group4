import numpy as np
import pandas as pd


def load_data():
    data = np.load(r"E:\pythonProject\project\EDA\2-Data\Sydney_NewCastle\current_wind_20100101_20241231_Sydney.npz")
    data_used = data[list(data.keys())[0]]
    return data_used


def load_stings():
    data = pd.read_csv(r"E:\pythonProject\project\EDA\2-Data\Sydney_NewCastle\Sydney_stings.csv")
    filtered = data[data["stings_binary"] == 1]
    res = filtered[["time", "stings_sum"]]
    return res
