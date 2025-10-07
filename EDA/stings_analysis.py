import os
import pandas as pd
import matplotlib.pyplot as plt


def stings_plot(time_series, channel_names, start_date, stings_data):
    output_dir = "./stings_timeseries"
    dates = pd.date_range(start=start_date, periods=time_series.shape[0], freq="D")
    data = pd.DataFrame(time_series, index=dates, columns=channel_names)
    monthly_mean = data.resample("M").mean()

    stings_copy = stings_data.copy()
    stings_copy["time"] = pd.to_datetime(stings_copy["time"])
    stings_copy = stings_copy[(stings_copy["time"] >= dates.min()) & (stings_copy["time"] <= dates.max())]

    for i, name in enumerate(channel_names):
        plt.figure()
        plt.plot(monthly_mean.index, monthly_mean[name], label="monthly mean")

        y_vals = data.loc[stings_copy["time"], name].values
        plt.scatter(stings_copy["time"], y_vals, c="black")

        plt.title(name)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.show()
