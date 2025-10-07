import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def daily_means(arr):
    daily = np.nanmean(arr, axis=(2, 3))
    return daily


def monthly(time_series, start_date, channel_names):
    dates = pd.date_range(start=start_date, periods=time_series.shape[0], freq="D")
    df = pd.DataFrame(time_series, index=dates, columns=channel_names)
    return df.groupby(df.index.month).mean()


def plot(time_series, channel_names, start_date, windows):
    dates = pd.date_range(start=start_date, periods=time_series.shape[0], freq="D")
    df = pd.DataFrame(time_series, index=dates, columns=channel_names)
    monthly_mean = df.resample("M").mean()
    month_all = monthly(time_series, start_date, channel_names)
    output_dir = "./timeseries"

    for i, name in enumerate(channel_names):
        plt.figure()
        pd.Series(time_series[:, i], index=dates).rolling(windows).mean().plot(label=f"{windows} days mean")

        plt.plot(monthly_mean.index, monthly_mean[name], label="monthly mean")
        seasonal = dates.to_series().dt.month.map(lambda m: month_all.loc[m, name])
        plt.plot(dates, seasonal.values, label="monthly overall")
        # plt.plot(dates, time_series[:, i], label="daily mean")
        title = f"{name}"
        plt.title(name)
        plt.legend()
        save_path = os.path.join(output_dir, f"{title}.png")
        plt.savefig(save_path)
        plt.show()
