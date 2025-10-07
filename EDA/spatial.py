import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_spatial_mean_var(arr, channel_names):
    nan_position = np.any(np.isnan(arr), axis=1)
    nan_mask = nan_position[:, None, :, :]
    arr_masked = np.where(nan_mask, np.nan, arr)

    mean_maps = np.nanmean(arr_masked, axis=0)
    var_maps = np.nanvar(arr_masked, axis=0)

    output_dir = "./mean_var"

    for i, name in enumerate(channel_names):
        plt.figure()
        plt.imshow(mean_maps[i])
        title = f"Mean of {name}"
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"{title}.png"))
        plt.show()

        plt.figure()
        plt.imshow(var_maps[i])
        title = f"Variance of {name}"
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"{title}.png"))
        plt.show()


def plot_spatial_mean_var_month(arr, channel_names, start_date, year, month):
    T, C, H, W = arr.shape
    nan_mask = np.any(np.isnan(arr), axis=1)[:, None, :, :]
    arr_masked = np.where(nan_mask, np.nan, arr)

    dates = pd.date_range(start=start_date, periods=T, freq="D")
    mask = (dates.year == year) & (dates.month == month)
    idx = np.where(mask)[0]

    month_mean = np.nanmean(arr_masked[idx, :, :, :], axis=0)
    month_var  = np.nanvar(arr_masked[idx, :, :, :], axis=0)

    output_dir = "./mean_var_monthly"

    for i, name in enumerate(channel_names):
        plt.figure()
        plt.imshow(month_mean[i])
        title = f"Mean of {name} - {year}-{month}"
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"{title}.png"))
        plt.show()

        plt.figure()
        plt.imshow(month_var[i])
        title = f"Variance of {name} - {year}-{month}"
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"{title}.png"))
        plt.show()
