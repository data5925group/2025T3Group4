import numpy as np
import pandas as pd


def general(arr, channel_names):
    # overall = {}
    # overall["global"] = {
    #     "min": float(np.nanmin(arr)),
    #     "max": float(np.nanmax(arr)),
    #     "mean": float(np.nanmean(arr)),
    #     "std": float(np.nanstd(arr)),
    # }
    res = []
    for i, name in enumerate(channel_names):
        ch = arr[:, i, :, :]
        res.append({
            "channel": name,
            "mean": float(np.nanmean(ch)),
            "std": float(np.nanstd(ch)),
            "min": float(np.nanmin(ch)),
            "max": float(np.nanmax(ch)),
        })
    summary = pd.DataFrame(res)
    return summary
