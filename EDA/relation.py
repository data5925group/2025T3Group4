import pandas as pd
import matplotlib.pyplot as plt


def correlation(arr, channel_names):
    X = arr.transpose(1, 0, 2, 3)
    X_transpose = X.reshape(len(channel_names), -1).T
    data = pd.DataFrame(X_transpose, columns=channel_names)
    corr = data.corr()
    plt.imshow(corr)
    plt.xticks(range(len(channel_names)), channel_names)
    plt.yticks(range(len(channel_names)), channel_names)
    plt.colorbar()
    plt.title("Correlation")
    plt.show()
    return corr
