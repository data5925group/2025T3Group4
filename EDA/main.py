import load_data
import general
import time_analysis
import spatial
import relation
import stings_analysis

channel_names = ["U_curr", "V_curr", "temp", "salinity", "U_wind", "V_wind"]
arr = load_data.load_data()
data_start_from = "2010-01-01"

summary = general.general(arr, channel_names)
print(summary)

time_series = time_analysis.daily_means(arr)
time_analysis.plot(time_series, channel_names, data_start_from, 30)
monthly_res = time_analysis.monthly(time_series, data_start_from, channel_names)
print(monthly_res)

spatial.plot_spatial_mean_var(arr, channel_names)
spatial.plot_spatial_mean_var_month(arr, channel_names, data_start_from, 2010, 1)

corr = relation.correlation(arr, channel_names)
print(corr)

stings_data = load_data.load_stings()
stings_analysis.stings_plot(time_series, channel_names, data_start_from, stings_data)