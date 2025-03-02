import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
from collections import Counter

def find_frequency(date_list):
    # Ensure date_list is a valid list (convert from string if necessary)
    if isinstance(date_list, str):
        try:
            date_list = ast.literal_eval(date_list)  # Convert string to actual list
        except (SyntaxError, ValueError):
            return "Error: Invalid date list format"

    # Remove empty, null, or improperly formatted dates
    clean_dates = []
    for date in date_list:
        try:
            if date and isinstance(date, str):  # Ensure it's a non-empty string
                parsed_date = datetime.strptime(date.strip(), "%Y-%m-%d %H:%M:%S")
                if parsed_date.year > 1700:  # Avoid invalid years like 0000
                    clean_dates.append(parsed_date)
        except ValueError:
            continue  # Skip any invalid date formats

    if len(clean_dates) < 2:
        return "Insufficient valid timestamps"

    # Calculate time differences in hours
    date_diffs = [(clean_dates[i+1] - clean_dates[i]).total_seconds() / 3600 for i in range(len(clean_dates)-1)]

    # Find the most common interval
    most_common_interval = Counter(date_diffs).most_common(1)[0][0]

    # Identify frequency based on the most common time difference
    if 0.9 <= most_common_interval <= 1.1:
        if set(date_diffs) == {1}:
            return int(24), "Hourly", "No"
        else:
            return int(24), "Hourly", "Yes"
    elif 23 <= most_common_interval <= 25:
        if set(date_diffs) == {24}:
            return int(7), "Daily", "No"
        else:
            return int(7), "Daily", "Yes"
    elif 165 <= most_common_interval <= 170:
        if set(date_diffs) == {168}:
            return int(52), "Weekly", "No"
        else:
            return int(52), "Weekly", "Yes"
    elif most_common_interval in [672, 696, 720, 744]:  # Monthly variations
        if set(date_diffs).issubset({672, 696, 720, 744}):
            return int(12), "Monthly", "No"
        else:
            return int(12), "Monthly", "Yes"
    elif 2160 <= most_common_interval <= 2208:  # Quarterly variations (~91 days)
        if set(date_diffs).issubset({2184, 2208, 2160}):
            return int(4), "Quarterly", "No"
        else:
            return int(4), "Quarterly", "Yes"
    elif most_common_interval in [8760, 8784]:  # Yearly variations
        if set(date_diffs).issubset({8760, 8784}):
            return int(1), "Yearly", "No"
        else:
            return int(1), "Yearly", "Yes"
    else:
        return int(2), f"Custom ({most_common_interval} hours)", "NA"

def find_strength(date, value, period):
  ts = pd.Series(value, index=dates)
  ts = ts.dropna()

  if len(ts) < 2 * period:
    return 15, 15
  decomposition = seasonal_decompose(ts, model='additive', period=period)


  var_trend = np.var(decomposition.trend + decomposition.resid)
  var_seasonal = np.var(decomposition.seasonal + decomposition.resid)
  if var_seasonal == 0 and var_trend != 0:
    F_t = max(0, 1 - (np.var(decomposition.resid) / var_trend))
    F_s = 10
    return F_t, F_s
  elif var_trend == 0 and var_seasonal != 0:
    F_t = 10
    F_s = max(0, 1 - (np.var(decomposition.resid) / var_seasonal))
    return F_t, F_s
  elif var_trend == 0 and var_seasonal == 0:
    F_t = 10
    F_s = 10
    return F_t, F_s
  else:
    F_t = max(0, 1 - (np.var(decomposition.resid) / var_trend))

    F_s = max(0, 1 - (np.var(decomposition.resid) / var_seasonal))
    return F_t, F_s

data = pd.read_parquet('1testv1.parquet')

print(len(data))
for i in range(len(data)):
  print("i is", i)
  train_data = data.iloc[i]

  dates_list = ast.literal_eval(train_data['date'])  # Convert string to list

  dates = pd.to_datetime(dates_list)

  values = np.array(train_data['value'])

  print("No. of attributes",len(values))


  period, _,_ = find_frequency(dates_list)
  print("period is", period)
  #F_t, F_s = find_strength(dates, values, period)

  strength_values = [find_strength(dates, val, period) for val in values]
  print("No. of strength values",len(strength_values))
  print(strength_values)



print("end")
