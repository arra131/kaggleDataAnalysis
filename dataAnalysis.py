"""imports parquet file and extracts the summary of the dataset
    1. name, domain, data points, number of time series, number of values, number of dates, frequency, missing date
    2. saves the summary to dataset_summary.xlsx"""

import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import ast
import itertools    
import statsmodels.api as sm

# Load the dataset from the Parquet file
train_data = pd.read_parquet('7Atestv1.parquet')

# Function to find the frequency of a time series
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
            return "Hourly", "No"
        else:
            return "Hourly", "Yes"
    elif 23 <= most_common_interval <= 25:
        if set(date_diffs) == {24}:
            return "Daily", "No"
        else:
            return "Daily", "Yes"
    elif 165 <= most_common_interval <= 170:
        if set(date_diffs) == {168}:
            return "Weekly", "No"
        else:
            return "Weekly", "Yes"  
    elif most_common_interval in [672, 696, 720, 744]:  # Monthly variations
        if set(date_diffs).issubset({672, 696, 720, 744}):
            return "Monthly", "No"
        else:
            return "Monthly", "Yes"
    elif 2160 <= most_common_interval <= 2208:  # Quarterly variations (~91 days)
        if set(date_diffs).issubset({2184, 2208, 2160}):
            return "Quarterly", "No"
        else:
            return "Quarterly", "Yes"
    elif most_common_interval in [8760, 8784]:  # Yearly variations
        if set(date_diffs).issubset({8760, 8784}):
            return "Yearly", "No"
        else:
            return "Yearly", "Yes"
    else:
        return f"Custom ({most_common_interval} hours)", "NA"
    
def autocorr(data, lag):
    clean_data = data[~np.isnan(data)]
    if len(clean_data) > lag:
        return sm.tsa.acf(clean_data, nlags=lag, fft=False)[lag]
    else:            
        return "NA"
    

summary_data = []

# Iterate over each dataset
for i in range(len(train_data)):
    try:
        name = train_data['name'][i]
        print(i, "currenty analysig dataset : ", name)
        domain = train_data['domain'][i]
        data_points = float(train_data['DataPoints'][i])
        value_column = train_data['value'][i]
        date_column = train_data['date'][i]

        # number of variables (timeseries)
        num_timeseries = len(value_column)

        # total number of data points
        tmp = list(itertools.chain(*value_column))
        num_values = len(tmp)

        # Range of values
        min_value = round(min(tmp), 3)
        max_value = round(max(tmp), 3)

        # Missing values
        total_missing_val = sum(1 for x in tmp if pd.isna(x) or x == '') 
        missing_value = 'Yes' if total_missing_val > 0 else 'No'

        # total number of dates
        dateToList = ast.literal_eval(date_column)
        num_dates = len(dateToList)

        # Range of dates
        dateList = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in ast.literal_eval(date_column)]
        date_range = (max(dateList) - min(dateList)).days

        # find frequency
        frequency, missing_date = find_frequency(dateToList)

        # Find autocorrelation for each variable
        autocorr_values = [autocorr(val, 1) for val in value_column]

        # Append extracted info
        summary_data.append([name, frequency, missing_date, data_points, num_dates, num_timeseries, date_range, num_values, min_value, max_value, 
                            missing_value, total_missing_val, autocorr_values, domain])

    except Exception:
        print(f"Error in dataset at index {i}. Skipping...")
        continue

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data, columns=["Name", "Frequency", "Missing Date", "Data Points", '# Dates', "# Time Series", "Date Range (Days)", 
                                                 '# Values', 'Min Val', 'Max Val', 'Missing Value', '# Missing Value', 'AutoCorrelation', 'Domain'])

# Save to Excel
summary_df.to_excel("7anaData.xlsx", index=False)

print("Summary saved to 2anaData.xlsx")
