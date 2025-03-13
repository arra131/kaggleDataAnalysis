import numpy as np
import pandas as pd
import ast
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose

def calculate_highest_peak(ts):
    nlags = len(ts) - 1
    ts = ts.dropna()  # Drop missing values

    # Compute autocorrelation
    autocorr = acf(ts, nlags=nlags, fft=False)
    peaks, _ = find_peaks(autocorr)

    # Determine highest peak
    if len(peaks) > 0:
        highest_peak_index = peaks[np.argmax(autocorr[peaks])]
        if len(ts) < 2 * highest_peak_index:
            highest_peak_index = peaks[0]  # Set to first peak if condition met
    else:
        highest_peak_index = 0  # If no peaks, seasonal = 0 
        return highest_peak_index, 0, 0  # Return F_t, F_s as 0 if no peaks

    # Perform seasonal decomposition
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=max(2, highest_peak_index))
        
        # Calculate variances
        var_trend = np.var(decomposition.trend + decomposition.resid.dropna())
        var_seasonal = np.var(decomposition.seasonal + decomposition.resid.dropna())

        # Compute strength of trend
        F_t = 0 if var_trend == 0 or np.isnan(var_trend) or np.isinf(var_trend) else max(0, 1 - (np.var(decomposition.resid.dropna()) / var_trend))
        
        # Compute strength of seasonality
        F_s = 0 if var_seasonal == 0 or np.isnan(var_seasonal) or np.isinf(var_seasonal) else max(0, 1 - (np.var(decomposition.resid.dropna()) / var_seasonal))
    except:
        F_t, F_s = 0, 0
    
    return highest_peak_index, F_t, F_s

# Load data
data = pd.read_parquet(r"downloaded datasets\new\5testv1.parquet")
results = []

# Process each row in the data
for i in range(len(data)):
    try:
        print("Processing row:", i)
        train_data = data.iloc[i]
        name = train_data['name']  # Access the name directly
        dates_list = ast.literal_eval(train_data['date'])  # Convert string to list
        dates = pd.to_datetime(dates_list)
        value_column = train_data['value']  # Access the value column directly

        # Initialize lists to store results for each time series
        F_t_values = []
        F_s_values = []

        # Process each time series in value_column
        for val in value_column:
            ts = pd.Series(val, index=dates)
            season_length, F_t, F_s = calculate_highest_peak(ts)
            F_t_values.append(round(F_t, 4))
            F_s_values.append(round(F_s, 4))

        # Append results for this dataset
        results.append((name, season_length, F_t_values, F_s_values))

    except Exception as e:
        print(f"Error processing row {i}: {e}")
        continue

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=['name', 'seasonal', 'F_t', 'F_s'])

# Save results to Excel
results_df.to_excel(r'ana excel\new ana excel\5str.xlsx', index=False)
print("Processing complete. Results saved to '5str.xlsx'.")
