"""import statistics
data =[2,4,6,8,10,12,14,16,18,20]
data_range=max(data)-min(data)

variance=statistics.variance(data)
mean=statistics.mean(data)
std_dev=statistics.stdev(data)
print(f"data: {data}")
print(f"data range: {data_range}")
print(f"variance: {variance}")
print(f"mean: {mean}")
print(f"standard deviation: {std_dev}")"""

import numpy as np
from scipy.stats import skew, kurtosis, percentileofscore

data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Calculate skewness and kurtosis
skewness = skew(data)
kurt = kurtosis(data)

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Calculate z-scores
z_scores = [(x - mean) / std_dev for x in data]

#calculate percentile of value 18
percentile_18 = percentileofscore(data, 18)

print("Skewness:", skewness)
print("Kurtosis:", kurt)
print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Z-scores:", z_scores)
print("Percentile of 18:", percentile_18)

