import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

print(df.head())
print(df.describe())
print(df.shape)




# numpy imports numpy library and gives it the alias np, which is a common convention. This allows us to use numpy functions with the prefix np. for numerical computations and array operations
#import pandas which allows for data manipulation and analysis
# import matplotib.pyplot for creating plots and visualizations
#sklearn.datasets gets datasets to lead californias housing dataset
