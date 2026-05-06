import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

print(df.head())
print(df.describe())
print(df.shape)


print(df.isnull().sum())

# numpy imports numpy library and gives it the alias np, which is a common convention. This allows us to use numpy functions with the prefix np. for numerical computations and array operations
#import pandas which allows for data manipulation and analysis
# import matplotib.pyplot for creating plots and visualizations
#sklearn.datasets gets datasets to lead californias housing dataset




# See how strongly each feature correlates with Price
print(df.corr()['Price'].sort_values(ascending=False))



plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()



# makine the model learn data
from sklearn.model_selection import train_test_split
X = df.drop('Price', axis=1) #the inputs
Y = df['Price']# what we predict


#radnom state is 42 so that the results are reproducible
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")





#now we need a scale since ML models with linear regressions needs a standard scale for the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) # learn scale from training data
X_test_scaled = scaler.transform(X_test) # apply same scale to test data


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, Y_train) #Where the model learns

print("model trained")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_}")