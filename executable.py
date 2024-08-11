#!/usr/bin/env python
# coding: utf-8

# # Project Abtract:

# The project focuses on analyzing residential property prices to gain insights into the factors influencing the housing market. By examining various variables such as property characteristics, location, market trends, and economic indicators, this project aims to provide a comprehensive understanding of the factors driving home prices and identify key trends and patterns.
# 
# The project utilizes a dataset containing information on residential properties, including features such as square footage, number of bedrooms and bathrooms, location attributes (e.g., neighborhood, proximity to amenities), and corresponding sale prices. By analyzing these variables, the project seeks to uncover correlations and trends that can help stakeholders understand the dynamics of the housing market and make informed decisions regarding real estate investments, pricing strategies, and policy development.
# 
# Through exploratory data analysis and statistical modeling techniques, the project aims to identify the key factors influencing home prices. This includes assessing the impact of property characteristics (e.g., size, amenities), location factors (e.g., proximity to schools, transportation), and market trends (e.g., supply and demand, interest rates) on property values. Additionally, the project explores the relationships between economic indicators (e.g., GDP growth, employment rates) and home prices to understand the broader macroeconomic influences on the housing market.
# 
# The outcomes of this project have implications for real estate professionals, investors, policymakers, and prospective homeowners. Real estate professionals can utilize the findings to better understand market trends, advise clients, and develop effective pricing strategies. Investors can gain insights into factors that drive property value appreciation and identify lucrative investment opportunities. Policymakers can use the findings to inform housing policies, zoning regulations, and urban planning initiatives. Prospective homeowners can make more informed decisions regarding property purchases, taking into account factors that influence home prices and long-term value appreciation.
# 

# # Columns Details

# date: This column represents the date associated with the property listing or sale.
#     
# price: This column denotes the price of the property.
#     
# bedrooms: This column indicates the number of bedrooms in the property.
#     
# bathrooms: This column represents the number of bathrooms in the property.
#     
# sqft_living: This column denotes the total living area of the property in square feet.
#     
# sqft_lot: This column represents the total area of the lot or land on which the property is situated, measured in square feet.
#     
# floors: This column indicates the number of floors in the property.
#     
# waterfront: This column is a binary indicator (e.g., 0 or 1) that represents whether the property has a waterfront view.
#     
# view: This column indicates the level of view from the property, which could be a rating or a categorical value.
#     
# condition: This column represents the overall condition of the property, which could be a rating or a categorical value.
#     
# sqft_above: This column denotes the square footage of the property that is above ground level.
#     
# sqft_basement: This column represents the square footage of the property's basement, if applicable.
#     
# yr_built: This column indicates the year the property was originally built.
#     
# yr_renovated: This column represents the year the property was last renovated, if applicable.
#     
# street: This column specifies the street address or location of the property.
#     
# city: This column indicates the city where the property is located.
#     
# statezip: This column provides the state and ZIP code of the property.
#     
# country: This column represents the country where the property is located.

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot  as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
dataset = pd.read_csv('data.csv')
dataset.head()
dataset.info()
dataset.shape
dataset.head()
dataset.drop(['date'], axis = 1, inplace = True)
dataset.head()
dataset.country.value_counts()
dataset.drop(['country'], axis = 1, inplace = True)
dataset.head()
dataset.drop(['street', 'city'], axis = 1, inplace = True)
dataset.head()
dataset.isnull().sum()
dataset.columns
cordataset=dataset.drop(["statezip"],axis=1)
a4_dims = (10, 8)
fig, ax = plt.subplots(figsize=a4_dims)
cor = cordataset.corr()
sns.heatmap(cor, annot = True, cmap="YlGnBu")
a4_dims = (15, 5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x = dataset.bedrooms, y = dataset.price)
dataset.groupby('bedrooms').price.agg([len, min, max])
df = dataset[(dataset.bedrooms > 0) & (dataset.bedrooms < 9)].copy()
df.shape
df.statezip.value_counts()
a4_dims = (5, 18)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax = ax, x = df.price, y = df.statezip)
a4_dims = (15, 8)
fig, ax = plt.subplots(figsize=a4_dims)
sns.distplot(a = df.price, bins = 1000, color = 'r', ax = ax)
df.price.agg([min, max])
len(df[(df.price == 0)])
a4_dims = (15, 5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x = df.bathrooms, y = df.price)
df['statezip'] = df['statezip'].str.replace('WA ', '')
df['statezip'] = pd.to_numeric(df['statezip'])
zero_price = df[(df.price == 0)].copy()
zero_price.shape
zero_price.head()
sns.distplot(zero_price.sqft_living)
zero_price.agg([min, max, 'mean', 'median'])
sim_from_ori = df[(df.bedrooms == 4) & (df.bathrooms > 1) & (df.bathrooms < 4) & (df.sqft_living > 2500) & (df.sqft_living < 3000) & (df.floors < 3) & (df.yr_built < 1970)].copy()
sim_from_ori.shape
sim_from_ori.head()
sim_from_ori.price.mean()
yr_sqft = df[(df.sqft_living > 2499) & (df.sqft_living < 2900)].copy()
yr_price_avg = yr_sqft.groupby('yr_built').price.agg('mean')
plt.plot(yr_price_avg)
df.price.replace(to_replace = 0, value = 735000, inplace = True)
len(df[(df.price == 0)])
df.head()
df.drop(['sqft_above'], axis = 1, inplace = True)
df.shape
df = df.reset_index()
df.info()
X = df.iloc[:, 1:]
X.drop("price",inplace=True,axis=1)
X.shape
y = df.price
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.1, random_state=42)
print(len(X_train) / len(df))
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
print(len(X_test) / len(y_rem))
print(len(X_train))
print(len(X_val))
print(len(X_val))
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Assuming X_train is a pandas DataFrame with mixed feature name types
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_val)
mse = mean_squared_error(y_pred, y_val)
mse
y_val.head(10)
y_pred
X_test.columns = X_test.columns.astype(str)
y_pred_test = lin_reg.predict(X_test)
mse = mean_squared_error(y_pred_test, y_test)
mse
lin_reg.score(X_test, y_test)
y_test
y_pred_test
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(lin_reg, file)
with open('feature_columns.pkl', 'wb') as columns_file:
    pickle.dump(X_train.columns.tolist(), columns_file)
print("Feature Columns:")
print(feature_columns)