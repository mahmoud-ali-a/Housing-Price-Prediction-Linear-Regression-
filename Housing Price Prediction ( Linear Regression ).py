#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[13]:


df = pd.read_csv('F:\\mahmoud ali\\oasis project\\task4\\Housing.csv')


# In[14]:


print(df.head())


# In[15]:


print(df.info())


# In[16]:


print(df.describe())


# In[17]:


print(df.isnull().sum())


# In[19]:


df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)


# In[24]:


print(df.columns)


# In[26]:


X = df.drop(columns=['price'])
y = df['price']


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


df_encoded = pd.get_dummies(df, drop_first=True)


# In[32]:


X = df_encoded.drop(columns=['price']) 
y = df_encoded['price']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[36]:


y_pred = model.predict(X_test)


# In[39]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[40]:


print(f"Mean Squared Error: {mse}")


# In[41]:


print(f"R-squared: {r2}")


# In[42]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# Recommendtions
# 
# 1- Improve Data Quality: Collecting more data could enhance the model's accuracy and effectiveness.
# 2- Add More Features: Adding new features, such as proximity to amenities, may improve predictions.
# 3- Use Advanced Models: Trying more complex machine learning models, like neural networks or decision trees, might yield better results.

# In[ ]:




