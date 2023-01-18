#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries we will be using

# In[52]:


# Importing common libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import statistics
import datetime

# Importing libraries for data visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import BASE_COLORS

# Importing libraries for data modeling 
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Importing libraries to ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# ### Exploring our Data

# In[2]:


data = pd.read_csv(r'C:\Users\galrodri\Documents\GitHub\data_mid_bootcamp_project_regression\regression_data.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.describe().T


# In[5]:


data.dtypes


# In[6]:


data['date'] = pd.to_datetime(data['date'], errors='coerce')


# In[7]:


data.isna().sum()


# In[8]:


data.nunique()


# #### Key takeaways
# 
# XXX

# # Data cleaning

# #### Step 1 - Removing duplicated rows

# In[9]:


# Check if there are more than 1 row for the same property ID. 
# It means that a property has been sold twice or more for the period we are analyzing 

dup = data[data.duplicated(['id'])]
dup.shape


# In[10]:


# We drop duplicated values and keep the most recent transaction 
data = data.drop_duplicates(['id'], keep='last')


# In[11]:


dup = data[data.duplicated(['id'])]
dup.shape


# #### Step 2 - Checking variables and removing irrelevant data for our analysis

# In[12]:


data['bedrooms'].value_counts()


# In[13]:


# We drop outliers for 10, 11 and 33
data = data[data['bedrooms']<=9]


# In[14]:


data['bathrooms'] = data['bathrooms'].round(decimals = 0)


# In[15]:


data['bathrooms'].value_counts()


# In[16]:


# We drop outliers for 0, 7 and 8
data = data[data['bathrooms']<=6]
data = data[data['bathrooms']!=0]


# In[17]:


data['bathrooms'].value_counts()


# In[18]:


data['waterfront'].value_counts()


# This is a boolean variable (Yes or No)

# In[19]:


data['view'].value_counts()


# This is a boolean variable (Yes or No)

# In[20]:


data['grade'].value_counts()


# In[21]:


# We drop outliers for 3
data = data[data['grade']>3]


# In[22]:


data['sqft_basement'].value_counts()


# This is an indicator if the property has basement or not, hence it can be converted to boolean variable

# In[23]:


data['yr_renovated'].value_counts()


# Almost all properties (20497) have not been renovated, hence we will drop this column as it's not relevant for our analysis
# Based on our limited data, we can't make hypothesis whether a renovation impacts or not in the Real Estate market.

# In[24]:


# We will drop irrelevant variables for our analysis
data.drop(['date', 'yr_renovated', 'lat', 'long','sqft_living','sqft_lot','sqft_above'], axis=1, inplace=True)
data.head()


# #### Step 3 - Modifying variables

# In[25]:


# Transforming waterfront, view and basement into Boolean variables

def feature_flag(x):
    if x == 0:  
        return 0
    else:
        return 1


# In[26]:


data['waterfront'] = data['waterfront'].apply(feature_flag)
data['view'] = data['view'].apply(feature_flag)
data['sqft_basement'] = data['sqft_basement'].apply(feature_flag)


# In[27]:


data['waterfront'].value_counts()


# In[28]:


data['view'].value_counts()


# In[29]:


data['sqft_basement'].value_counts()


# In[30]:


data = data.astype({'bathrooms': 'int64'})


# In[31]:


data = data.rename(columns={'sqft_basement':'has_basement','waterfront':'has_waterfront','view':'has_view','sqft_living15':'size'})


# In[32]:


data.dtypes


# #### Summary
# 1 - We dropped duplicated information when the house was sold twice or more for the period given
# 
# 2 - We removed outliers, removed irrelevant columns
# 
# 3 - We converted three variables into booleans to identify if the property has a feature or not

# # Data visualization

# In[33]:


mask=np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)]=True
fig, ax=plt.subplots(figsize=(14, 8))
ax=sns.heatmap(data.corr(), mask=mask, annot=True)
plt.show()


# We see that variables most correlated with price are: i) Grade, ii) # of Bathrooms, iii) Size (sqft_living15)
# In a second position, # of bedrooms and view are also variables positively correlated with price of the property
# 
# We see that id, condition, yr_built and sqft_lot15 have very low correlation, so we will not use them in our models

# #### Visualizing some of the variables

# In[34]:


sns.barplot(x = data['grade'].value_counts().index,
                y = data['grade'].value_counts()).set(title='Distribution of properties by Grade');


# In[35]:


sns.barplot(x = data['bathrooms'].value_counts().index,
                y = data['bathrooms'].value_counts()).set(title='Distribution of properties by # of Bathrooms');


# In[36]:


sns.barplot(x = data['bedrooms'].value_counts().index,
                y = data['bedrooms'].value_counts()).set(title='Distribution of properties by # of Bedrooms');


# # Data Modeling

# ### Preparing our data to model it

# In[37]:


# We create a new dataframe with the variables that we will be analyzing 
df = data[['grade','has_view', 'bedrooms', 'bathrooms', 'has_waterfront', 'floors', 'has_basement', 'size', 'price']]
df.head()


# In[38]:


df.shape


# #### Model 1 - Random Forest Regressor

# In[39]:



x = df.drop(columns = "price", axis=1)
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
# Create and train a Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=100)
regressor.fit(X_train, y_train)
# Use the trained model to make predictions on the test set
y_pred = regressor.predict(X_test)
# Evaluate the model's performance
score = regressor.score(X_test, y_test)
print("R^2:", score)


# #### Model 2 - Support Verctor Regression model

# In[45]:


SupportVectorRegModel = SVR()
SVR_model = SupportVectorRegModel.fit(X_train,y_train)


# In[46]:


SVR_predictions = SVR_model.predict(X_test)


# In[47]:


MAE_svr = mean_absolute_error(y_test,SVR_predictions)
MSE_svr = mean_squared_error(y_test,SVR_predictions)
RMSE_svr = np.sqrt(MSE_svr)
R2_svr = r2_score(y_test,SVR_predictions)

print("The mean absolute error of the model in the test set is: %6.2f" % (MAE_svr))
print("The mean squared error of the model in the test set is: %6.2f" % (MSE_svr))
print("The root mean squared error of the model in the test set is: %6.2f" % (RMSE_svr))
print("The R2 of the model in the test set is: %4.2f" % (R2_svr))


# #### Model 3 - Multiple Linear Regression

# In[48]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[49]:


lin_test_pred = regressor.predict(X_test)
reg = LinearRegression().fit(X_train, y_train)

score_mlr = round(reg.score(X_test, y_test)*100,2)
score_mlr


# In[53]:


y_pred = regressor.predict(X_test)


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Part 2: Factors driving higher property value (>= $650.0K)

# In[43]:


data_second_part = data[data.price >= 650000]
data_second_part.shape


# We have 5297 properties valued above $650.0K 

# In[44]:


mask=np.zeros_like(data_second_part.corr())
mask[np.triu_indices_from(mask)]=True
fig, ax=plt.subplots(figsize=(14, 14))
ax=sns.heatmap(data_second_part.corr(), mask=mask, annot=True)
plt.show()


# #### We see that the variables that are more correlated with the price are: i) Grade (51.0%), ii) # of Bathrooms (42.0%) and iii) size of the property (sqft_living15 38.0%)

# # Conclusions

# In[ ]:





# In[ ]:




