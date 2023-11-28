#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import datetime
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#LOADING THE DATASET
data = pd.read_csv(r"C:\Users\PC\Downloads\supermarket_sales - Sheet1.csv")


# In[7]:


data.head()


# In[8]:


data.describe()


# In[9]:


data.info()


# In[10]:


data.value_counts()


# In[11]:


data.shape


# In[12]:


data.dtypes


# In[13]:


data.columns


# In[14]:


#Checking Null Value
data.isnull().sum()



# In[15]:


data.isnull().any()


# In[16]:


#Exploratory Data Analysis
#HISTOGRAM

data.hist(figsize=(20,14))
plt.show()


# In[17]:


data.corr()


# In[18]:


#HEATMAP
plt.figure(figsize = (12,10))

sns.heatmap(data.corr(), annot =True)


# In[19]:


data.columns


# In[20]:


#BOXPLOT
plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,3,1)
sns.boxplot(x='Unit price',data=data)
plt.subplot(2,3,2)
sns.boxplot(x='Quantity',data=data)
plt.subplot(2,3,3)
sns.boxplot(x='Total',data=data)
plt.subplot(2,3,4)
sns.boxplot(x='cogs',data=data)
plt.subplot(2,3,5)
sns.boxplot(x='Rating',data=data)
plt.subplot(2,3,6)
sns.boxplot(x='gross income',data=data)



# In[21]:


#PAIRPLOT
sns.pairplot(data=data)


# In[22]:


#SCATTER PLOT
sns.scatterplot(x='Rating', y= 'cogs', data=data)


# In[23]:


#JOINTPLOT
sns.jointplot(x='Rating', y= 'Total', data=data)


# In[24]:


#CATPLOT
sns.catplot(x='Rating', y= 'cogs', data=data)


# In[25]:


#REGPLOT
sns.regplot(x='Rating', y='gross income', data=data)


# In[26]:


#LMPLOT
sns.lmplot(x='Rating', y= 'cogs', data=data)




# In[27]:


data.columns


# In[28]:


#KDE PLOT (DENSITY PLOT)

plt.style.use("default")

sns.kdeplot(x='Rating', y= 'Unit price', data=data)


# In[29]:


#LINEPLOT

sns.lineplot(x='Rating', y= 'Unit price', data=data)


# In[30]:


#BARPLOT
plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Unit price", data=data[170:180])
plt.title("Rating vs Unit Price",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Unit Price")
plt.show()



# In[31]:


data.columns


# In[32]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Gender", data=data[170:180])
plt.title("Rating vs Gender",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Gender")
plt.show()


# In[33]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Quantity", data=data[170:180])
plt.title("Rating vs Quantity",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Quantity")
plt.show()


# In[34]:


#lets find the categorialfeatures
list_1=list(data.columns)


# In[35]:


list_cate=[]
for i in list_1:
    if data[i].dtype=='object':
        list_cate.append(i)


# In[36]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[37]:


for i in list_cate:
    data[i]=le.fit_transform(data[i])


# In[38]:


data


# In[39]:


y=data['Gender']
x=data.drop('Gender',axis=1)


# In[40]:


#TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[41]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# In[42]:


# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)



# In[43]:


y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)


# In[44]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)





# In[45]:


RandomForestClassifier()


# In[46]:


y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)


# In[47]:


y=data['Gender']
x=data.drop('Gender',axis=1)


# In[48]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[49]:


y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)


# In[50]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression


# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


# In[51]:


# Calculate R-squared
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


# In[52]:


# Calculate MAE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)


# In[53]:


# Get feature importances
importances = rfc.feature_importances_

# Print feature importances
for i in range(len(importances)):
    print("Feature", i+1, "importance:", importances[i])


# In[55]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Predict on new data
y_pred = rfc.predict(x_test)
print("Predictions:", y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




