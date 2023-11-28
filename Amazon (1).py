#!/usr/bin/env python
# coding: utf-8

# In[421]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import os
import warnings
import datetime
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[360]:


path = r"C:\Users\PC\Desktop\Amazon.csv"
df = pd.read_csv(path, low_memory=False)
df.head()


# In[361]:


# Knowing the number of rows and columns
df.shape


# In[362]:


# Checking the missing values
df.isnull().sum()


# In[363]:


# Droping the columns with 70% or more missing data
perc = 70.0 
min_count =  int(((100 - perc)/100) * df.shape[1] + 1)
mod_df = df.dropna(axis = 1, thresh = min_count)


# In[364]:


# Number of columns are reduced significantly from 895 to 14
mod_df.shape


# In[365]:


# Checking the column name
mod_df.columns


# In[366]:


# Checking the data type, missing values in remaining columns
mod_df.info()


# In[367]:


# Change commas to dots and change the type to float
mod_df['discount_price'] = mod_df["Price"].astype(str).str.replace(',', '').astype(float)


# In[368]:


# Modify ratings values
mod_df['no_of_ratings'].unique()


# In[369]:


# Extract the digits and change the type to float
mod_df['ratings'] = mod_df['Rating'].replace(['Get','nan','₹68.99', '₹65','₹70', '₹100', '₹99', '₹2.99'], '0.0')
mod_df['ratings'] = mod_df["Rating"].astype(float)
mod_df['Rating'].unique()


# In[370]:


# Add column 'correct_no_of_ratings' which value is 'True' if 'no_of_ratings' begins from digit 
mod_df['no_of_ratings'] = mod_df['no_of_ratings'].astype(str)
mod_df['correct_no_of_ratings'] = pd.Series([mod_df['no_of_ratings'][x][0].isdigit() for x in range(len(mod_df['no_of_ratings']))])
# Drop columns with incorrect 'no_of_ratings'
mod_df = mod_df[mod_df['correct_no_of_ratings'] == True]
mod_df['correct_no_of_ratings'].value_counts()


# In[371]:


# Dataframe after first phase of cleaning
mod_df.head()


# In[372]:


mod_df.info()


# In[373]:


# Plot the total missing values
x = mod_df.isnull().sum()

fig = px.bar(x, orientation = "h",  text_auto='.2s',
            color_discrete_sequence= ["#ff6b00"] * len(x))
fig.update_layout(
    title="<b>Missing Value Count</b>",
    xaxis_title="Total missing values",
    yaxis_title="Column Names",
    plot_bgcolor = "#ECECEC",
    showlegend=False
)
fig.show()


# In[374]:


# Let us check and create a dataframe of missing ratings
missing_no_of_ratings = mod_df[mod_df['Price'].isnull()]

missing_no_of_ratings.head(2)


# In[375]:


# Since our further analysis is based on the price column so let us drop it.
df = mod_df.dropna(subset=['Price','MRP Price'])
df.head()


# In[376]:


df['manufacturer'] = df['Title'].str.split(' ').str[0]
cols = df.columns.tolist()
cols


# In[377]:


cols = ['Title',
 'Image-src',
 'single-href',
 'ASIN',
 'Brand',
 'Rating',
 'no_of_',
 'MRP Price',
 'Offer Price',
 'Price',
 'correct_no_of_',
 'no_of_ratings',
 'manufacturer']


# In[378]:


df.info()


# In[379]:


df['actual_price'] = df['MRP Price'].apply(str).apply(lambda x: x.replace(',', '')).astype(float) # convert to string, remove comma, and convert to float
df['discount_price'] = df['Price'].apply(str).apply(lambda x: x.replace(',', '')).astype(str) # convert to string, remove comma, and convert to float
df['discount_value'] = df['MRP Price'].astype(float) - df['Price'].astype(float) # subtract using the converted columns
df['discounting_percent'] = 1 - df['Price']/df['MRP Price']


# In[380]:


df.head()


# In[381]:


# Let us check the manufactures according to their prices
df[["MRP Price", 'manufacturer']].groupby("manufacturer").mean().round(2).sort_values(by = "MRP Price",
                                                                    ascending = False)


# In[382]:


df.info()


# In[383]:


# Detail of the minimum price row
df[df["Price"] == df["Price"].min()]


# In[384]:


# Detail of the maximum price row
df[df["MRP Price"] == df["MRP Price"].max()]


# In[385]:


# Detail of the minimum price row
df[df["discount_value"] == df["Price"].min()]


# In[386]:


# Let us check the common manufacture
values = df["manufacturer"].value_counts().keys().tolist()[:10]
counts = df["manufacturer"].value_counts().tolist()[:10]


# In[387]:


fig = px.bar(df, y = counts, x = values,
            color_discrete_sequence = ["#EC2781"] * len(df))


fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  yaxis_title = "Count",
                xaxis_title = "Name of Manufacturers",
                  title = "<b>Popular Manufacturers Category</b>"
                 )
fig.show()


# In[388]:


# Creating the dataframe of top 10 manufacturer
df_list = []
for i in values:
    x = df[df["manufacturer"] == i]
    df_list.append(x)
frame = pd.concat(df_list)
frame.head(2)


# In[389]:


# Average rating of the manufactures
frame[["manufacturer", "Rating"]].groupby("manufacturer").mean().sort_values(by = "Rating",
                                                ascending = False)


# In[390]:


# Different main categories present
frame["Brand"].unique()


# In[391]:


fig = px.bar(frame, "Brand", 
             color_discrete_sequence = ["#2377a4"] * len(frame))
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  yaxis_title = "Count",
                  xaxis_title = "Main Categories",
                  title = "<b>Count of Main Categories of Products</b>"
                 )
fig.show()


# In[392]:


#Let us select the 5 popular main categories

value_main = frame["Brand"].value_counts().keys().tolist()[:5]
count_main = frame["Brand"].value_counts().tolist()[:5]
value_main


# In[393]:


df_list = []
for i in value_main:
    x = frame[frame["Brand"] == i]
    df_list.append(x)
    #print(df)
frame = pd.concat(df_list)
frame.head(2)


# In[394]:


# Let us check the popular subcategory
import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)
frame_sub = frame[["Brand", "Rating"]].groupby("Brand").count()
frame_sub.style.background_gradient(cmap=cm)


# In[395]:


value_sub = frame["Rating"].value_counts().keys().tolist()[:10]
count_sub = frame["Rating"].value_counts().tolist()[:10]


# In[396]:


# New dataframe with selected sub_category
df_list = []
for i in value_sub:
    x = frame[frame["Rating"] == i]
    df_list.append(x)
frame = pd.concat(df_list)
frame.head(2)


# In[397]:


# Rating of the products
print("The average Rating: ",frame["Rating"].unique())

# After processing our data we have significantly reduced the size of the dataframe.
# Also the rating are now 4 or greater.
# Let us now check new average price ### check above before processing to compare.
print("The average Price: ", frame["MRP Price"].mean())


# In[398]:


import plotly.figure_factory as ff
x = frame["MRP Price"]
hist_data = [x]
group_labels = ["MRP Price"]

fig = ff.create_distplot(hist_data, group_labels, show_rug = False,
                        colors=["#ffd514"])
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  title = "<b>Price Distribution of Data</b>"
                 )

fig.show()


# In[399]:


# plot the quartiles and check for outliers 
fig = px.box(frame, "MRP Price")
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  title = "<b>Price Data Distribution</b>",
                 xaxis_title = "Price of Products"
                 )
fig.show()


# In[400]:


# Check the statistics of the price_new column
frame.Price.describe()


# In[401]:


# plot the quartiles and check for outliers 
fig = px.box(frame, "actual_price")
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  title = "<b>Price Data Distribution</b>",
                 xaxis_title = "Price of Products"
                 )
fig.show()


# In[402]:


# Let us check the rating of the products
fig = px.violin(frame, "Rating", 
               color_discrete_sequence = ["#FFBF00"] * len(frame))
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  xaxis_title = "Rating",
                  title = "<b>Rating Distribution of the Popular Products</b>"
                 )
fig.show()


# In[403]:


# Let us find the outliers
Q1 = 1399
Q2 = 2199
Q3 = 3599
IQR = Q3 - Q1
outlier1 = (Q1 - 1.5 * IQR)
outlier2 = (Q3 + 1.5 * IQR)
print("outlier1: ", outlier1)
print("outlier2: ", outlier2)


# In[404]:


outlier_price = []

for i in frame.actual_price:
    if i < outlier1 or i > outlier2:
        outlier_price.append("outlier")
    elif i > outlier1 or i < outlier2:
        outlier_price.append("normal")
    
frame["outlier_price"] = outlier_price


# In[405]:


fig = px.pie(frame, names = frame["outlier_price"], color = frame["outlier_price"],
             color_discrete_map={'normal': '#2377a4', 'outlier': '#ffd514'})

fig.update_layout(title = "<b>Distribution of Outlier</b>")

fig.show()


# In[406]:


# Let us see the outlier value
frame_outlier = frame.loc[frame["outlier_price"] == "outlier"].head()
frame_outlier


# In[407]:


print("Manufacturers with outlier price: ", frame_outlier.manufacturer.value_counts())


# In[408]:


# Let us check the rating of the products
fig = px.violin(frame, "ratings", 
               color_discrete_sequence = ["#FFBF00"] * len(frame))
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  xaxis_title = "Rating",
                  title = "<b>Rating Distribution of the Popular Products</b>"
                 )
fig.show()


# In[409]:


fig = px.histogram(frame, "no_of_ratings",
                  color_discrete_sequence = ["#8B4000"] * len(frame))
fig.update_xaxes(range=[10, 5000])
fig.update_yaxes(range=[0, 2000])
fig.update_layout(
                 plot_bgcolor = "#ECECEC",
                  xaxis_title = "Number of Reviews",
                  title = "<b>Number of Reviews Distribution</b>"
                 )
fig.show()


# In[410]:


# Let us check if there are any null review
print("Number of null values: ",frame['no_of_ratings'].isnull().sum())
# It seems that with high end products people love to leave a review


# In[411]:


frame.head(2)


# In[412]:


frame['no_of_ratings']


# In[440]:


#lets find the categorialfeatures
list_1=list(df.columns)


# In[441]:


list_cate=[]
for i in list_1:
    if df[i].dtype=='object':
        list_cate.append(i)


# In[442]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[443]:


for i in list_cate:
    df[i]=le.fit_transform(df[i])


# In[444]:


df


# In[445]:


y=df['Brand']
x=df.drop('Brand',axis=1)


# In[446]:


#TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[447]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# In[448]:


# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)


# In[449]:


y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)


# In[450]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[451]:


RandomForestClassifier()


# In[452]:


y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)


# In[453]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression


# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


# In[454]:


# Calculate R-squared
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


# In[455]:


# Calculate MAE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)


# In[456]:


# Get feature importances
importances = rfc.feature_importances_

# Print feature importances
for i in range(len(importances)):
    print("Feature", i+1, "importance:", importances[i])


# In[457]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Predict on new data
y_pred = rfc.predict(x_test)
print("Predictions:", y_pred)


# In[ ]:




