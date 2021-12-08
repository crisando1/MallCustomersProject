#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[23]:


data = pd.read_csv('Mall_Customers.csv')


# In[24]:


data


# In[25]:


data.head()


# In[26]:


data.rename(columns = {'Spending Score (1-100)':'Spending_Score','Annual Income (k$)':'Annual_Income'},inplace=True)
data.drop('CustomerID', axis=1, inplace=True)


# In[27]:


encoder = LabelEncoder()
data['Genre'] = encoder.fit_transform(data['Genre'])

gender_mappings = {index: label for index, label in enumerate(encoder.classes_)}
gender_mappings


# In[28]:


print('There are {} customers.'.format(len(data)))


# In[29]:


data.Genre.value_counts()


# In[30]:


sns.set() 
sns.countplot(x=data.Genre, color = 'black')
plt.show()


# In[31]:


pd.pivot_table(data,index=["Genre"],values=["Spending_Score"])


# In[32]:


mean = data.Age.mean()


# In[33]:


print('mean is ')
mean


# In[34]:


sns.histplot(data.Age,kde=True,color='blue')


# In[35]:


plt.figure(figsize=(10,5))
sns.scatterplot(x=data.Age,y=data.Spending_Score,s=100, color = 'purple')


# In[36]:


plt.figure(figsize=(10,5))
sns.scatterplot(data.Annual_Income,data.Spending_Score,color='orange')
plt.xlabel('Annual Income'),plt.ylabel('Spending Score'),plt.title('Spending Score vs Annual Income')
plt.show()


# In[37]:


data.head()


# In[50]:


kmeans = KMeans(n_clusters=5, random_state=0)

sc = StandardScaler()
data_std = sc.fit_transform(data)


# In[52]:


ssq_distance = []
for k in range(1 ,10):
    cluster = KMeans(n_clusters=k, random_state=0)
    cluster.fit(data_std)
    ssq_distance.append(cluster.inertia_)
    
plt.figure(figsize = (10 ,5))
plt.plot(list(range(1,10)), ssq_distance)
plt.xlabel('Number of Clusters') , plt.ylabel('Sum of Squared Distance')
plt.show()


# In[40]:


kmeans.fit(data)
labels = kmeans.labels_


# In[41]:


sns.scatterplot(x=data.Age, y=data.Spending_Score,hue=labels,palette='Set1')
plt.title('Age Vs Spending Score')
plt.show()


# In[42]:


sns.scatterplot(x=data.Annual_Income, y=data.Spending_Score,hue=labels,palette='Set1')
plt.title('Annual Income Vs Spending Score')
plt.show()


# In[ ]:




