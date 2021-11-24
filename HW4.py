#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split


# In[35]:


data = load_breast_cancer()


# In[36]:


x = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.DataFrame(data.target,columns = ['target'])


# In[37]:


x.head()


# In[38]:


y.head()


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)


# In[40]:


from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd


# In[41]:


breast = load_breast_cancer()
breast_data = breast.data


# In[42]:


breast_data.shape


# In[43]:


breast_labels = breast.target
breast_labels.shape


# In[44]:


labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape


# In[45]:


breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features


# In[46]:


features_labels = np.append(features,'label')


# In[47]:


breast_dataset.columns = features_labels
breast_dataset.head()


# In[48]:


breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[49]:


breast_dataset.tail()


# In[50]:


from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)


# In[51]:


x.shape


# In[52]:


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)


# In[53]:


principal_breast_Df = pd.DataFrame(data = principalComponents_breast, 
columns = ['principal component 1', 'principal component 2'])


# In[54]:


principal_breast_Df.tail()


# In[55]:


#result of number 2
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Principal Component - 1',fontsize=12)
plt.ylabel('Principal Component - 2',fontsize=12)
plt.title("PCA of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1'], 
    principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[56]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2,random_state = 20)


# In[57]:


from sklearn.svm import SVC


# In[58]:


svc_model = SVC()


# In[59]:


svc_model.fit(X_train, Y_train)


# In[60]:


y_predict = svc_model.predict(X_test)


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix


# In[62]:


cm = np.array(confusion_matrix(Y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns = ['predicted_cancer', 'predicted_healthy'])
confusion


# In[63]:


print(classification_report(Y_test, y_predict))


# In[64]:



import numpy as np 
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
 

X = np.sort(5 * np.random.rand(40, 1), axis=0) 
y = np.sin(X).ravel() 
 

y[::5] += 3 * (0.5 - np.random.rand(8)) 
 
# ############################################################################# 
# Fit regression model 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_lin = SVR(kernel='linear', C=1e3) 
svr_poly = SVR(kernel='poly', C=1e3, degree=2) 
y_rbf = svr_rbf.fit(X, y).predict(X) 
y_lin = svr_lin.fit(X, y).predict(X) 
y_poly = svr_poly.fit(X, y).predict(X) 
 
# ############################################################################# 
# Look at the results 
lw = 2 
plt.scatter(X, y, color='darkorange', label='data') 
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model') 
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model') 
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model') 
plt.xlabel('data') 
plt.ylabel('target') 
plt.title('Support Vector Regression') 
plt.legend() 
plt.show() 


# In[ ]:




