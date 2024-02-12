#!/usr/bin/env python
# coding: utf-8

# #### this notebook i did a while ago, but still effective one i hope it's will a good for you 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('/Users/prithwishghosh/Downloads/water_potability.csv')
df_new = pd.read_csv('/Users/prithwishghosh/Downloads/water_potability.csv')


# In[3]:


df.isnull().sum()


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe().T


# In[8]:


df.Potability.value_counts()


# In[9]:


import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import make_blobs

imputer = IterativeImputer(max_iter=10, random_state=0)
df_imp = imputer.fit_transform(df)


# In[10]:


imp_df = pd.DataFrame(df_imp, columns=df.columns)


# In[11]:


imp_df


# In[12]:


imp_df.isnull().sum()


# In[13]:


imp_df.corr().T


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(imp_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.show()


# In[15]:


X = imp_df.drop('Potability',axis=1)
y = imp_df['Potability']


# In[16]:


X.shape


# In[17]:


y.shape


# In[18]:


import sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[23]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[24]:


#### Here the accuracy is bit better so i am going to apply other machine learning model here only


# In[25]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(accuracy_score(y_pred,y_test))


# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_pred,y_test))


# In[27]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[28]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_pred,y_test))


# In[29]:


from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=4)
nn.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_pred,y_test))


# In[30]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[31]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[32]:


y_pred_xgboost = model.predict(X_test)


# In[33]:


accuracy = accuracy_score(y_test, y_pred_xgboost)
print(f'Accuracy: {accuracy}')


# In[34]:


dim = pd.read_csv('/Users/prithwishghosh/Downloads/dim1.csv')


# In[35]:


dim


# In[36]:


import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import make_blobs

imputer = IterativeImputer(max_iter=10, random_state=0)
dim_imp = imputer.fit_transform(dim)


# In[37]:


imp_dim = pd.DataFrame(dim_imp, columns=dim.columns)


# In[56]:


imp_dim


# In[57]:


imp_df


# In[39]:


X1 = imp_dim
y1 = imp_df['Potability']


# In[40]:


import sklearn
from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.3)


# In[41]:


from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)


# In[42]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X1_train,y1_train)
y1_pred = lr.predict(X1_test)


# In[43]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y1_test, y1_pred)
print("Accuracy:", accuracy)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X1_train,y1_train)
y1_pred = dt.predict(X1_test)
print(accuracy_score(y1_pred,y1_test))


# In[45]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X1_train,y1_train)
y1_pred = rfc.predict(X1_test)
print(accuracy_score(y1_pred,y1_test))


# In[46]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X1_train, y1_train)
y1_pred = clf.predict(X1_test)
accuracy = accuracy_score(y1_test, y1_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[47]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X1_train,y1_train)
y1_pred = rfc.predict(X1_test)
print(accuracy_score(y1_pred,y1_test))


# In[48]:


from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=4)
nn.fit(X1_train,y1_train)
y1_pred = rfc.predict(X1_test)
print(accuracy_score(y1_pred,y1_test))


# In[49]:


model = XGBClassifier()
model.fit(X1_train, y1_train)


# In[50]:


y1_pred_xgboost = model.predict(X1_test)


# In[51]:


accuracy = accuracy_score(y1_test, y1_pred_xgboost)
print(f'Accuracy: {accuracy}')


# In[129]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

