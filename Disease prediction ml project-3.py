#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyp
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score


# In[12]:


os.chdir("C:\\Users\\khush\\OneDrive\\Documents\\csv\\")
datasetTraining =  pd.read_csv("diseaseTraining.csv").dropna(axis=1)
datasetTesting =  pd.read_csv("diseaseTesting.csv")


# In[13]:


datasetTraining


# In[14]:


datasetTesting


# In[15]:


sns.countplot(x=datasetTraining["prognosis"])
pyp.xticks(rotation=90)
pyp.show()


# In[17]:


# all types of disease have same count.
#label encoding the progonosis column.

encoder = LabelEncoder()
datasetTraining["prognosis"] = encoder.fit_transform(datasetTraining["prognosis"])


# In[19]:


x = datasetTraining.iloc[:,:-2]
y = datasetTraining.iloc[:,-2]
x_tain, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)


# In[20]:


svm_model = SVC()
svm_model.fit(x_tain,y_train)


# In[21]:


pred = svm_model.predict(x_test)


# In[25]:


accuracy_score(pred,y_test*100)


# In[ ]:


#the accuracy is quite good.

