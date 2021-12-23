#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("F:\ipynb files\ML Sessions\data.csv")


# In[2]:


data.head()


# In[3]:


X = data.iloc[:,0]
Y = data.iloc[:,1]
plt.scatter(X,Y)


# In[4]:


plt.show()


# In[5]:


#Predicting

m = 0
c = 0

Y_pred = m*X + c


# In[6]:


plt.scatter(X,Y)


# In[7]:


plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color = 'red')
plt.show()


# In[8]:


# Build Gradient Descent Model

# Initializing m, c, learning rate and iterations

m = 0 
c = 0 

L = 0.0001
iter_ = 1000

n = float(len(X))
print(n)


# In[9]:


# Performing the Gradient Descent

for i in range(iter_):
    Y_pred = m*X + c
    D_m = (-2/n)*sum(X*(Y-Y_pred))
    D_c = (-2/n)*sum(Y-Y_pred)
    m = m - (L*D_m)
    c = c - (L*D_c)
    print ("Slope & Intercept for iteration: ", i+1)
    print ("Slope: ",m)
    print ("Intercept: ",c)
    
print("After 1000 iterations values of slope: %2f and intercept: %2f"%(m,c))
Y_pred = m*X + c
plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color = 'red')
plt.show()


# In[ ]:




