#!/usr/bin/env python
# coding: utf-8

# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#Bunch dataType
iris = load_iris()

#print(iris.feature_names)
#print(iris.target)
#print(iris.target_names)
#print(type(iris.data))
#iris



x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#testing accuracy from k=1 through 25
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(scores[k])

scores_list

#dependence between K and the testing accuracy
plt.plot(k_range, scores_list)
plt.xlabel('K value')
plt.ylabel('Model accuracy')




# In[ ]:





# In[ ]:





# In[ ]:




