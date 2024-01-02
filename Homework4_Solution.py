#!/usr/bin/env python
# coding: utf-8

# # Edward Lu 

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets
import seaborn as sns

#Citation of websites:
#https://kneed.readthedocs.io/en/stable/


# In[78]:


pip install kneed


# In[79]:


#1.（a)
data = pd.read_csv("fifa22.csv")
data.head()


# 1.(b)
# The unit of analysis in this dataset is soccer player. 

# In[80]:


#1.(c)
data.shape
#Therefore, we have 19630 observations and 20 features.


# In[81]:


#1.(d)
data['gender'].value_counts()
#Therefore, we know there are 19,239 male players and 391 female players. 


# 1.(e)
# No. That is because this dataset does not include enough female players. The male players number is much larger than the female players number. It's unbalanced. 

# In[82]:


#1.(f)

data = data.dropna(subset=['passing'])

data.shape


# In[83]:


#2.(a)
results = smf.ols('rank ~ passing + attacking + defending + skill', data=data).fit()     
results.summary()


# 2.(b)
# Based on R-squared, 70.5% of the variation in the rank is explained by the multiple linear regression model. 

# 2.(c)
# There are two features which are significant at the 1% level: attacking and defending. That is because the p-values of attacking and defending are 0.000 < 0.01. Therefore, they are significant. 

# 2.(d)
# Holding passing, attacking, and defending constant, a 1-unit increase in “skill” will increase 0.0066 unit in ranking. 

# 3.(a)
# I think these four features will do a good job at predicting rank for out-of-sample data. That is because the 70.5% of the variation in the rank is explained by the multiple linear regression model based on R-squared. And that means these four features can precisely predict rank for a large proportion of data.

# In[84]:


#3.(b)
X = data[['passing', 'attacking', 'defending', 'skill']]
print(X.head())
Y = data[['rank']]
print(Y.head())


# In[85]:


#3.(c)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=123)
X_train.head()


# In[86]:


#3.(d)
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("Intercept:",lr.intercept_)
print("Coefficients:",lr.coef_)


# 3.(e)
# The coefficients in the two regression models are slightly different from each other.  
# The coefficient for "attacking" in Q2 is slightly smaller than that in Q3. 

# In[87]:


#3.(f)
Y_pred = lr.predict(X_test)
for i in range(3):
    print(Y_pred[i])
    
    


# In[88]:


#3.(g)
plt.xlabel("Actual value of the Y validation data")
plt.ylabel("Predicted Y values for the X validation data")
plt.title("Scatter Plot of actual value and predicted values")

plt.scatter(Y_test,Y_pred)
plt.show()



# In[89]:


#3.(h)
print(f"RMSE: {metrics.mean_squared_error(Y_test, Y_pred, squared = False):.3f}")
print("RMSE measures the standard deviation of the residuals. On average, the model's predictions deviate by about 3.745 units from the true values.")


# 3.(i)
# I think this model does a good job in predicting player rank. That is because the scatter plot shows a linear relationship and RMSE is not too large. That means the actual value of Y and the predicted Y values are similar. 

# In[55]:


#4.(a)
data['preferred_foot'].value_counts()


# In[73]:


#4.(b)
right_foot_percent = 13044 / (13044 + 4406)
print(right_foot_percent)
print("Around 74.8% players prefer right foot. So we can make a correct classification for 74.8% of the time.")


# In[57]:


#4.(c)
X = data[["shooting", "passing", "dribbling", "defending", "attacking", "skill", "movement", "power", "mentality", "goalkeeping"]]
X.head()


# In[58]:


#4.(d)
sc = StandardScaler()

sc.fit(X)

X_scaled = sc.transform(X)

for i in range(3):
    print(X_scaled[i])
    print()
    
# Can we make to dataframe here?


# In[59]:


#4.(e)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, data['preferred_foot'], test_size=0.3, random_state=456)
for i in range(3):
    print(X_train[i])
    print()


# In[52]:


#4.(f)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

accuracies = []

for i in range (1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_std, Y_train)

    Y_pred = knn.predict(X_test_std)
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    print(i,"Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    accuracies.append(accuracy)
    
    
plt.plot(range(1, 31), accuracies, marker='o')
plt.title('KNN Classifier Accuracy for Different Values of k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.show()


# In[60]:


#4.(g)
# I choose k=27
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(X_train_std, Y_train)
Y_pred = knn.predict(X_test_std)
for i in range(3):
    print("Prediction", i+1, ":", Y_pred[i])


# In[62]:


#4.(h)
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Left', 'Right'])
cm_display.plot()
plt.show()

print("Based on the confusion matrix, approximately 1.3e+03 players who actually prefer their left foot (“True Lefts”) were predicted to prefer their right foot.")


# In[64]:


#4.(i)
from sklearn.metrics import classification_report
classification_report_result = classification_report(Y_test, Y_pred)

print(classification_report_result)

print("Recall is the ratio of correctly predicted positive observations to the all observations in actual class. The recall for \"Left\" suggests our model cannot capture a large portion players who prefer their left foot.")


# 4.(j)
# This model does a bad job of predicting a player’s preferred foot. That is because this model cannot correctly predict players who prefer left foot.

# In[65]:


#5.(a)
X_scaled_df = pd.DataFrame(X_scaled, columns=["shooting", "passing", "dribbling", "defending", "attacking", "skill", "movement", "power", "mentality", "goalkeeping"])

X_scaled_df.head()


# In[66]:


#5.(b)
new_df = X_scaled_df.sample(n=5000, random_state=2022)

new_df.head()


# In[67]:


#5.(c)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_values = []
silhouette_scores = []

for k in range (2,21):
    kmeans = KMeans(n_clusters=k, random_state=789)
    kmeans.fit(new_df)
    
    inertia_values.append(kmeans.inertia_)

    silhouette_scores.append(silhouette_score(new_df, kmeans.labels_))

print("Inertia values:")
print(inertia_values)


# In[68]:


#5.（d)

plt.plot(range(2, 21), inertia_values, marker='o')
plt.title('Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


# In[69]:


#5.(e)
from kneed import KneeLocator

plt.plot(range(2, 21), inertia_values, marker='o')
plt.title('Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

kneedle = KneeLocator(range(2, 21), inertia_values, curve='convex', direction='decreasing')
suggested_k = kneedle.knee

plt.vlines(suggested_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', label='Suggested Elbow')
plt.legend()

plt.show()

print("Suggested Elbow Value:", suggested_k)


# In[28]:


#5.(f)
plt.plot(range(2, 21), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()


# In[70]:


#5.(g)
kneedle = KneeLocator(range(2, 21), inertia_values, curve='convex', direction='decreasing')
reasonable_k = kneedle.knee

kmeans = KMeans(n_clusters=reasonable_k, random_state=789)

new_df['cluster_label'] = kmeans.fit_predict(new_df)

new_df.head()


# In[71]:


#5.(h)
plt.scatter(new_df['attacking'], new_df['defending'], c=new_df['cluster_label'], marker='o', cmap='viridis')
plt.title('Player\'s Attacking vs Defending Scores with Clusters')
plt.xlabel('Attacking Score')
plt.ylabel('Defending Score')
plt.colorbar(label='Cluster Label')


plt.show()


# 5.(i)
# Yes, I think clustering is a meaningful technique for this data. It is because using clustering we can clearly see classifications of players based on their attacking scores and defending scores.

# 5.(j)
# For further analyses, I'm interested in trying more independent variables such as "wage_eur" and "club" in the multiple linear regression model and then predict players' ranks. 
# Also, I want to try more the number of neighbors k in KNN classifiers to see if the accuracy can be improved further. 
# Finally, I want to find the the means of columns with numeric values and variance of them. I think that can be useful to know. Maybe I can plug all the means in a multiple linear regression model and predict the rank. Maybe I can get that point which all regression models must have. I'm not sure. 
