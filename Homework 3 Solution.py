#!/usr/bin/env python
# coding: utf-8

# # Edward Lu

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[40]:


#1.(a)
def my_standardize(arr):
    mean =np.mean(arr)
    std=np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    standardized_arr = (arr - mean) / std
    return standardized_arr
    


# In[41]:


#1.(b)
def my_corr(arr1,arr2):
    std_arr1 = my_standardize(arr1)
    std_arr2 = my_standardize(arr2)

    sum_of_product = np.sum(std_arr1 * std_arr2)

    correlation = sum_of_product / (len(arr1))

    return correlation



# In[42]:


#1.(c)
data_df=pd.read_csv("states_data.csv")
prcapinc=data_df["prcapinc"]
vep12_turnout=data_df["vep12_turnout"]
plt.xlabel("prcapinc")
plt.ylabel("vep12_turnout")
plt.title("Scatter Plot of prcapinc vs vep12_turnout")
plt.scatter(prcapinc,vep12_turnout)
plt.show()


standard_prcapinc=my_standardize(prcapinc)
standard_vep12_turnout=my_standardize(vep12_turnout)
plt.xlabel("Standardized prcapinc")
plt.ylabel("Standardized vep12_turnout")
plt.title("Scatter Plot of Standardized prcapinc vs Standardized vep12_turnout")
plt.scatter(standard_prcapinc,standard_vep12_turnout)
plt.show()


# 1.(d)
# When the two variables are not standardized, the scatterplot shows that the correlation between the two variables is positive. When we standardize the two variables, the scatterplot also clearly shows the correlation between the two variables. But the scale after standardizing makes it easier to see the pattern. So the variables are easier to compare once standardized. 

# In[43]:


#1.(e)
my_corr(data_df["prcapinc"],data_df["vep12_turnout"])
#The Pearson's correlation coefficient between prcapinc and vep12_turnout is approximately 0.39. It means the linear association between the two variables is around 0.39. The implication is that there is a positive correlation between prcapinc and vep12_turnout but the correlation is not so strong.


# In[44]:


#2.(a)
def my_slope(x,y):
    regression_slope=my_corr(x,y) * (np.std(y) / np.std(x))
    return regression_slope


# In[45]:


#2.(b)
def my_intercept(x,y):
    regression_intercept=np.mean(y)-(my_slope(x,y)*np.mean(x))
    return regression_intercept


# In[46]:


#2.(c)
print(my_slope(prcapinc,vep12_turnout))
print(my_intercept(prcapinc,vep12_turnout))


# #2.(d)
# The regression slope means that a 1-unit increase in prcapinc produces a 0.0005735134590888054 units increase in vep12_turnout.
# The regression intercept means that a unit with a value of 0 of prcapinc will have a value of vep12_turnout of 41.61161411730768.
# These values tell us the predicted effect of a one unit change in prcapinc on an estimated variable vep12_turnout, which is increasing by 0.0005735134590888054. Also, when a unit with a value of prcapinc is 0, the value of vep12_turnout is 41.61161411730768.
# Correlation Coefficient can only tell us the strength of linear association between two varialbes and its direction(positive or negative).
# 

# In[47]:


#2.(e)
def predict_reg(b, a, x):
    y = a + b*x
    return y


# In[48]:


#2.(f)
array = [15000,25000,30000]

for i in array:
    print(predict_reg(my_slope(prcapinc,vep12_turnout), my_intercept(prcapinc,vep12_turnout), i))


# In[49]:


#2.(g)
three_points = [50.214316003639766,55.94945059452782,58.817017889971844]

data_df=pd.read_csv("states_data.csv")
prcapinc=data_df["prcapinc"]
vep12_turnout=data_df["vep12_turnout"]
plt.xlabel("prcapinc")
plt.ylabel("vep12_turnout")
plt.title("Scatter Plot of prcapinc vs vep12_turnout")
plt.scatter(prcapinc,vep12_turnout)
for i in range(3):
    plt.scatter(array[i],three_points[i],color='red')

plt.grid()
plt.show()



# #2.(h)
# The two points with prcapinc at 25000 and 30000 and their respective predictions are close to the observed data. But the point with prcapinc at 15000 is far away from observed data. So the predictions of the two points with prcapinc at 25000 and 30000 are trustworthy because 25000 and 30000 are in the range of observed data. But the prediction of the point with prcapinc at 15000 is not trustworthy because 15000 is far outside the range of observed data. It is an example of extrapolation.

# In[50]:


#3.(a)
results = smf.ols('vep12_turnout ~ prcapinc', data=data_df).fit()     
print(results.summary())
print()
print("Slope Coefficient:", 0.0006)
print("Z-statistics:",  2.939)
print("p-value:",  0.005)


# 3.(b)
# we can reject the hypothesis that there is no relationship between income and turnout in the population that the data is drawn from because the p-value in 3.(a) is 0.005 < 0.05, which is the general significance level. Therefore, the p-value is statistically significant enough to reject the null hypothesis. Also, the confidence interval does not contain zero. 

# In[51]:


#3.(c)
results = smf.ols('vep12_turnout ~ prcapinc + pop2010 + college + unemploy + urban', data=data_df).fit()     
print(results.summary())
print()


# 3.(d)
# Yes, the slope coefficient for prcapinc has changed. It decreases by 0.0001. That is because we incorporate more predictors into our regression model. 

# 3.(e)
# No, we fail to reject the null hypothesis that there is no relationship between income and turnout in the population that the data is drawn from. It is because the p-value is 0.155>0.05, which is the common significance level. The result in (c) is more trustworthy because multiple regression take many other independent variables into account and this is closer to the complex real-world situation. 

# 3.(f)R-squared in (a) is 0.153, which means 15.3% of the variation in turnout is explained by the entire model. R-squared in (c) is 0.294, which means 29.4% of the variation in turnout is explained by the entire model. The one in (c) is greater. I expect this result because in (c) we added more variables. And adding variables will push up R-squared. 

# In[1]:


#3.(g)
predictions= results.predict(data_df)
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.title("Scatter Plot of scatterplot ")
plt.scatter(predictions,results.resid)
plt.show()
print("No, there seems to be no trend.")


# In[53]:


#3.(h)
#correlation coefficient between fitted values and residuals
print(my_corr(predictions,results.resid))

#This correlation coefficient is very small and close to zero. It suggests there is no linear relation between the fitted values and residuals. Since there is no clear trend, heteroskedasticity is not psresent in the model in part(c). 


# In[54]:


#3.(i)
prcapinc = 'prcapinc'
pop2010 = 'pop2010'
college = 'college'
unemploy = 'unemploy'
urban = 'urban'


independent_variables = [data_df[prcapinc], data_df[pop2010], data_df[college], data_df[unemploy], data_df[urban]]

new_dataDF = pd.DataFrame(independent_variables, index=[prcapinc, pop2010, college, unemploy, urban]).T

print(new_dataDF.corr())


# 3.(j)
# prcapinc and college are highly correlated. This can be a problem for model in (c). It's because the correlated independent variables can cause multicollinearity. Multicollinearity implies that the standard error of the coefficients will likely be large. It's also difficult to find the individual effect of each independent variable on the dependent variable. 
