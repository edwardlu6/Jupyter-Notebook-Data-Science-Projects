#!/usr/bin/env python
# coding: utf-8

# # Edward Lu

# 1.(a) A one million dollar (USD) increase in healthcare expenditure predicts an increase of approximately 0.3269 year of life expectancy, holding the other variables constant.

# 1.(b) The R-squared in the table is approximately 0.707. Therefore, around 70.7 percentage of the variation in lifeex total is explained by the four independent variables.

# 1.(c) If all four independent variables are 0 in one observation, based on the 'Intercept' row, the model will predict around 78.3454 of 'lifeex_total' for that observation.

# 1.(d) Yes, we can. Based on the table, the p-value for 'gdp' is 0.000, which is smaller than 0.05. So we can reject the null hypothesis.

# 1.(e) 95% of the times the regression coefficient for fertility would be between -5.174 and -3.743.

# 1.(f)Yes, the evidence casts doubt on the validility. That is because a quadratic relationship implies a nonlinear association between life expectancy and independent variables. The linear regression is in form of DV = coefficient * IV + intercept. The exponents are 1 for all IVs. So we cannot apply a linear regression on a nonlinear relationship. The predictions from the linear regression will be inaccurate.

# 1.(g) We are conducting multiple linear regression. The dependent variable in this regression model is life expectancy. The predictors or independent variables are GDP (GDP (millions USD)), total population (pop total), healthcare spending (spendhealth (millions USD)), and fertility (fertility (children per capita)).

# 1.(h) The researcher is carrying out a statistical approach to regression. We can see all the statistics in the table.

# 2.(a) 
# The research is an observational study. That is because independent variables were not randomly assigned by the researchers. They only collected the information and study it. 

# 2.(b)
# One potential reciprocal causal relationship between the variables in this study is life expectancy and GDP. GDP can definitely influence life expectancy. Higher GDP can cause high life expectancy. Life expectancy can also influence GDP. High life expectancy means more human capitals and workforce, which can increase GDP.

# 2.(c) Yes. One possible confounder is war. War can significantly influence GPD, total population, healthcare spending and fertility. All the independent may drop because of the casualty in war. Also, war can influence life expectancy as war can cause mortality.

# 2.(d) The error will be Ecological Fallacy. That is because the researcher used correlation based on national level to draw conclusions about the individual units that made up the national level.

# 3.(a) Supervised machine learning means we have labelled data, both input and output data. However, unsupervised machine learning means we have unlabelled data, meaning only input data. The datasets in supervised machine learning is labelled while the datasets in unsupervised machine learning is unlabelled.

# 3.(b) RMSE and R-squared are quite different conceptually. Rooted Mean Squared Error(RMSE) measures the standard deviation of the residuals. R-squared is also called coefficient of determination. It measures the percentage of the variation in the dependent variable explained by the entire model. 

# 3.(c)
# An odd number of k in KNN can ensure our vote does not lead to ties. An even number of k in KNN may cause ties in our vote, which will be an ambiguous prediction. For example: if we have k=4 and 2 of class A and 2 of class B, it will cause an ambiguous prediction.

# 3.(d)
# When placing the initial centroids in k-means, we provide guidelines which are initializing with random placements of centroids(init="random"), number of clusters(n_clusters=#), and random state(random_state=#).
