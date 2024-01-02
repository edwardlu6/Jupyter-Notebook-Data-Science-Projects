#!/usr/bin/env python
# coding: utf-8

# # Edward Lu

# 1.(a)False
# (b)False
# (c)True
# (d)True
# (e)False

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#2.(a)
arr = np.array([11,4,1,1,2,8,12,60])
print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))


# In[3]:


#2.(b)
arr[7] = 6
print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))


# 2.(c) The mean is not so resistent to outliers. But the median is resistent to outliers. In other word, the median is more robust to outliers than the mean.

# 2.(d) That is because the standard deviation is to measure how far away the data points are from the sample mean on average. There is a "60" in the mis-recorded sample which is significantly far away from the sample mean. Thus, this increases the standard deviation. However, there are no outliers in the corrected sample. All numbers are close to the sample mean. So the standard deviation in the corrected sample is smaller than the mis-recorded sample.

# In[4]:


#2.(e)

def multiply():
    return(arr*3)
    
new_arr = multiply()
print(new_arr)


# In[5]:


#2.(f)
for i in range(len(arr)):
    arr[i]=arr[i]*3
print(arr)


# In[6]:


#3.(a)
np.random.seed(1234)
die4 = np.random.choice(range(1,5), size=10,replace=True) 
unique, frequency = np.unique(die4, return_counts = True)
plt.bar(unique,frequency,color=["red","blue","green","yellow"])
plt.xticks(unique)
plt.title("Distribution of times each value is rolled")
plt.xlabel("Faces")
plt.ylabel("Times Count")



# 3.(b)
# 3/10 came up '4' Theoretically, 1/4 should come up '4'. This is because we assume this is a fair die. So the four sides of the die should have equal chance which is 1/4. The proportion of '4' in the previous sample  is not 1/4 because it's a random sample with small sample size.

# In[7]:


#3.(c)
np.random.seed(1234)
die4 = np.random.choice(range(1,5), size=100,replace=True) 
unique, frequency = np.unique(die4, return_counts = True)
plt.bar(unique,frequency,color=["red","blue","green","yellow"])
plt.xticks(unique)
plt.title("Distribution of times each value is rolled")
plt.xlabel("Faces")
plt.ylabel("Times Count")


# In[8]:


#3.(d)
np.random.seed(1234)
die4 = np.random.choice(range(1,5), size=100,replace=True) 
unique, frequency = np.unique(die4, return_counts = True)
plt.bar(unique,frequency,color=["red","blue","green","yellow"])
plt.xticks(unique)
plt.title("Distribution of times each value is rolled")
plt.xlabel("Faces")
plt.ylabel("Times Count")
plt.axhline(25, color = "red", label = "the theoretical number of each outcome")


# In[9]:


#3.(e)
np.random.seed(1234)
die4 = np.random.choice(range(1,5), size=10000,replace=True) 
unique, frequency = np.unique(die4, return_counts = True)
plt.bar(unique,frequency,color=["red","blue","green","yellow"])
plt.xticks(unique)
plt.title("Distribution of times each value is rolled")
plt.xlabel("Faces")
plt.ylabel("Times Count")
plt.axhline(2500, color = "red", label = "the theoretical number of each outcome")


# 3.(f) This is because of Law of Large Number. Law of Large Number tells us that as the sample size grows, the sample proportion of occurances of an event will converge to a specific value, which is the population proportion or probability of the event. The sample proportion of '4's in the third experiment of 10,000 rolls is closer to 1/4 because the sample size is very large, which is 10,000 rolls. 

# In[10]:


#4.(a)

data_df = pd.read_csv("population_data.csv")
data_df.squeeze()

print(data_df.mean())
print(data_df.median())
print(data_df.std())


# In[11]:


#4.(b)
np.random.seed(1234)
sample = np.random.choice(data_df["0"], 25)
sample_mean = np.mean(sample)
sample_std = np.std(sample)
sample_median = np.median(sample)
print(sample_mean)
print(sample_std)
print(sample_median)


# In[12]:


#4.(c)
np.random.seed(1234)
sample = np.random.choice(data_df["0"], 1000)
sample_mean = np.mean(sample)
sample_std = np.std(sample)
sample_median = np.median(sample)
print(sample_mean)
print(sample_std)
print(sample_median)


# 4.(d) Yes, the sample with larger size outputs a mean, sd and median that are closer to the population quantities. It is because the law of large number. When the sample size increases, the sample mean will converge to the population mean. So do the sample sd and median.

# In[13]:


#4.(e)

def transform(x):
    y_1 = np.power(x,0.25)
    y_2 = - np.power(y_1,0.5)
    y_3 = np.exp(y_2)
    y_4 = np.sin(y_3)
    T = np.percentile(y_4,80)
    return T


# In[14]:


#4.(f)
print(transform(data_df))


# In[15]:


#4.(g)
np.random.seed(1234)
sample1 = np.random.choice(data_df["0"], 100)
print(transform(sample1))


# In[16]:


#4.(h)
np.random.seed(1234)
sample2 = np.random.choice(data_df["0"], 1000)
print(transform(sample2))
'''
The second one which has larger sample size is closer to the population statistics. 
LLN still seems to work well when statistics are complex.
'''


# In[17]:


#5.(a)
data_df = pd.read_csv("rents_county.csv")
data_df.head(5)


# In[18]:


#5.(b)

plt.hist(data_df.rent)
plt.title("The Distribution of rent") # title
plt.xlabel("rent") # x axis label
plt.ylabel("count") # y axis label
plt.show()

"""
The distribution is right-skewed.
"""


# In[19]:


#5.(c)
#i
print(data_df['rent'].size)
#ii
print(round(np.mean(data_df['rent']),1))
#iii
print(np.max(data_df['rent']))
#iv
print(round(np.std(data_df['rent']),1))


# In[20]:


#5.(d)
np.random.seed(1234)
sample = np.random.choice(data_df['rent'], 100)
print(round(np.mean(sample), 2))


# In[21]:


#5.(e)
np.random.seed(1234)
all_means = []
for i in range(10):
    sample = np.random.choice(data_df['rent'], 100)
    all_means.append(round(np.mean(sample), 2))
new_means = np.array(all_means)


plt.hist(new_means)
plt.title("Distribution of 10 sample means from sample-size 100") # title
plt.xlabel("Sample Means") # x axis label
plt.ylabel("Count") # y axis label
plt.axvline(round(np.mean(data_df['rent']),1), color = "red", label = "True Mean")
plt.axvline(round(np.mean(new_means),1), color = "black", label = "The mean of the sampled means")
plt.legend()
plt.show()


# In[22]:


#5.(e)
np.random.seed(1234)
all_means = []
for i in range(1000):
    sample = np.random.choice(data_df['rent'], 100)
    all_means.append(round(np.mean(sample), 2))
new_means = np.array(all_means)


plt.hist(new_means)
plt.title("Distribution of 1000 sample means from sample-size 100") # title
plt.xlabel("Sample Means") # x axis label
plt.ylabel("Count") # y axis label
plt.axvline(round(np.mean(data_df['rent']),1), color = "red", label = "True Mean")
plt.axvline(round(np.mean(new_means),1), color = "black", label = "The mean of the sampled means")
plt.legend()
plt.show()


# 5.(g)
# The mean of the sampling distribution converges to the population mean as the number of samples increases. 

# 5.(h)
# The shape of the sampling distribution of the mean is normally distributed as the number of samples increases.
# 

# 5.(i)
# Central Limit Theorem.
