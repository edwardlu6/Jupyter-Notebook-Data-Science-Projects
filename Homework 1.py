#!/usr/bin/env python
# coding: utf-8

# 1.
# (a) The independent variable is minimum wage(increases in minimum wage). The dependent variable is “unemployment”.
# 
# (b) The independent variable is weather condition(rainy weather). The dependent variable is “bicycle usage”.
# 
# (c) The independent variables are “age, blood sugar levels and sedentary lifestyle”. The dependent variable is “heart rate”.
# 
# (d) The independent variable is “distance to the Broad Street pump”. The dependent variable is “cholera incidence”.
# 
# (e) The independent variables are plane weight and plane speed. The dependent variable is fuel consumption.

# 2.
# (a) One plausible confounder is the temperature of the environment. If the environment temperature is very low, the thermometer can go below zero and the water can freeze. So the thermometer going below zero does not cause the water to freeze.
# 
# (b) The talent of a guitar player can be a confounder. Great rock guitarists are usually talented in playing guitar and that can be the reason which makes them great. The talent of a guitar player can also make them prefer Fender Stratocaster because the guitar may have their desired sounds. But playing a Fender Stratocaster cannot make a guitar player to be great rock musicians. 
# 
# (c) The severity of patients’ health can be a confounder. Patients with severe diseases may be more likely to undergo a potentially dangerous experimental treatment. Severe diseases make them more difficult to recover even though they undergo an experimental treatment. 
# 
# (d) The soccer players’ proficiency can be a confounder. Soccer players’ proficiency can be the reason why they can score many goals. Proficient soccer player can also do acrobatic tricks with the ball. But only learning acrobatic tricks with the ball does not cause players to score more goals.

# 3.
# (a) The cats falling from open windows in New York City, which are 129 out of 132 cats. 
# (b) The treatment is the height from which the cats fall.
# The outcome is the cats' survival rate.
# (c) The study is an observational study. This is because the study focused on the cats fallen from different heights. The researchers did not control the heights from which cats fell. The researchers only observed cats falling from open windows in NYC and came up with a conclusion about it. So this is an observational study.
# (d) Cats can reach a “terminal velocity” when falling, which is 60 miles per hour. Once the terminal velocity is reached, cats may relax and stretch their legs out like a flying squirrel, which can increase air resistance and help to distribute the impact more evenly.
# (e) No it’s not feasible.To replicate the study with randomized treatment, the researchers need to drop cats from different heights. That can hurt the animals and raise ethical concerns.
# (f) The physical condition of cats can be one possible confounder. Some weak or small cats may be more vulnerable and have less chances of survival when falling from open windows. Whereas strong or big cats may have more chance of survival.

# 4.
# (a) The flaw in the 1987 JAMVA study is that the study was based only on cats which were brought to the hospital, which were the survived cats. The study ignored the dead cats which were not sent to hospital.
# (b) Survivor Bias.
# (c) Flying squirrel hypothesis does not apply to all falling cats situation. It may only be found on the survived cats. There might be cats died from falling but did not got observed. And the dead cats did not behaved like what is described in “flying squirrel hypothesis”.
# (d) The piece of evidence is that cats which had fallen from great heights usually had injuries indicating they had landed on their chest. It supports flying squirrel hypothesis.
# (e) No, because this only applies to the survived cats. A cat landing on its chest has a chance of surviving long enough to be brought into the hospital. Cats landing on other position might be dead instantly and did not send to the hospital. And researchers never had a chance to study the dead cats and concluded that all falling cats landed on their chest to survive.

# 5.
# (a) The independent variable is the number of passes. The dependent variable is the probability of scoring a goal.
# (b) The conclusion is only based on an observational study. Reep and Benjamin came up with this conclusion only based on statistics. It can suggest there is a correlation between the number of passes and the probability of scoring a goal. But it does mean there is a causual relationship. There are other confounding variables for the conclusion. There are many other factors that can increase the probability of scoring a goal. For example, the players’ skills or proficiency can also increase the probability of scoring a goal. So the number of passes does not have a causal relationship with the probability of scoring a goal.
# (c) No, it can only suggest correlation but not causation. And there are confounding variables. There are other factors that can increase the probability of scoring a goal. For example, the players’ improved skills or proficiency can also increase the probability of scoring a goal. So the number of passes does not have a causal relationship with the probability of scoring a goal.
# (d) The type of reasoning is counterfactual reasoning. The hypothetical situation is that if the team can score more goals with sequences of five or more passes. Thus, the causality can be found between the number of goals and the number of passes by comparing the goals with sequences of three passes or fewer and the goals with sequences of five passes or more.

# In[2]:


# 6(a)
import numpy as np
myarray = np.array([110000, 75000, 73000, 70000, 65000, 65000, 62000])


# In[3]:


# 6(b)
myarray[0] + myarray[6]


# In[4]:


# 6(c)
np.mean(myarray)
print(np.mean(myarray))


# In[5]:


# 6(d)
a = 1
b = "2"
print(a + b)
#The error is: unsupported operand type(s) for +: 'int' and 'str'. The problem is that String cannot be concatenated with Integer. String can only be concatenated with another String.


# In[6]:


# 6(e)
a = 1
b = "2"
print(a + int(b))


# In[7]:


# 6(f)
list = [62000, 60000]


# In[8]:


# 6(g)
list.append(60000)


# In[10]:


# 6(h)
new_array = np.array(list)


# In[12]:


# 7(a)
def function1(a, b):
    """Takes in two numeric values a and b, multiply them together and cube the product"""
    return (a*b)**3


# In[16]:


# 7(b)
print(function1(2,3))


# In[13]:


# 7(c)

def function2(a, b):
    """Takes in two numeric values a and b. If function1(a,b) is positive, return True. If function1(a,b) is negative, return False"""
    if function1(a,b) > 0:
        return True
    elif function1(a,b) < 0:
        return False


# In[14]:


# 7(d)
print(function2(1,1))
print(function2(-1,1))


# In[21]:


# 7(e)
def function3(a, b):
    if function1(a,b) > 0:
        return True
    elif function1(a,b) < 0:
        return None
        


# In[22]:


# 7(f)
print(function3(1,1))
print(function3(-1,1))


# In[23]:


# 7(g)
i = 5
while i < 12:
    print(i)
    i += 1.5
    print(i)

