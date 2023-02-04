#!/usr/bin/env python
# coding: utf-8

# # Importing the packages

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.stats import zscore
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# # Collecting the sales data

# In[5]:



sales=pd.read_csv(r"C:\Users\91707\Desktop\0Data Science  Programme Material\Data Science Case Studies\06 MLAI  Module4 Machine Learning\8 Linear Regression\G.csv")


# In[6]:


sales.shape


# In[11]:


sales.tail(20)


# In[ ]:





# In[7]:


sales.duplicated().any()


# In[22]:


sales.columns


# In[13]:


type(sales)


# In[19]:


sales.describe()


# # I Model_1 Sales with Advt

# we are building the model only on advt as per client requirment future 4Q sales predictions 
# budget allocating
# 2020 june 17L 
# 2020 sep 11L
# 2020 dec 9L
# 2021 march 16L

# # 1.1 Model_1(direct)

# In[24]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Sales~Advt',data=sales).fit()
model1=sm.stats.anova_lm(model)
model1
print(model.summary())


# In[26]:


pre1=model.predict()
pre1


# In[27]:


res1=sales['Sales'].values-pre1
res1


# In[12]:


pre_1=pd.DataFrame(pre1,columns=["pre1"])
pre_1


# In[13]:


res_1=pd.DataFrame(res1,columns=["res1"])
res_1


# In[14]:


zscore1=pd.DataFrame(zscore(res1),columns=['zscore1'])
zscore1


# In[15]:


sales1=pd.concat([sales,pre_1,res_1,zscore1],axis=1)
sales1
sales_1=pd.DataFrame(sales1)
sales_1


# In[16]:


zscore1[zscore1['zscore1']>1.96]


# In[17]:


zscore1[zscore1['zscore1']<-1.96]


# # 1.2 Model_1 Applying (Dummy)

# We are applying dummy,where the value is above 1.96 as 1 and below 1.96 as 0 because those are the outliers to improve the model dummy variable is used.

# In[18]:


a=sales_1.copy()
for i in range(0,len(a)):
    if(np.any(a['zscore1'].values[i]>1.96)):
        a['zscore1'].values[i]=0
    else:
        a['zscore1'].values[i]=1
        test=a['zscore1']
        test
sales_1['dummy']=test
sales_1


# In[19]:


x=sales_1[["Advt","dummy"]]
y=sales_1["Sales"]
y


# In[20]:


plt.scatter(y,res1)
plt.xlabel("res_adv")
plt.ylabel("Sales")


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=0)


# In[22]:


x_train


# In[23]:


y_train


# In[24]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[25]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[26]:


y_pred = regr.predict(x_test)
y_pred


# In[27]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[ ]:





# # 1.3 Model_1 Applying(Square)

# In[28]:


sales_1["sqr_Advt"]=sales_1["Advt"]**2
sales_1


# In[29]:


x=sales_1[["Advt","sqr_Advt"]]
y=sales_1["Sales"]
y


# In[30]:


plt.scatter(y,res1)
plt.xlabel("res_adv")
plt.ylabel("Sales")


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=0)


# In[32]:


x_train


# In[33]:


y_train


# In[34]:


x_test


# In[35]:


y_test


# In[36]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[37]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[38]:


y_pred = regr.predict(x_test)
y_pred


# In[39]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 1.4 Model_1 Applying(Square root)

# In[40]:


sales_1["squareRoot_Advt"]=sales1["Advt"]**(1/2)
sales_1


# In[41]:


x_ADVT=sales_1[["Advt","squareRoot_Advt"]]
y_ADVT=sales_1["Sales"]
y_ADVT


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x_ADVT, y_ADVT, test_size=0.20,random_state=0)


# In[43]:


x_train


# In[44]:


y_train


# In[45]:


x_test


# In[46]:


y_test


# In[47]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[48]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 1.5 Model_1 Applying(log10)

# In[ ]:


sales_1['log_Advt'] = np.log10(sales_1['Advt'])
sales_1   


# In[ ]:


x_advt=sales_1[["Advt","log_Advt"]]
y_advt=sales_1["Sales"]
y_advt


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_advt, y_advt, test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # II Model_2 Sales with PC
# 

# we are building the model only on PC as per client requirment future 4Q sales predictions budget allocating 2020 june 17L 2020 sep 11L 2020 dec 9L 2021 march 16L

# # 2.1 Model_2(direct)

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Sales~PC',data=sales).fit()
model2=sm.stats.anova_lm(model)
model2
print(model.summary())


# In[ ]:


pre2=model.predict()
pre2


# In[ ]:


pre_2=pd.DataFrame(pre2,columns=['pre2'])
pre_2


# In[ ]:


res2=sales['Sales'].values-pre2
res2


# In[ ]:


res_2=pd.DataFrame(res2,columns=['res2'])
res_2


# In[ ]:


zscore2=pd.DataFrame(zscore(res2),columns=['zscore2'])
zscore2


# In[ ]:


sales2=pd.concat([sales,pre_2,res_2,zscore2],axis=1)
sales2
sales_2=pd.DataFrame(sales2)
sales_2


# In[ ]:


zscore2[zscore2['zscore2']>1.96]


# In[ ]:


zscore2[zscore2['zscore2']<-1.96]


# In[ ]:


b=sales_2.copy()
for i in range(0,len(b)):
    if(np.any(b['zscore2'].values[i]>1.96)):
        b['zscore2'].values[i]=0
    else:
        b['zscore2'].values[i]=1         
        test=b['zscore2']
        test
sales_2['dummy']=test
sales_2


# # 2.2 Model_2 Applying(Square)

# In[ ]:


sales_2["sqr_PC"]=sales_2["PC"]**2
sales_2


# In[ ]:


x_PC=sales_2[["PC","sqr_PC"]]
y_PC=sales_2["Sales"]
x_PC


# In[ ]:


plt.scatter(y_PC,res2)
plt.xlabel("res_pc")
plt.ylabel("Sales")


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_PC, y_PC, test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 2.4 Model_2 Applying(SquareRoot)

# In[ ]:


sales_2["squareRoot_PC"]=sales_2["PC"]**(1/2)
sales_2


# In[ ]:


x_PC=sales_2[["PC","squareRoot_PC"]]
y_PC=sales_2["Sales"]
y_PC


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_PC, y_PC, test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 2.5 Model_2 Applying(Log10)

# In[ ]:


sales_2['log_PC'] = np.log10(sales_2['PC'])
(sales_2)          


# In[ ]:


x_pc=sales_2[["PC","log_PC"]]
y_pc=sales_2["Sales"]
y_pc


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_pc, y_pc, test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# y_pred = regr.predict(x_test)
# y_pred

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # III Model_3 Sales with Advt and PC 

# we are building the model only on Advt and PC as per client requirment future 4Q sales predictions budget allocating 2020 june 17L 2020 sep 11L 2020 dec 9L 2021 march 16L

# # 3.1 Model_3(direct)

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Sales~Advt+PC',data=sales).fit()
model3=sm.stats.anova_lm(model)
model3
print(model.summary())


# In[ ]:


pre3=model.predict()
pre3


# In[ ]:


pre_3=pd.DataFrame(pre3,columns=['pre3'])
pre_3


# In[ ]:


res3=sales['Sales'].values-pre3
res3


# In[ ]:


res_3=pd.DataFrame(res3,columns=['res3'])
res_3


# In[ ]:


zscore3=pd.DataFrame(zscore(res3),columns=['zscore3'])
zscore3


# In[ ]:


zscore3[zscore3['zscore3']>1.96]


# In[ ]:


zscore3[zscore3['zscore3']<-1.96]


# In[ ]:


sales3=pd.concat([sales,pre_3,res_3,zscore3],axis=1)
sales3
sales_3=pd.DataFrame(sales3)
sales_3


# # 3.2 Model_3 Applying(Dummy)

# In[ ]:


c=sales_3.copy()
for i in range(0,len(c)):
    if(np.any(c['zscore3'].values[i]<-1.96)):
        c['zscore3'].values[i]=0
    else:
        c['zscore3'].values[i]=1         
        test=c['zscore3']
        test
sales_3['dummy']=test
sales_3


# # 3.3 Model_3 Applying(Square)

# In[ ]:


sales_3["sqr_pc"]=sales_3["PC"]**2
sales_3


# In[ ]:


x_adpc =sales_3[["Advt","PC","sqr_pc"]]
y_adpc = sales_3['Sales']
y_adpc


# In[ ]:


plt.scatter(y_adpc,res3)
plt.xlabel("res_adpc")
plt.ylabel("Sales")


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_adpc,y_adpc,test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 3.4 Model_3 Applying(Square Root)

# In[ ]:


sales_3["squareRoot_pc"]=sales_3["PC"]**(1/2)
sales_3


# In[ ]:


x_adpc=sales_3[["Advt","PC","squareRoot_pc"]]
y_adpc=sales_3["Sales"]
y_adpc


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_adpc,y_adpc,test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 3.5 Model_3 Applying(Log10)

# In[ ]:


sales_3['log_PC'] = np.log(sales_3['PC'])
(sales_3)


# In[ ]:


x_ADPC =sales_3[["Advt","PC","log_PC"]]
y_ADPC = sales_3['Sales']
x_ADPC


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_ADPC,y_ADPC,test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


x_test
y_test


# In[ ]:





# In[ ]:




