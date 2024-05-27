#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from factor_analyzer import FactorAnalyzer # Perform statistical tests before PCA 
import warnings
warnings.filterwarnings("ignore")
from statsmodels.formula.api import ols      # For n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import dataframe_image as dfi


# In[ ]:





# In[5]:



salarydata=pd.read_csv('SalaryData.csv')
salarydata


# ### Sample of the dataset: 
#  
# 

# In[4]:


len(salarydata)


# In[5]:


salarydata.head()


# ### Shape of the dataset
# 

# In[6]:


row, col = salarydata.shape
print("There are total {}".format(row), "rows and {}".format(col), "columns in the dataset")


# ### Let us check the types of variables in the data frame. 
# 

# In[7]:


salarydata.dtypes


# In[8]:


salarydata.info()


# ### Check for missing values in the dataset:

# In[9]:


salarydata.isnull().sum()


# ### Summary of Data frame

# In[10]:


salarydata.describe(include='all')


# ### 1.1 State the null and the alternate hypothesis for conducting one-way ANOVA for both Education and Occupation individually

# Null and Alternate Hypothesis for conducting one-way ANOVA for Education:
# 
# H0: The Salary slab for each educational level is equal
# 
# Ha: The Salary slab for each educational level is NOT equal 
# 
# Null and Alternate Hypothesis for conducting one-way ANOVA for Occupation:
# 
# H0: The Salary slab for each occupation type is equal
# 
# Ha: The Salary slab for each occupation type is not equal.
# 
# 

# ### 1.2 Perform a one-way ANOVA on Salary with respect to Education. State whether the null hypothesis is accepted or rejected based on the ANOVA results.
# 

# In[11]:


formula = 'Salary ~ C(Education)'
model = ols(formula, salarydata).fit()
aov_table = anova_lm(model)
print(aov_table)


# ### 1.3 Perform a one-way ANOVA on Salary with respect to Occupation. State whether the null hypothesis is accepted or rejected based on the ANOVA results.
# 

# In[12]:


formula = 'Salary ~ C(Occupation)'
model = ols(formula, salarydata).fit()
aov_table = anova_lm(model)
print(aov_table)


# ### 1.4 If the null hypothesis is rejected in either (2) or in (3), find out which class means are significantly different. Interpret the result.

# In[13]:


tukey_res=pairwise_tukeyhsd(salarydata.Salary,salarydata.Education,0.05)
print(tukey_res)


# In[14]:


tukey_res1=pairwise_tukeyhsd(salarydata.Salary,salarydata.Occupation,0.05)
print(tukey_res1)


# ### 1.5 Analyze the effects of one variable on the other (Education and Occupation) with the help of an interaction plot

# In[15]:


sns.pointplot(x='Occupation', y='Salary', data=salarydata, hue='Education',ci=None);


# ### 1.6 Perform a two-way ANOVA based on Salary with respect to both Education and Occupation (along with their interaction Education*Occupation). State the null and alternative hypotheses and state your results. How will you interpret this result?

# In[16]:


formula = 'Salary ~ C(Education) + C(Occupation)'
model = ols(formula, salarydata).fit()
aov_table = anova_lm(model)
(aov_table)


# In[17]:


formula = 'Salary ~ C(Education) + C(Occupation)+ C(Education):C(Occupation)'
model = ols(formula, salarydata).fit()
aov_table = anova_lm(model)
(aov_table)


# In[ ]:





# # 2. Education Post 12th Standard

# In[18]:


edu12=pd.read_csv('Education+-+Post+12th+Standard.csv')


# ### Sample of the dataset: 
# 

# In[19]:


edu12.head()


# ### Summary of dataset
# 

# In[20]:


edu12.describe(include='all')


# ### Shape of the dataset
# 

# In[21]:


row, col = edu12.shape
print("There are total {}".format(row), "rows and {}".format(col), "columns in the dataset")


# ### Let us check the types of variables in the data frame. 
# 

# In[22]:


edu12.dtypes


# In[23]:


edu12.info()


# ### Check for missing values in the dataset: 
# 

# In[24]:


edu12.isnull().sum()


# ### Check for presence of duplicate rows

# In[25]:


edu12.duplicated().sum()


# ### Drop all columns other than the ones suitable for PCA
# 

# In[26]:


edu12_pca = edu12.drop(['Names'], axis = 1)


# In[27]:


edu12_pca.head()


# ### 2.1 Perform Exploratory Data Analysis [both univariate and multivariate analysis to be performed]. What insight do you draw from the EDA?

# ### Univariate

# In[28]:


def univariateAnalysis_numeric(column,nbins):
    print("Description of " + column)
    print("----------------------------------------------------------------------------")
    print(edu12_pca[column].describe(),end=' ')
    
    
    plt.figure()
    print("Distribution of " + column)
    print("----------------------------------------------------------------------------")
    sns.distplot(edu12_pca[column], kde=False, color='g');
    plt.show()
    
    plt.figure()
    print("BoxPlot of " + column)
    print("----------------------------------------------------------------------------")
    ax = sns.boxplot(x=edu12_pca[column])
    plt.show()


# In[29]:


pca_num = edu12_pca.select_dtypes(include = ['float64', 'int64'])
lstnumericcolumns = list(pca_num.columns.values)
len(lstnumericcolumns)


# In[30]:


for x in lstnumericcolumns:
    univariateAnalysis_numeric(x,20)
     


# ### Multivariate

# In[1]:


corr = pca_num.corr(method='pearson')


# In[2]:


mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
fig = plt.subplots(figsize=(10, 9))
sns.heatmap(pca_num.corr(), annot=True,fmt='.2f',mask=mask)
plt.xticks(rotation = 90)
plt.show()


# In[49]:


sns.pairplot(pca_num)


# In[ ]:





# ### 2.2 Is scaling necessary for PCA in this case? Give justification and perform scaling.?

# In[33]:


from scipy.stats import zscore
pca_num_scaled=pca_num.apply(zscore)
pca_num_scaled.head()


# In[68]:


dfi.export(pca_num_scaled.head(),'Scaled.png')


# Histogram before scaling

# In[51]:


pca_num.hist(figsize=(20,30));


# Histogram after scaling

# In[52]:


pca_num_scaled.hist(figsize=(20,30));


# In[80]:


pca_num_scaled.describe(include='all').T


# In[81]:


dfi.export(pca_num_scaled.describe(include='all').T,'des scal.png')


# ### 2.3 Comment on the comparison between the covariance and the correlation matrices from this data [on scaled data].

# ### Covariance Matrix

# In[75]:


pca_num_scaled.cov()


# In[77]:


dfi.export(pca_num_scaled.cov(),'cov.png') 


# ### Correlation Matrix

# In[54]:


mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
fig = plt.subplots(figsize=(10, 9))
sns.heatmap(pca_num_scaled.corr(), annot=True,fmt='.2f',mask=mask)
plt.xticks(rotation = 90)
plt.show()


# In[35]:


pca_num_scaled.corr()


# In[78]:


dfi.export(pca_num_scaled.corr(),'corr.png')


# ### 2.4 Check the dataset for outliers before and after scaling. What insight do you derive here? [Please do not treat Outliers unless specifically asked to do so]

# Before scaling

# In[36]:


plt.figure(figsize = (12,8))
feature_list = pca_num.columns
for i in range(len(feature_list)):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(y = pca_num[feature_list[i]], data = pca_num)
    plt.title('Boxplot of {}'.format(feature_list[i]))
    plt.tight_layout()


# In[ ]:





# After Scaling

# In[37]:


plt.figure(figsize = (12,8))
feature_list = pca_num_scaled.columns
for i in range(len(feature_list)):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(y = pca_num_scaled[feature_list[i]], data = pca_num_scaled)
    plt.title('Boxplot of {}'.format(feature_list[i]))
    plt.tight_layout()


# ### 2.5 Extract the eigenvalues and eigenvectors. [Using Sklearn PCA Print Both]

# In[38]:


from sklearn.decomposition import PCA
pca = PCA(n_components=17, random_state=123)
pca_transformed = pca.fit_transform(pca_num_scaled)


# #Extract eigen vectors

# In[39]:


pca.components_


# #Check the eigen values
# #Note: This is always returned in descending order
# 

# In[40]:


pca.explained_variance_


# ### 2.6 Perform PCA and export the data of the Principal Component (eigenvectors) into a data frame with the original features

# In[53]:


pca_extracted_loadings = pd.DataFrame(pca.components_.T,columns = ['PC1','PC2', 'PC3', 'PC4', 'PC5', 'PC6','PC7','PC8', 'PC9', 'PC10', 'PC11', 'PC12','PC13','PC14','PC15','PC16','PC17'],index = pca_num_scaled.columns)


# In[42]:


pca_extracted_loadings


# In[82]:


dfi.export(pca_extracted_loadings,'pca.png')


# In[83]:


total = sum(pca.explained_variance_)
var_exp = [( i /total ) * 100 for i in sorted(pca.explained_variance_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)


# In[84]:


pca_ex_load = pca_extracted_loadings.drop(['PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17'], axis = 1)


# In[85]:


pca_ex_load


# In[86]:


dfi.export(pca_ex_load,'pca load.png')


# ### 2.7 Write down the explicit form of the first PC (in terms of the eigenvectors. Use values with two places of decimals only). [hint: write the linear equation of PC in terms of eigenvectors and corresponding features]?

# In[87]:


PC1=([ 2.48765602e-01,  2.07601502e-01,  1.76303592e-01,
         3.54273947e-01,  3.44001279e-01,  1.54640962e-01,
         2.64425045e-02,  2.94736419e-01,  2.49030449e-01,
         6.47575181e-02, -4.25285386e-02,  3.18312875e-01,
         3.17056016e-01, -1.76957895e-01,  2.05082369e-01,
         3.18908750e-01,  2.52315654e-01])


# In[101]:


PC1_round=['%.2f' % elem for elem in PC1]


# In[102]:


PC1_round


# ### 2.8 Consider the cumulative values of the eigenvalues. How does it help you to decide on the optimum number of principal components? What do the eigenvectors indicate?

# In[46]:


pca.explained_variance_ratio_


# In[47]:


total = sum(pca.explained_variance_)
var_exp = [( i /total ) * 100 for i in sorted(pca.explained_variance_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)


# In[67]:


plt.figure(figsize=(8,5))
sns.lineplot(y=pca.explained_variance_ratio_,x=range(1,18),markers='o')
plt.xlabel('No. of components',fontsize=10)
plt.ylabel('variance explained',fontsize=10)
plt.title('Scree Plot',fontsize=12)
plt.grid()
plt.show()


# # End of Project
