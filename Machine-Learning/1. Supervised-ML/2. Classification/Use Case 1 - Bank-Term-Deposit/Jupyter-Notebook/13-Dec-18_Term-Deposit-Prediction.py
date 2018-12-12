
# coding: utf-8

# # Problem Statement :

# We have a set of Predictor / Independent / Feature variables in order to predict y which is the Target / Dependent / Label Variable

# # Predictor / Independent Variables :

# 1. age (numeric)
# 2. job : type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
# 3. marital : marital status (categorical: “divorced”, “married”, “single”, “unknown”)
# 4. education (categorical: “basic.4y”, “basic.6y”, “basic.9y”, “high.school”, “illiterate”, “professional.course”, “university.degree”, “unknown”)
# 5. default: has credit in default? (categorical: “no”, “yes”, “unknown”)
# 6. housing: has housing loan? (categorical: “no”, “yes”, “unknown”)
# 7. loan: has personal loan? (categorical: “no”, “yes”, “unknown”)
# 8. contact: contact communication type (categorical: “cellular”, “telephone”)
# 9. month: last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
# 10. day_of_week: last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
# 11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
# 12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14. previous: number of contacts performed before this campaign and for this client (numeric)
# 15. poutcome: outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)
# 16. emp.var.rate: employment variation rate — (numeric)
# 17. cons.price.idx: consumer price index — (numeric)
# 18. cons.conf.idx: consumer confidence index — (numeric)
# 19. euribor3m: euribor 3 month rate — (numeric)
# 20. nr.employed: number of employees — (numeric)

# # Target / Dependent Variable :

# y - Has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)

# # Setting Location of DataSet (CSV File) :

# In[1]:


# Setting Location of Dataset :

location = "C:/Users/vetri/Desktop/JPA/Predictive Analytics/1. Logistic Regression/logistic_reg/Use Case 2 - Bank Term Deposit/bank/"


# In[2]:


# Importing Required Libraries :

import pandas as pd
import numpy as np



# Classification Algorithms :

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Metrics :

from sklearn import metrics as m
# from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import statsmodels.formula.api as smf


# Visualization :

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing CSV File into Pandas DataFrame :

# In[3]:


# Reading CSV File into Pandas DataFrame using ";" as Seperator :

data = pd.read_csv(location+'bank-full.csv',sep=";",header=0)

# View the First 5 rows of DataFrame :

data.head()


# In[4]:


# View the Last 5 rows of DataFrame :

data.tail()


# # Extracting Information on Columns :

# In[5]:


# Prints Information of All Columns :

data.info(verbose=True) 
# or 
# data.info()


# In[6]:


# Prints a Summary of Columns Count and its dtypes but not per column Information :

data.info(verbose=False)


# # Extracting Statistical Information on Numerical Columns :

# In[7]:


# Shows Descriptive Statistics Values on Numerical Value based Features :

data.describe()


# # Finding Correlation between Features and Class for Selection :

# ### 1. Using PairPlot :

# In[8]:


sns.pairplot(data)


# ### 2. Correlation Matrix :

# In[9]:


data.corr()


# ### 3. Heatmap to Visualise Correlation

# In[10]:


sns.heatmap(data.corr())


# <b>As per the Pair Plot, Correlation Matrix, and Heatmap, Observations are as follows :</b> 
# 1. Data is non-linear, asymmetric
# 2. Hence selection of features will not depend upon correlation factor.
# 3. Also not a single feature is correlated completely with class, hence requires combinantion of features.
# <br>
# 
# <b>Feature Selection techniques : </b>
# 1. Univariate Selection (non-negative features)
# 2. Recursive Feature Elimination (RFE)
# 3. Principal Component Analysis (PCA) (data reduction technique)
# 4. Feature Importance (decision trees)
# <br>
# 
# <b>Which feature selection technique should be used for our data?</b>
# 1. Contains negative values, hence Univariate Selection technique cannot be used.
# 2. PCA is data reduction technique. Aim is to select best possible feature and not reduction and this is classification type of data. PCA is also an unsupervised method, used for dimensionality reduction.
# 3. Hence Decision tree technique and RFE can be used for feature selection.
# 4. Best possible technique will be which gives extracts columns who provide better accuracy.
# 

# # Exploring Target Varaible :

# In[11]:


# Barplot for the Target / Dependent Variable :

sns.countplot(x='y',data=data)
plt.show()


# # Exploring Predictor Variables / Features :

# In[12]:


# View the (Rows,Columns) in DataFrame :

data.shape


# ### Find and Impute Missing Values : 

# In[13]:


# Sum of Missing Values in Each Column :

data.isnull().sum()

# NOTE : Since there are No Missing Values in any of the Columns, Imputation is not needed.


# ### Finding Duplicates using Unique and Value Counts :

# In[14]:


data.head(2)


# In[15]:


# Unique Education Values :

print(data['education'].unique())
print(data['education'].nunique())


# In[16]:


# Cross Tab to display Education stats with respect to y (ie) Target variable :

pd.crosstab(index=data["education"], columns=data["y"])


# In[17]:


# Education Categories and Frequency :

data.education.value_counts().plot(kind="barh")


# In[18]:


# Barplot for the Predictor / Independent Variable - job : 

sns.countplot(y="job", data=data)
plt.show()


# In[19]:


# Barplot for the Predictor / Independent Variable - marital : 

sns.countplot(x="marital", data=data)
plt.show()


# In[20]:


# Barplot for the Predictor / Independent Variable - default : 

sns.countplot(x="default", data=data)
plt.show()


# In[21]:


# Barplot for the Predictor / Independent Variable - housing : 

sns.countplot(x="housing", data=data)
plt.show()


# In[22]:


# Barplot for the Predictor / Independent Variable - loan : 

sns.countplot(x="loan", data=data)
plt.show()


# In[23]:


# Barplot for the Predictor / Independent Variable - poutcome : 

sns.countplot(x="poutcome", data=data)
plt.show()


# # Assumption 1 :
# 
# - Our prediction will be based on the customer’s job, marital status, whether he(she) has credit in default, whether he(she) has a housing loan, whether he(she) has a personal loan, and the outcome of the previous marketing campaigns. So, we will drop the variables that we do not need.

# In[24]:


data.head(2)


# In[25]:


# Removing 
# 0 - age
# 3 - education
# 8 - contact
# 9 - day
# 10 - month
# 11 - duration
# 12 - campaign
# 13 - plays

data.drop(data.columns[[0,3,8,9,10,11,12,13]],axis=1,inplace=True)


# In[26]:


data.head(2)


# # Exploring Categorical and Numerical Data into Digits Form :

# ### Converting Object Type to Integer using One-Hot Encoding :

# In[27]:


# Fetching Data Type of all Columns :

data.dtypes


# In[28]:


# Creating Dummies for Categorical Variables :

data_new = pd.get_dummies(data, columns=['job','marital',
                                         'default',
                                         'housing','loan',
                                         'poutcome'])

# data_new = pd.get_dummies(data, columns=['job','marital',
#                                          'education','default',
#                                          'housing','loan',
#                                          'contact','month',
#                                          'poutcome'])


# In[29]:


# Coverting Target Variable / Column into Binary Format :

data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)


# In[30]:


# Successfully converted Object data into  Integer data types

data_new.dtypes


# # Classifiers : Based on the values of different parameters we can conclude to the following classifiers for Binary Classification.
# 
# <br>
# 1. Gradient Boosting
# 2. AdaBoosting
# 3. Logistics Regression
# 4. Random Forest Classifier
# 5. Linear Discriminant Analysis
# 6. K Nearest Neighbour
# 7. Decision Tree
# 8. Gaussian Naive Bayes 
# 9. Support Vector Classifier

# # Performance Metric using Precision and Recall Calculation along with roc_auc_score & accuracy_score :

# In[31]:


# Creating Dictionary with Classifiers :

classifiers = {
               'Adaptive Boosting Classifier':AdaBoostClassifier(),
               'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),
               'Logistic Regression':LogisticRegression(),
               'Random Forest Classifier': RandomForestClassifier(),
               'K Nearest Neighbour':KNeighborsClassifier(8),
               'Decision Tree Classifier':DecisionTreeClassifier(),
               'Gaussian Naive Bayes Classifier':GaussianNB(),
               'Support Vector Classifier':SVC(),
               }
print(classifiers)


# In[32]:


# View the (Rows,Columns) in DataFrame :
# Due to One Hot Encoding Increase in the Number of Columns :

data_new.shape


# In[33]:


# Seperating Predictor and Target Columns into X and y Respectively :

data_X = data_new.drop(['y'], axis=1)
data_y = pd.DataFrame(data_new['y'])

print(data_X.columns)
print(data_y.columns)


# In[34]:


# Log Columns Headings :

log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","ROC_AUC_Score"]
log = pd.DataFrame(columns=log_cols)


# In[ ]:


# Metric Columns Headings :

metrics_cols = ["Precision Score","Recall Score","F1-Score","roc-ROC_AUC_Score"]
metric = pd.DataFrame(columns=metrics_cols)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3,random_state=2)
rs.get_n_splits(data_X,data_y)
for Name,classify in classifiers.items():
    for train_index, test_index in rs.split(data_X,data_y):
        print(Name)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X,X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        # Scaling of Features 
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        X_test = sc_X.transform(X_test)
        cls = classify
        cls =cls.fit(X,y)
        y_out = cls.predict(X_test)
        accuracy = m.accuracy_score(y_test,y_out)
        precision = m.precision_score(y_test,y_out,average='macro')
        recall = m.recall_score(y_test,y_out,average='macro')
        roc_auc = m.roc_auc_score(y_out,y_test)
        f1_score = m.f1_score(y_test,y_out,average='macro')
        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)
        metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
        log = log.append(log_entry)
        metric = metric.append(metric_entry)



#Scroll complete output to view all the accuracy scores and bar graph.


# In[ ]:


print(log)


# In[ ]:


print(metric)
metrics_cols = ["Precision Score","Recall Score","F1-Score","roc-ROC_AUC_Score"]


# In[ ]:


plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="g")  
plt.show()


# In[ ]:


plt.xlabel('Precision Score')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Precision Score', y='Classifier', data=log, color="g")  
plt.show()


# In[ ]:


plt.xlabel('Recall Score')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Recall Score', y='Classifier', data=log, color="g")  
plt.show()


# In[ ]:


plt.xlabel('F1-Score')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='F1-Score', y='Classifier', data=log, color="g")  
plt.show()


# In[ ]:


plt.xlabel('roc-ROC_AUC_Score')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='roc-ROC_AUC_Score', y='Classifier', data=log, color="g")  
plt.show()


# In[ ]:




