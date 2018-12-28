#!/usr/bin/env python
# coding: utf-8

# # Problem Statement :

# The <b>Bank Marketing data</b> is related with <b>direct marketing campaigns</b> of a Portuguese banking institution. 
# 
# - The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
#    in order to access if the product (bank term deposit) would be (or not) subscribed. 
# <br>
# 
# - The <b>classification goal</b> is to <b>predict</b> if the <b>client will subscribe a term deposit (variable y)</b>.

# # Predictor / Independent Variables :

# <b>1. age :</b> (numeric)
# <br>
# 
# <b>2. job :</b> type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
# <br>
# 
# <b>3. marital :</b> marital status (categorical: “divorced”, “married”, “single”, “unknown”)
# <br>
# 
# <b>4. education :</b> (categorical: “basic.4y”, “basic.6y”, “basic.9y”, “high.school”, “illiterate”, “professional.course”, “university.degree”, “unknown”)
# <br>
# 
# <b>5. default :</b> has credit in default? (categorical: “no”, “yes”, “unknown”).
# <br>
# 
# <b>6. housing :</b> has housing loan? (categorical: “no”, “yes”, “unknown”)
# <br>
# 
# <b>7. loan :</b> has personal loan? (categorical: “no”, “yes”, “unknown”)
# <br>
# 
# <b>8. contact :</b> contact communication type (categorical: “cellular”, “telephone”)
# <br>
# 
# <b>9. month :</b> last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
# <br>
# 
# <b>10. day_of_week :</b> last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
# <br>
# 
# <b>11. duration :</b> last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
# <br>
# 
# <b>12. campaign :</b> number of contacts performed during this campaign and for this client (numeric, includes last contact)
# <br>
# 
# <b>13. pdays :</b> number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# <br>
# 
# <b>14. previous :</b> number of contacts performed before this campaign and for this client (numeric)
# <br>
# 
# <b>15. poutcome :</b> outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)
# <br>
# 
# <b>16. emp.var.rate :</b> employment variation rate — (numeric)
# <br>
# 
# <b>17. cons.price.idx :</b> consumer price index — (numeric)
# <br>
# 
# <b>18. cons.conf.idx :</b> consumer confidence index — (numeric)
# <br>
# 
# <b>19. euribor3m :</b> euribor 3 month rate — (numeric)
# <br>
# 
# <b>20. nr.employed :</b> number of employees — (numeric)
# <br>

# # Target / Dependent Variable :

# <b>1. y : </b> has the client subscribed a <b>term deposit ? </b> (binary: <b>“1” means “Yes”, “0” means “No”</b> )

# # 1.1 Setting Location of DataSet (CSV File) :

# In[1]:


# Setting Location of Dataset :

dataset_location = "D:/Hackathon FS-ADM/bank/"

result_location = "D:/Hackathon FS-ADM/Final_Outcome/"


# In[2]:


# Scientific and Data Manipulation Libraries :

import pandas as pd
import numpy as np

# Visualization Libraries:

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Classification Model Libraries :

from sklearn.svm                   import SVC
from sklearn.naive_bayes           import GaussianNB
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.linear_model          import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Pre-Processing Libraries:

from sklearn.preprocessing         import LabelEncoder, StandardScaler, MinMaxScaler

# Metrics Libraries:

from sklearn                       import metrics as m
from sklearn.model_selection       import ShuffleSplit, StratifiedShuffleSplit, cross_val_score, train_test_split, GridSearchCV

# import statsmodels.formula.api as smf
# from sklearn.metrics             import roc_curve , accuracy_score , precision_score , recall_score , roc_auc_score , f1_score


# # 1.2 Importing CSV File into Pandas DataFrame :

# In[3]:


# Reading CSV File into Pandas DataFrame using ";" as Seperator :

data = pd.read_csv(dataset_location+'bank-full.csv',sep=";",header=0)

# View the First 5 rows of DataFrame :

data.head()

print(type(data))


# In[4]:


# View the Last 5 rows of DataFrame :

data.tail()


# # 2.1 Extracting Information on Columns :

# In[5]:


# Prints Information of All Columns :

data.info()
# or 
# data.info(verbose=True) 

# Prints a Summary of Columns Count and its dtypes but not per column Information :

# data.info(verbose=False)


# # 2.2 Extracting Statistical Information on Numerical Columns :

# In[6]:


# Shows Descriptive Statistics Values on Numerical Value based Features :

data.describe()


# # 2.3 Finding Correlation between Features and Class for Selection :

# ### 1. Using PairPlot :

# In[7]:


sns.pairplot(data)


# ### 2. Correlation Matrix :

# In[8]:


data.corr()


# ### 3. Heatpot to Visualise Correlation

# In[9]:


sns.heatmap(data.corr())


# # <b>As per Pair Plot, Correlation Matrix and Heatmap</b> 
# 
# <b>Observations are as follows :</b> 
# <br>
# 
# 1. Data is non-linear and asymmetric
# <br>
# 
# 2. Hence selection of features will not depend upon correlation factor.
# <br>
# 
# 3. Also not a single feature is correlated completely with class, hence requires combinantion of features.
# 
# # <b>Feature Selection techniques : </b>
# 
# 1. <b> Univariate Selection (Non-Negative features) </b>
# <br>
# 
# 2. <b> Recursive Feature Elimination (RFE) </b>
# <br>
# 
# 3. <b> Principal Component Analysis (PCA) (Dimensionality Reduction Technique) </b>
# <br>
# 
# 4. <b> Feature Importance (Decision Trees Technique) </b>
# 
# # <b>Which feature selection technique should be used for our data?</b>
# 1. Contains negative values, hence Univariate Selection technique cannot be used.
# <br>
# 
# 2. PCA is data reduction technique. <b>Our Aim is to select best possible feature and not reduction</b> and this is classification type of data. PCA is also an unsupervised method, used for dimensionality reduction.
# <br>
# 
# 3. Hence <b>Feature Importance (Decision Tree Technique)</b> and <b>Recursive Feature Elimination (RFE)</b> can be used for feature selection.
# <br>
# 
# 4. Best possible technique will be which extracts columns who provide better accuracy.
# <br>

# # 3. Exploring Predictor Variables / Features :

# In[11]:


# View the (Rows,Columns) in DataFrame :

data.shape


# ### 3.1 Find and Impute Missing Values : 

# In[12]:


# Sum of Missing Values in Each Column :

data.isnull().sum()

# NOTE : Since there are No Missing Values in any of the Columns, Imputation is not needed.


# # 4. Finding Duplicating using Unique and Value Counts :

# In[13]:


data.head(2)


# In[14]:


# Unique Education Values :

print(data['education'].unique())
print(data['education'].nunique())


# In[15]:


# Cross Tab to display Education stats with respect to y (ie) Target variable :

pd.crosstab(index=data["education"], columns=data["y"])


# In[16]:


# Education Categories and Frequency :

data.education.value_counts().plot(kind="barh")


# In[17]:


# Barplot for the Predictor / Independent Variable - job : 

sns.countplot(y="job", data=data)
plt.show()


# In[18]:


# Barplot for the Predictor / Independent Variable - marital : 

sns.countplot(x="marital", data=data)
plt.show()


# In[19]:


# Barplot for the Predictor / Independent Variable - default : 

sns.countplot(x="default", data=data)
plt.show()


# In[20]:


# Barplot for the Predictor / Independent Variable - housing : 

sns.countplot(x="housing", data=data)
plt.show()


# In[21]:


# Barplot for the Predictor / Independent Variable - loan : 

sns.countplot(x="loan", data=data)
plt.show()


# In[22]:


# Barplot for the Predictor / Independent Variable - poutcome : 

sns.countplot(x="poutcome", data=data)
plt.show()


# # 5. Exploring Target Varaible :

# In[10]:


# Barplot for the Target / Dependent Variable :

sns.countplot(x='y',data=data, palette='hls')
plt.show()


# # Assumption 1 :
# 
# - <b>Our prediction will be based on the customer’s job, marital status, whether he(she) has credit in default, whether he(she) has a housing loan, whether he(she) has a personal loan, and the outcome of the previous marketing campaigns. So, we will drop the variables that we do not need.</b>

# In[23]:


data.head(2)


# In[24]:


# Removing 
# 0 - age
# 3 - education
# 8 - contact
# 9 - day
# 10 - month
# 11 - duration
# 12 - campaign
# 13 - plays
data_new_2 = data.copy(deep=True)
data.drop(data.columns[[0,3,8,9,10,11,12,13]],axis=1,inplace=True)


# In[25]:


data.head(2)


# # 6. Exploring Categorical and Numerical Data into Digits Form :

# ### 6.1 Converting Object Type to Integer using One-Hot Encoding :

# In[26]:


# Fetching Data Type of all Columns :

data.dtypes


# In[27]:


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


# In[28]:


# Coverting Target Variable / Column into Binary Format :

data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)


# In[29]:


# Successfully converted Object data into  Integer data types

data_new.dtypes


# ### 6.2 Classifiers : Based on the values of different parameters we can conclude to the following classifiers for Binary Classification.
# 
# <br>
# 1. AdaBoosting
# 2. Linear Discriminant Analysis
# 3. Logistics Regression
# 4. Random Forest Classifier
# 5. K Nearest Neighbour
# 6. Decision Tree
# 7. Gaussian Naive Bayes
# 8. Support Vector Classifier 
# 9. Gradient Boosting

# In[30]:


# Creating Dictionary with Classifiers :

classifiers = {
               '1. Gradient Boosting Classifier':GradientBoostingClassifier(random_state=5),
               '2. Adaptive Boosting Classifier':AdaBoostClassifier(learning_rate=.9,n_estimators=40),#BEST
               '3. Linear Discriminant Analysis':LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.2),  
               '4. K Nearest Neighbour':KNeighborsClassifier(n_neighbors=128),
               '5. Logistic Regression':LogisticRegression(random_state=5),
               '6. Decision Tree Classifier':DecisionTreeClassifier(class_weight="balanced",max_depth=6,max_leaf_nodes=20,presort =True,random_state=10),
               '7. Random Forest Classifier':RandomForestClassifier(bootstrap=False,class_weight="balanced",max_depth=6,max_leaf_nodes=20,n_estimators=100,random_state=10),
#              '8. Gaussian Naive Bayes Classifier':GaussianNB(),# NO CHANGE
#              '9. Support Vector Classifier':SVC(kernel='rbf'),                              
               

               }
print(classifiers)


# In[31]:


# View the (Rows,Columns) in DataFrame :
# Due to One Hot Encoding Increase in the Number of Columns :

data_new.shape


# # 7. Splitting Predictor and Target Variables into X and y : 

# In[32]:


# Seperating Predictor and Target Columns into X and y Respectively :

data_X = data_new.drop(['y'], axis=1)
data_y = pd.DataFrame(data_new['y'])

print(data_X.head())
print(data_y.head())


# In[33]:


# Log Columns Headings :

log_cols = ["Classifier", "Accuracy","Precision","Recall","F1","ROC_AUC","CV_2_Fold","CV_5_Fold","CV_10_Fold","CV_20_Fold"]
# 
log = pd.DataFrame(columns=log_cols)

# Metric Columns Headings :

# metrics_cols = ["Precision Score","Recall Score","F1-Score","roc-ROC_AUC_Score"]
# metric = pd.DataFrame(columns=metrics_cols)


# # 8. Splitting Training and Test Data and Applying Classification Models :

# # 9. Standardizing, Fitting, Predicting and Scoring the Data using all Classifiers along with Confusion Matrix :

# In[34]:


import warnings
warnings.filterwarnings('ignore')

rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3,random_state=2)
rs.get_n_splits(data_X,data_y)

for Name,classify in classifiers.items():
    for train_index, test_index in rs.split(data_X,data_y):
        
        print(Name)
        
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        # Splitting Training and Testing Data :
        
        X_train,X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y_train,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
                              
        # Standardizing Features :
        
        sc_X    = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test  = sc_X.transform(X_test)
        
        # Normalizing Features :
        
#         Mn_Mx_X = MinMaxScaler()
#         X_train = Mn_Mx_X.fit_transform(X_train)
#         X_test =  Mn_Mx_X.transform(X_test)
        
        # Fitting of Train Data and Predicting with Test Data :
        
        cls   = classify
        cls   = cls.fit(X_train,y_train)
        y_out = cls.predict(X_test)
        
        # Calculating Accuracy, Precision, Recall, ROC_AUC and F1 Scores :
        
        accuracy  = m.accuracy_score(y_test,y_out)
        precision = m.precision_score(y_test,y_out,average='macro')
        recall    = m.recall_score(y_test,y_out,average='macro')
        roc_auc   = m.roc_auc_score(y_out,y_test)
        f1_score  = m.f1_score(y_test,y_out,average='macro')
        
        # Calculating Cross-Validation AUC Score :
        
        print(y_train.shape)
        print(y_train['y'].shape)
        
        cross_val_score_2_fold  = cross_val_score(classify, X_train, y_train['y'], cv=2, scoring='roc_auc').mean()
        cross_val_score_5_fold  = cross_val_score(classify, X_train, y_train['y'], cv=5, scoring='roc_auc').mean()
        cross_val_score_10_fold = cross_val_score(classify, X_train, y_train['y'], cv=10, scoring='roc_auc').mean()
        cross_val_score_20_fold = cross_val_score(classify, X_train, y_train['y'], cv=20, scoring='roc_auc').mean()
                
        # Classification Report for All Classification Models :
        
        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc,cross_val_score_2_fold,cross_val_score_5_fold,cross_val_score_10_fold,cross_val_score_20_fold]], columns=log_cols)
        # 
        log       = log.append(log_entry)
        
        # Creating DataFrame with  2 Columns , Replacing Numbers with Category, Saving Target Prediction as CSV File with Index :
                
        df = pd.DataFrame({'Index': y_test.index.tolist(), 'Term_Deposit': y_out})        
        df['Term_Deposit'].replace((1, 0), ('yes', 'no'), inplace=True)        
        df.to_csv(result_location + "Term_Deposit_"+Name+".csv", header=['Index','Term_Deposit'],index=0)
        
        # Plotting ROC-AUC using True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) :  
        
        if(Name!="8. Support Vector Classifier"):
            
            y_pred_proba        = classify.predict_proba(X_test)[::,1]
            fpr, tpr, threshold = m.roc_curve(y_test,  y_pred_proba)

            # First argument is True values, second argument is Predicted Probabilities :

            auc = m.roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr,tpr,label="data 1, AUC="+str(auc))
            plt.legend(loc=4)
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.rcParams['font.size'] = 12        
            plt.plot([0, 1], [0, 1], color='blue', linestyle='--')        
            plt.show()          
                
# Scroll complete output to view all the accuracy scores and bar graph.


# In[35]:


log


# # 10. Performance Metric using Precision and Recall Calculation along with roc_auc_score & accuracy_score :

# ### 1. Accuracy Score Comparison for All Classification Models :

# In[36]:


# Accuracy Score Comparison :

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="r")  
plt.show()


# ### 2. Precision Score Comparison for All Classification Models :

# In[37]:


# Precision Score Comparison :

plt.xlabel('Precision')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Precision', y='Classifier', data=log, color="g")  
plt.show()


# ### 3. Recall Score Comparison for All Classification Models :

# In[38]:


# Recall Score Comparison :

plt.xlabel('Recall')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Recall', y='Classifier', data=log, color="b")  
plt.show()


# ### 4. ROC_AUC Score Comparison for All Classification Models :

# In[39]:


# ROC_AUC Score Comparison :

plt.xlabel('ROC_AUC')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='ROC_AUC', y='Classifier', data=log, color="m")  
plt.show()


# ### 5. F1 Score Comparison for All Classification Models :

# In[40]:


# F1 Score Comparison :

plt.xlabel('F1')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='F1', y='Classifier', data=log, color="y")  
plt.show()


# ### 6. CV_2_Fold Comparison for All Classification Models :

# In[41]:


# CV_2_Fold Comparison :

plt.xlabel('CV_2_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_2_Fold', y='Classifier', data=log, color="r")  
plt.show()


# ### 7. CV_5_Fold Comparison for All Classification Models :

# In[42]:


# CV_5_Fold Comparison :

plt.xlabel('CV_5_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_5_Fold', y='Classifier', data=log, color="g")  
plt.show()


# ### 8. CV_10_Fold Comparison for All Classification Models :

# In[43]:


# CV_10_Fold Comparison :

plt.xlabel('CV_10_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_10_Fold', y='Classifier', data=log, color="m")  
plt.show()


# ### 9. CV_20_Fold Comparison for All Classification Models :

# In[44]:


# CV_20_Fold Comparison :

plt.xlabel('CV_20_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_20_Fold', y='Classifier', data=log, color="b")  
plt.show()


# # Assumption 2 :
# 
# - <b>Our prediction will be based on the customer’s age, education, job, marital status, whether he(she) has credit in default, whether he(she) has a housing loan, whether he(she) has a personal loan, and the outcome of the previous marketing campaigns. So, we will drop the variables that we do not need.</b>

# In[45]:


data_new_2.head(2)


# In[46]:


# Removing 
# 8 - contact
# 9 - day
# 10 - month
# 11 - duration
# 12 - campaign
# 13 - plays

data_new_2.drop(data_new_2.columns[[8,9,10,11,12,13]],axis=1,inplace=True)


# In[47]:


data_new_2.head(2)


# In[48]:


# Fetching Data Type of all Columns :

data_new_2.dtypes


# In[49]:


# Creating Dummies for Categorical Variables :

data_dummy = pd.get_dummies(data_new_2, columns=['job','marital',
                                         'default',
                                         'housing','loan',
                                         'poutcome','education'])

# data_new = pd.get_dummies(data, columns=['job','marital',
#                                          'education','default',
#                                          'housing','loan',
#                                          'contact','month',
#                                          'poutcome'])


# In[50]:


# Coverting Target Variable / Column into Binary Format :

data_dummy.y.replace(('yes', 'no'), (1, 0), inplace=True)


# In[51]:


# Successfully converted Object data into  Integer data types

data_dummy.dtypes


# In[52]:


# View the (Rows,Columns) in DataFrame :
# Due to One Hot Encoding Increase in the Number of Columns :

data_dummy.shape


# In[53]:


# Seperating Predictor and Target Columns into X and y Respectively :

data_X = data_dummy.drop(['y'], axis=1)
data_y = pd.DataFrame(data_dummy['y'])

print(data_X.head())
print(data_y.head())


# In[54]:


# Log Columns Headings :

log_cols = ["Classifier", "Accuracy","Precision","Recall","F1","ROC_AUC","CV_2_Fold","CV_5_Fold","CV_10_Fold","CV_20_Fold"]
# 
log = pd.DataFrame(columns=log_cols)

# Metric Columns Headings :

# metrics_cols = ["Precision Score","Recall Score","F1-Score","roc-ROC_AUC_Score"]
# metric = pd.DataFrame(columns=metrics_cols)


# In[55]:


# Creating Dictionary with Classifiers :

classifiers_new = {
                '1. Gradient Boosting Classifier':GradientBoostingClassifier(random_state=5),
               '2. Adaptive Boosting Classifier':AdaBoostClassifier(learning_rate=.9,n_estimators=40),#BEST
               '3. Linear Discriminant Analysis':LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.2),  
               '4. K Nearest Neighbour':KNeighborsClassifier(n_neighbors=128),
               '5. Logistic Regression':LogisticRegression(random_state=5),
               '6. Decision Tree Classifier':DecisionTreeClassifier(class_weight="balanced",max_depth=6,max_leaf_nodes=20,presort =True,random_state=10),
               '7. Random Forest Classifier':RandomForestClassifier(bootstrap=False,class_weight="balanced",max_depth=6,max_leaf_nodes=20,n_estimators=100,random_state=10),
#                '7. Gaussian Naive Bayes Classifier':GaussianNB(),# NO CHANGE
#                '8. Support Vector Classifier':SVC(kernel='rbf'),                              

#                '3.1 Logistic Regression':LogisticRegression(class_weight"balanced",solver='lbfgs',random_state=5),                
#                '3.3 Logistic Regression':LogisticRegression(),                 
#                '5.1 K Nearest Neighbour':KNeighborsClassifier(),    
#                '5.3 K Nearest Neighbour':KNeighborsClassifier(algorithm='kd_tree'), 
#                '5.4 K Nearest Neighbour':KNeighborsClassifier(algorithm='brute'),           
#                '6.2 Decision Tree Classifier 6':DecisionTreeClassifier(random_state=1,max_depth=6),
 
               }
print(classifiers_new)


# # Performance Metric using Precision and Recall Calculation along with roc_auc_score & accuracy_score :

# In[56]:


import warnings
warnings.filterwarnings('ignore')

rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3,random_state=2)
rs.get_n_splits(data_X,data_y)

for Name,classify in classifiers_new.items():
    for train_index, test_index in rs.split(data_X,data_y):
        
        print(Name)
        
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        # Splitting Training and Testing Data :
        
        X_train,X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y_train,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
                              
        # Standardizing Features :
        
        sc_X    = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test  = sc_X.transform(X_test)
        
        # Normalizing Features :
        
#         Mn_Mx_X = MinMaxScaler()
#         X_train = Mn_Mx_X.fit_transform(X_train)
#         X_test =  Mn_Mx_X.transform(X_test)
        
        # Fitting of Train Data and Predicting with Test Data :
        
        cls   = classify
        cls   = cls.fit(X_train,y_train)
        y_out = cls.predict(X_test)
        
        # Calculating Accuracy, Precision, Recall, ROC_AUC and F1 Scores :
        
        accuracy  = m.accuracy_score(y_test,y_out)
        precision = m.precision_score(y_test,y_out,average='macro')
        recall    = m.recall_score(y_test,y_out,average='macro')
        roc_auc   = m.roc_auc_score(y_out,y_test)
        f1_score  = m.f1_score(y_test,y_out,average='macro')
        
        # Calculating Cross-Validation AUC Score :
        
        print(y_train.shape)
        print(y_train['y'].shape)
        
        cross_val_score_2_fold  = cross_val_score(classify, X_train, y_train['y'], cv=2, scoring='roc_auc').mean()
        cross_val_score_5_fold  = cross_val_score(classify, X_train, y_train['y'], cv=5, scoring='roc_auc').mean()
        cross_val_score_10_fold = cross_val_score(classify, X_train, y_train['y'], cv=10, scoring='roc_auc').mean()
        cross_val_score_20_fold = cross_val_score(classify, X_train, y_train['y'], cv=20, scoring='roc_auc').mean()
                
        # Classification Report for All Classification Models :
        
        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc,cross_val_score_2_fold,cross_val_score_5_fold,cross_val_score_10_fold,cross_val_score_20_fold]], columns=log_cols)
        # 
        log       = log.append(log_entry)
        
        # Creating DataFrame with  2 Columns , Replacing Numbers with Category, Saving Target Prediction as CSV File with Index :
                
        df = pd.DataFrame({'Index': y_test.index.tolist(), 'Term_Deposit': y_out})        
        df['Term_Deposit'].replace((1, 0), ('yes', 'no'), inplace=True)        
        df.to_csv(result_location + "Term_Deposit_"+Name+".csv", header=['Index','Term_Deposit'],index=0)
        
        # Plotting ROC-AUC using True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) :  
        
        if(Name!="8. Support Vector Classifier"):
            
            y_pred_proba        = classify.predict_proba(X_test)[::,1]
            fpr, tpr, threshold = m.roc_curve(y_test,  y_pred_proba)

            # First argument is True values, second argument is Predicted Probabilities :

            auc = m.roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr,tpr,label="data 1, AUC="+str(auc))
            plt.legend(loc=4)
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.rcParams['font.size'] = 12        
            plt.plot([0, 1], [0, 1], color='blue', linestyle='--')        
            plt.show()          
                
# Scroll complete output to view all the accuracy scores and bar graph.


# In[57]:


log


# ### 1. Accuracy Score Comparison for All Classification Models :

# In[58]:


# Accuracy Score Comparison :

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="r")  
plt.show()


# ### 2. Precision Score Comparison for All Classification Models :

# In[59]:


# Precision Score Comparison :

plt.xlabel('Precision')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Precision', y='Classifier', data=log, color="g")  
plt.show()


# ### 3. Recall Score Comparison for All Classification Models :

# In[60]:


# Recall Score Comparison :

plt.xlabel('Recall')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Recall', y='Classifier', data=log, color="b")  
plt.show()


# ### 4. ROC_AUC Score Comparison for All Classification Models :

# In[61]:


# ROC_AUC Score Comparison :

plt.xlabel('ROC_AUC')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='ROC_AUC', y='Classifier', data=log, color="m")  
plt.show()


# ### 5. F1 Score Comparison for All Classification Models :

# In[62]:


# F1 Score Comparison :

plt.xlabel('F1')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='F1', y='Classifier', data=log, color="y")  
plt.show()


# ### 6. CV_2_Fold Comparison for All Classification Models :

# In[63]:


# CV_2_Fold Comparison :

plt.xlabel('CV_2_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_2_Fold', y='Classifier', data=log, color="r")  
plt.show()


# ### 7. CV_5_Fold Comparison for All Classification Models :

# In[64]:


# CV_5_Fold Comparison :

plt.xlabel('CV_5_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_5_Fold', y='Classifier', data=log, color="g")  
plt.show()


# ### 8. CV_10_Fold Comparison for All Classification Models :

# In[65]:


# CV_10_Fold Comparison :

plt.xlabel('CV_10_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_10_Fold', y='Classifier', data=log, color="m")  
plt.show()


# ### 9. CV_20_Fold Comparison for All Classification Models :

# In[66]:


# CV_20_Fold Comparison :

plt.xlabel('CV_20_Fold')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='CV_20_Fold', y='Classifier', data=log, color="b")  
plt.show()

