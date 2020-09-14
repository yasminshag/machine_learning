#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


##Loading data train and test data from CSV file#

# In[2]:

###loading data from the csv files##
df_train = pd.read_csv('Trainv1.csv')
df_test = pd.read_csv('TestV1.csv')

###checking the number of rows columns in both test and train data sets###
df_train.shape, df_test.shape


##Functions to Clean Data#

# In[3]:

###function to replace nan values with mean value for int and float columns###
def nanCheckReplace(indata):
    
    num = ['int64', 'float64']
    
    for tkey in indata.columns:
        
        if (indata[tkey].dtypes in num) and (indata[tkey].isnull().sum() > 0 ):
            avg_value = indata[tkey].mean()
            indata[tkey].fillna(avg_value, inplace = True)
                           
    return indata

###function to convert string values to int values for computational purpose###
def convertStringtoInt(df,lkey,loc_map={}):
    labels=np.zeros(df[lkey].values.size,dtype=int)
    count = 0
    label_str=df[lkey].values
    for tt,tkey in enumerate(df.groupby(lkey).groups.keys()):
        ind=label_str==tkey
       
        if(tkey not in loc_map.keys()):
                loc_map[tkey]=tt
        labels[ind]=loc_map[tkey]
        print(tkey,tt)
        
    return labels, loc_map  

###function to calculate joining age using two dates###
def calc_age(startdate, joindate):
    startdate = pd.to_datetime(startdate)
    joindate = pd.to_datetime(joindate)
    age = joindate.dt.year - startdate.dt.year
    return age, startdate, joindate

###function to handle Null values in age column.replacing age with mean age by occupation or designation###
def ReplaceNanAge(df, ageData):
    if ageData == 'Applicants_joining_age':
        g = df.groupby('Applicant_Occupation').Applicants_joining_age.mean()
        groupbyData = 'Applicant_Occupation'
    elif ageData == 'Manager_joining_age':
        g = df.groupby('Manager_Joining_Designation').Manager_joining_age.mean()
        groupbyData = 'Manager_Joining_Designation'
    else:
        print('invalid call')
        return
    Occupation = g.index.values.tolist()
    Avg_age = g.values
    age_str = df[ageData]

    for index, rows in df.iterrows():
        for i in range(0,len(Occupation)):
            if rows[groupbyData] == Occupation[i] and np.isnan(rows[ageData]) :
                age_str[index] = Avg_age[i]

    df[ageData] = age_str
    return df[ageData]

###main function for data cleaning###
def cleanData(df,keymap={},typeobj='train'):
        
    list_column=['Applicant_Qualification','Manager_Current_Designation','Manager_Joining_Designation',
                'Applicant_Marital_Status','Applicant_Gender','Applicant_Occupation',
                'Manager_Status', 'Manager_Gender']
    
    for ll, lkey in enumerate(list_column):
        print('*** converting %s ****'%(lkey))
        if(typeobj=='train'):
            keymap[lkey]={}
        df[lkey], keymap[lkey] = convertStringtoInt(df,lkey,keymap[lkey])
    
   
    #adding new columns for the applicant and manager age at the time of joining
    df['Applicants_joining_age'],df['Applicant_BirthDate'],df['Application_Receipt_Date'] = calc_age(df['Applicant_BirthDate'], df['Application_Receipt_Date'])
    df['Manager_joining_age'],df['Manager_DoB'],df['Manager_DOJ'] = calc_age(df['Manager_DoB'], df['Manager_DOJ'])
        
    
    #Replacing null values in age data with the average age of applicant grouped by occupation#  
    age_value = ['Applicants_joining_age', 'Manager_joining_age']
    for i in age_value:
        if df[i].isnull().sum() > 0:
            df[i] = ReplaceNanAge(df, i)
            
        
    df = nanCheckReplace(df)
    df['Applicant_City_PIN'] = df['Applicant_City_PIN'].apply(np.int64)
            
    
    return df ,keymap

    


##function calls to clean and tranform Train and test data#

# In[4]:


training_data, keymap = cleanData(df_train,typeobj='train')

test_data, _ = cleanData(df_test,typeobj='test',keymap=keymap)
test_data['Business_Sourced'] = np.zeros(test_data.index.size, dtype = int)


##Split and Train Function#

# In[5]:


###function to select data for training the model###
def select_data_label(indata):
    data_list=[]
    reject_list=['Business_Sourced','ID','Office_PIN','Application_Receipt_Date',
                 'Applicant_BirthDate','Manager_DOJ','Manager_DoB']
    
    for tkey in indata.columns:
        if(tkey not in reject_list):
            data_list.append(tkey)
            
    return indata[data_list]

###function to split the data into train and test data####
def train_test_split(df, ycol, xcols, split_ratio):
    
    mask =  np.random.rand(len(df)) < split_ratio
    
    df_train = df[mask]
    df_test = df[~mask]  
    
    ytrain = df_train[ycol].values
    ytest = df_test[ycol].values
    xtrain = df_train[xcols].values
    xtest = df_test[xcols].values
    
    return df_train, df_test, xtrain, ytrain, xtest, ytest

   
###function to train the model for prediction
def classifyTrain(X_train, Y_train, X_test, Y_test):
    sel_classifier=["Random Forest", "Gradient Boosting Classifier", "Logistic Regression","Nearest Neighbors",
                  "Linear SVM", "Decision Tree", "Neural Net", "Naive Bayes"
                   ]

    dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=500, max_depth=9),
    "Decision Tree": tree.DecisionTreeClassifier(max_depth =9),
    "Random Forest": RandomForestClassifier(n_estimators=100,max_samples=9),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB()
    } 

    model_results = pd.DataFrame(columns = ['model','prediction', 'train_score', 'test_score', 'auc'])

    for classifier_name, classifier in list(dict_classifiers.items()):
        if(classifier_name not in sel_classifier):
            continue
        
        ###training the model
        classifier.fit(X_train, Y_train)
        
        ###predicting the Y_test values
        prediction = classifier.predict(X_test)

        ###calculating the model score 
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        auc=metrics.roc_auc_score(Y_test, prediction)
        
        ###saving the score in the model_results dataframe
        model_results = model_results.append({'model':classifier_name,'prediction': prediction , 'train_score': train_score, 'test_score': test_score, 'auc': auc} , ignore_index=True)
        
    
    return model_results, dict_classifiers


##calling functions on train dataset for Data selection and training the model#

# In[6]:


###calling functions for selecting data to train the model ####
ycol = 'Business_Sourced'
xcols = list(select_data_label(training_data).columns.values)

###calling function to split the selected data for training###
df_train, df_test, X_train, Y_train, X_test, Y_test = train_test_split(training_data, ycol, xcols, 0.9)

###calling the function to train the model
df_result, dict_class = classifyTrain(X_train, Y_train, X_test, Y_test)

df_result


##predicting the target data#

# In[7]:


###predicting the values using best performing classifier
for idx in range(len(df_result)):
    if df_result.loc[idx,"auc"] == df_result["auc"].max():
        selected_model = df_result.loc[idx,"model"]
        
print("Model used for prediction is: ", selected_model)

###selecting test data for prediction
X1_test = select_data_label(test_data)

###applying the selected model using test data for prediction
test_data['Business_Sourced'] = dict_class[ selected_model].predict(X1_test)   

test_data.to_csv('out.csv',columns = ['ID', 'Business_Sourced'], header = ['ID' ,'Business_Sourced'], index = False )

