# %% [markdown]
# # Importing the libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import mode
import seaborn as sns
import plotly.express as px
# import pingouin as pg
import plotly.figure_factory as ff
import scipy
from scipy import stats
from scipy.stats import t, ttest_1samp, ttest_ind
import math
import pickle

from numpy import mean, std

import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format

import sklearn
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification, load_iris
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import yellowbrick
from yellowbrick.model_selection import CVScores, FeatureImportances
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.classifier import ROCAUC

# %%
iris = datasets.load_iris()

# %% [markdown]
# # Importing the dataset

# %%
survey = pd.read_csv('survey.csv')
survey.head()

# %% [markdown]
# # Exploratory Data Analysis

# %%
survey.sample(5)

# %%
survey.shape

# %%
survey.dtypes

# %%
survey.select_dtypes(exclude='object').agg(['mean','median','min','max','std','kurt'])

# %%
survey.head()

# %% [markdown]
# # Data cleaning

# %% [markdown]
# ## Duplicates

# %%
survey.duplicated().sum()

# %% [markdown]
# No duplicates found

# %% [markdown]
# ## Missing values

# %%
survey.shape

# %%
survey.isna().sum()

# %% [markdown]
# ### Dropping the entire comments and state column for too many missing values

# %% [markdown]
# The comments and state columns will need to be dropped entirely as they contain to many missing values and cannot be filled with specific data

# %%
survey.drop(['comments','state'], axis=1, inplace=True)

# %%
survey.isna().sum()

# %%
survey.shape

# %% [markdown]
# ### Drop missing values

# %%
survey.dropna(inplace=True)

# %%
survey.isna().sum()

# %%
survey.shape

# %% [markdown]
# ## Structural errors

# %%
pd.to_datetime(survey.Timestamp)

# %%
cat_df = survey.select_dtypes(include='object')
cat_df.head()

# %%
def value_counter():
    
    for col in cat_df:
        freq = cat_df[col].value_counts()
        
        print(col)
        print('_' * 40)
        print(freq)
        print('_' * 40)
        print('_' * 40)
value_counter()

# %% [markdown]
# ### Replacing categories in Gender column to only have Male, Female and Other

# %%
survey.Gender.replace(['male','M','m','make','Make','Man','Male ', 'Cis Male','Mail','msle','something kinda male?','Malr','Mal','ostensibly male, unsure what that really means',
                       'cis male','maile','Guy (-ish) ^_^','Guy','Male-ish','Cis Man','male leaning androgynous','Male (CIS)'], 'Male',inplace=True)

survey.Gender.replace(['Female', 'Female ','female','F','f','Woman','Female (trans)','Female  ', 'Femake','femail','Cis Female','Trans-female','woman','Trans woman','cis-female/femme',
'Female (cis)'], 'Female',inplace=True)

survey.Gender.replace(['non-binary','p','Nah','Androgyne','queer','Neuter', 'queer/she/they','Genderqueer','All','A little about you','Agender','fluid','Enby'], 'Other',inplace=True)

# %%
survey.Gender.value_counts()

# %% [markdown]
# ## Outliers

# %%
survey.head()

# %%
num_df = survey.select_dtypes(exclude='object')
num_df.head()

# %%
num_df.mean()

# %%
num_df.plot.box()

# %%
survey.Age.unique()

# %% [markdown]
# There are extreme, unrealistic values in the Age column

# %%
#find Q1, Q3, and interquartile range for each column
Q1 = num_df.quantile(q=.25)
Q3 = num_df.quantile(q=.75)
IQR = num_df.apply(stats.iqr)

# %%
survey = survey[~((survey.select_dtypes(exclude='object') < (Q1-1.5*IQR)) | (survey.select_dtypes(exclude='object') > (Q3+1.5*IQR))).any(axis=1)]

# %%
num_df = survey.select_dtypes(exclude='object')
num_df.head()

# %%
num_df.plot.box()

# %%
num_df.agg(['min','max','mean','median','std','kurt'])

# %%
survey.shape

# %% [markdown]
# # Visualization (EDA)

# %%
survey.head()

# %% [markdown]
# ## Visualization of people seeking treatment by gender

# %%
survey.Gender.value_counts()

# %%
plt.figure(figsize=(15,7))
sns.countplot(data = survey, x = 'treatment', hue = 'Gender')
plt.title('Visualization of people seeking treatment by gender')
plt.grid()

# %% [markdown]
# The data has a majority of male respondents as the tech industry has more Males than Females at the time of analysis (2021).
# 
# In all gender categories, there are more respondents seeking treatment as oppossed to those who do not.
# 
# The number of males not seeking treatment is significantly higher than the other gender categories (Female and Other)

# %%
survey.head()

# %% [markdown]
# ## Visualization of people in self employment by gender

# %%
plt.figure(figsize=(15,7))
sns.countplot(data = survey, x = 'self_employed', hue = 'Gender')
plt.title('Visualization of people in self employment by gender')
plt.grid()

# %% [markdown]
# While most people are not self employed, a majority of those who are, are male.
# 
# This is likely due to majority of the respondents being male and the tech industry being mostly male dominated.

# %%
survey.head()

# %% [markdown]
# ## Visualization of Family history

# %%
plt.figure(figsize=(15,7))
sns.countplot(data = survey, x = 'family_history')
plt.title('Visualization of Family history')
plt.grid()

# %%
survey.head()

# %% [markdown]
# ## Visualization of Number of employees by Tech company

# %%
plt.figure(figsize=(15,7))
sns.countplot(data = survey, x = 'no_employees', hue = 'tech_company')
plt.title('Visualization of number of Number of employees by Tech company')
plt.xticks(rotation = 60)
plt.grid()

# %%
survey.head()

# %% [markdown]
# ## Visualization of number of Wellness program by Number of employees

# %%
plt.figure(figsize=(15,7))
sns.countplot(data = survey, x = 'wellness_program', hue = 'no_employees')
plt.title('Visualization of number of Wellness program by Number of employees')
plt.xticks(rotation = 60)
plt.grid()

# %% [markdown]
# ## Visualization of number of respondents by Country

# %%
plt.figure(figsize=(30,15))
sns.countplot(data = survey, x = 'Country')
plt.title('Visualization of number of respondents by Country')
plt.xticks(rotation = 60)
plt.grid()

# %% [markdown]
# An overwhelming majority of the respondents are from the US followed by the UK.
# 
# This is because a majority of world leading tech companies are based in the US eg Amazon, Facebook, Google etc.

# %% [markdown]
# ## Visualization of distribution of age

# %%
survey.Age.mean()

# %%
plt.figure(figsize=(15,7))
sns.distplot(survey.Age)
plt.title('Distribution of Age')
plt.grid()

# %%
survey.Age.agg(['skew', 'kurt'])

# %% [markdown]
# Age seems to be slightly positively skewed as the right tail is slightly longer than the left tail
# 
# It is also normally distributed based on the skewness as it falls between -0.5 and 0.5

# %%
plt.figure(figsize=(15,7))
sns.violinplot(data = survey, y= 'Age')
plt.title('Distribution of Age')
plt.grid()

# %%
survey.shape

# %%
survey.head()

# %% [markdown]
# # Machine Learning (Classification)

# %% [markdown]
# ## Preparation of data for Machine Learning

# %% [markdown]
# ### Dropping the Timestamp and Country column as they are almost insignificant to the task ahead

# %%
survey.head()

# %% [markdown]
# The Timestamp column will be dropped as it does affect the ML classification that will be done and has no significance
# 
# The Country column will also be dropped as the work people in tech do is really similar 

# %%
survey.drop(['Timestamp', 'Country'], axis=1, inplace=True)

# %%
survey.head()

# %%
survey.shape

# %% [markdown]
# ### Separation of the predictor and dependent variables

# %%
x = survey.drop('treatment', axis=1)
y = survey.treatment #response/dependent variable

print(x.shape)
x.head()

# %% [markdown]
# ### Encoding using LabelEncoder

# %% [markdown]
# Using Label encoder instead of dummies will make feature selection much easier

# %%
def encoder():
  le = LabelEncoder()
  new_cat_df = x.select_dtypes(include='object')

  for col in new_cat_df:
    x[col] = le.fit_transform(x[col])

  print(x.shape)
encoder()

# %%
x.sample(10)

# %%
survey.head()

# %% [markdown]
# ### Splitting the data into training and testing sets

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2500)

print('size of train predictors: {} and size of train labels: {}'.format(x_train.shape, y_train.shape))
print('size of test predictors: {} and size of test labels: {}'.format(x_test.shape, y_test.shape))

# %% [markdown]
# ## Building classification models

# %%
LogReg = LogisticRegression()
LogReg_fit = LogReg.fit(x_train, y_train)

KNN = KNeighborsClassifier()
KNN_fit = KNN.fit(x_train, y_train)

SVM = SVC()
SVM_fit = SVM.fit(x_train, y_train)

DecTree = DecisionTreeClassifier(max_depth=5)
DecTree_fit = DecTree.fit(x_train, y_train)

RF = RandomForestClassifier(max_depth=15,max_features=10,random_state=15)
RF_fit = RF.fit(x_train, y_train)

LDA = LinearDiscriminantAnalysis()
LDA_fit = LDA.fit(x_train, y_train)

GBC = GradientBoostingClassifier()
GBC_fit = GBC.fit(x_train, y_train)

ADA = AdaBoostClassifier()
ADA_fit = ADA.fit(x_train, y_train)

ETC = ExtraTreesClassifier()
ETC_fit = ETC.fit(x_train, y_train)

# %% [markdown]
# ## Performance of the models

# %%
score_df = pd.DataFrame({
    'models': ['LogReg', 'KNN', 'SVM', 'DecTree', 'RF', 'LDA', 'GBC', 'ADA', 'ETC'],

    'train_score': [LogReg_fit.score(x_train, y_train)*100,
                    KNN_fit.score(x_train, y_train)*100,
                    SVM_fit.score(x_train, y_train)*100,
                    DecTree_fit.score(x_train, y_train)*100,
                    RF_fit.score(x_train, y_train)*100,
                    LDA_fit.score(x_train, y_train)*100,
                    GBC_fit.score(x_train, y_train)*100,
                    ADA_fit.score(x_train, y_train)*100,
                    ETC_fit.score(x_train, y_train)*100],

    'test_score': [LogReg_fit.score(x_test, y_test)*100,
                   KNN_fit.score(x_test, y_test)*100,
                   SVM_fit.score(x_test, y_test)*100,
                   DecTree_fit.score(x_test, y_test)*100,
                   RF_fit.score(x_test, y_test)*100,
                   LDA_fit.score(x_test, y_test)*100,
                   GBC_fit.score(x_test, y_test)*100,
                   ADA_fit.score(x_test, y_test)*100,
                   ETC_fit.score(x_test, y_test)*100]
})

score_df

# %% [markdown]
# ## Crossvalidation

# %%
# prepare models
def crossValidater():
    
    models = []
    models.append(('LogReg', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('DecTree', DecisionTreeClassifier(max_depth=5)))
    models.append(('RF', RandomForestClassifier(max_depth=15,max_features=10,random_state=15)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('ADA', AdaBoostClassifier()))
    models.append(('ETC',ExtraTreesClassifier()))
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = '%s: %.3f%% (%.3f)' % (name, cv_results.mean()*100, cv_results.std())
        print(msg)
        print('\n')
    
crossValidater()

# %% [markdown]
# ## Feature selection

# %%
def feat_select():
  
    cv = KFold(n_splits=10)
    models = []
    models.append(LogisticRegression())
    # models.append(KNeighborsClassifier())
    # models.append(SVC())
    models.append(DecisionTreeClassifier())
    models.append(RandomForestClassifier())
    models.append(LinearDiscriminantAnalysis())
    models.append(GradientBoostingClassifier())
    models.append(AdaBoostClassifier())
    models.append(ExtraTreesClassifier())
    
    for model in models:
        plt.figure(figsize=(15,7))
        visualizer = FeatureImportances(model)
        visualizer.fit(x, y)        # Fit the data to the visualizer
        visualizer.show()
        
feat_select()

# %% [markdown]
# ## Scaling

# %%
scaled_x = survey.drop('treatment', axis=1)

print(scaled_x.shape)
scaled_x.head()

# %% [markdown]
# ### Encoding using LabelEncoder()

# %%
def encoder():
  le = LabelEncoder()
  new_cat_df = scaled_x.select_dtypes(include='object')

  for col in new_cat_df:
    scaled_x[col] = le.fit_transform(scaled_x[col])

  print(scaled_x.shape)
encoder()

# %% [markdown]
# ### Scaling with MinMaxScaler()

# %%
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(scaled_x)

X_scaled = pd.DataFrame(X_scaled, columns=scaled_x.columns)
X_scaled.head()

# %% [markdown]
# MinMaxScaler worked better a little better than StandardScaler and RobustScaler

# %% [markdown]
# ## Splitting scaled data into training and testing sets

# %%
X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=2500)

print('size of train predictors: {} and size of train labels: {}'.format(X_scaled_train.shape, y_train.shape))
print('size of test predictors: {} and size of test labels: {}'.format(X_scaled_test.shape, y_test.shape))

# %% [markdown]
# ## Rebuilding ML models using scaled data

# %%
LogReg = LogisticRegression()
LogReg_fit = LogReg.fit(X_scaled_train, y_train)

KNN = KNeighborsClassifier()
KNN_fit = KNN.fit(X_scaled_train, y_train)

SVM = SVC()
SVM_fit = SVM.fit(X_scaled_train, y_train)

DecTree = DecisionTreeClassifier(max_depth=5)
DecTree_fit = DecTree.fit(X_scaled_train, y_train)

RF = RandomForestClassifier(max_depth=15,max_features=10,random_state=15)
RF_fit = RF.fit(X_scaled_train, y_train)

LDA = LinearDiscriminantAnalysis()
LDA_fit = LDA.fit(X_scaled_train, y_train)

GBC = GradientBoostingClassifier()
GBC_fit = GBC.fit(X_scaled_train, y_train)

ADA = AdaBoostClassifier()
ADA_fit = ADA.fit(X_scaled_train, y_train)

ETC = ExtraTreesClassifier()
ETC_fit = ETC.fit(X_scaled_train, y_train)

# %% [markdown]
# ## Performance of the models

# %%
new_score_df = pd.DataFrame({
    'models': ['LogReg', 'KNN', 'SVM', 'DecTree', 'RF', 'LDA', 'GBC', 'ADA', 'ETC'],

    'train_score': [LogReg_fit.score(X_scaled_train, y_train)*100,
                    KNN_fit.score(X_scaled_train, y_train)*100,
                    SVM_fit.score(X_scaled_train, y_train)*100,
                    DecTree_fit.score(X_scaled_train, y_train)*100,
                    RF_fit.score(X_scaled_train, y_train)*100,
                    LDA_fit.score(X_scaled_train, y_train)*100,
                    GBC_fit.score(X_scaled_train, y_train)*100,
                    ADA_fit.score(X_scaled_train, y_train)*100,
                    ETC_fit.score(X_scaled_train, y_train)*100],

    'test_score': [LogReg_fit.score(X_scaled_test, y_test)*100,
                   KNN_fit.score(X_scaled_test, y_test)*100,
                   SVM_fit.score(X_scaled_test, y_test)*100,
                   DecTree_fit.score(X_scaled_test, y_test)*100,
                   RF_fit.score(X_scaled_test, y_test)*100,
                   LDA_fit.score(X_scaled_test, y_test)*100,
                   GBC_fit.score(X_scaled_test, y_test)*100,
                   ADA_fit.score(X_scaled_test, y_test)*100,
                   ETC_fit.score(X_scaled_test, y_test)*100]
})

new_score_df

# %% [markdown]
# ## Cross validation after scaling

# %%
# prepare models
def crossValidater():
    
    models = []
    models.append(('LogReg', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('DecTree', DecisionTreeClassifier(max_depth=5)))
    models.append(('RF', RandomForestClassifier(max_depth=15,max_features=10,random_state=15)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('ADA', AdaBoostClassifier()))
    models.append(('ETC',ExtraTreesClassifier()))
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_scaled, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = '%s: %.3f%% (%.3f)' % (name, cv_results.mean()*100, cv_results.std())
        print(msg)
        print('\n')
    
crossValidater()

# %% [markdown]
# Ada Boost Classifier is the most robust model used thus it will be tuned to see if there will be an impropvement in its performance

# %% [markdown]
# ## HyperParam Tuning using GridSearchCV() with RandomForestClassifier() as the base_estimator

# %%
# ABC = AdaBoostClassifier(base_estimator = RandomForestClassifier())

# param_grid = {'base_estimator__max_depth' : [int(x) for x in np.linspace(10, 30, 10)],
#               'base_estimator__min_samples_leaf' : [1, 2, 4, 6, 8],
#               'n_estimators' : [int(x) for x in np.linspace(10, 30, 10)],
#               'learning_rate' : [0.01, 0.1]}

# clf = GridSearchCV(ABC, param_grid, cv = 10, scoring = 'accuracy', n_jobs=-1)
# clf.fit(X_scaled_train,y_train)

# %%
# print(clf.best_params_)

# %%
# print(clf.best_score_ * 100)



# %%
Ada_boost = AdaBoostClassifier()
Ada_boost_fit = Ada_boost.fit(X_scaled_train, y_train)
# %%
pickle.dump(Ada_boost, open('model.pkl', 'wb'))
# %%
model = pickle.load(open('model.pkl', 'rb'))