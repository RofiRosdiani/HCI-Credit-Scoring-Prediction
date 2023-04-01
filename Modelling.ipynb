# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:01:15 2023

@author: admin
"""

# 1. Importing necessary libraries and packages and reading files
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()

from plotly import __version__
%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.io as pio
pio.renderers.default='svg'

from kaleido.scopes.plotly import PlotlyScope
import plotly.graph_objects as go
scope = PlotlyScope(
    plotlyjs="https://cdn.plot.ly/plotly-latest.min.js",
    # plotlyjs="/path/to/local/plotly.js",
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# print(__version__)

application_train = pd.read_csv("application_train.csv")


application_train.info(10)

# Melihat feature apa saja yang memiliki missing value lebih dari 50%
missing_values = pd.DataFrame(application_train.isnull().sum()/application_train.shape[0])
missing_values = missing_values[missing_values.iloc[:,0] > 0.50]
missing_values.sort_values([0], ascending=False)

# Drop feature tersebut
application_train.dropna(thresh = application_train.shape[0]*0.5, how='all', axis=1, inplace=True)


# Pengecheckan ulang apakah feature tersebut berhasil di drop
missing_values = pd.DataFrame(application_train.isnull().sum()/application_train.shape[0])
missing_values = missing_values[missing_values.iloc[:,0] > 0.50]
missing_values.sort_values([0], ascending=False)

#List of non-numerical variables
total_categorical_var = application_train.select_dtypes(include=['O']).columns


# We cannot have non-numerical columns for modelling. We can have only numerical columns. Non-numerical columns can also be ordinal or categorical variables.  
col_for_dummies = application_train.select_dtypes(include=['O']).columns.drop(['FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE'])
application_train_dummies = pd.get_dummies(application_train, columns = col_for_dummies, drop_first = True)

application_train_dummies.select_dtypes(include=['O']).columns

application_train_dummies['EMERGENCYSTATE_MODE'].value_counts()

#We cannot convert flag_own_car and flag_own_realty to column with yes or no etc. Lets rather map yes to 1 and no to 0
application_train_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})
application_train_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})
application_train_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})

print(application_train_dummies.shape)

application_train_dummies.columns

# Melihat feature apa saja yang memiliki missing value lebih dari 30%
missing_values1 = pd.DataFrame(application_train_dummies.isnull().sum()/application_train_dummies.shape[0])
missing_values1 = missing_values1[missing_values1.iloc[:,0] >= 0.30]
missing_values1.sort_values([0], ascending=False)

# Mengubah index menjadi columns
missing_values1.reset_index(inplace=True)
missing_values1 = missing_values1.rename(columns = {'index':'variable'})

# Menampilkan isi column
missing_values1['variable'].values

# Drop feature tersebut
application_train_dummies = application_train_dummies.drop(['YEARS_BEGINEXPLUATATION_AVG', 'FLOORSMAX_AVG',
       'YEARS_BEGINEXPLUATATION_MODE', 'FLOORSMAX_MODE',
       'YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_MEDI', 'TOTALAREA_MODE',
       'EMERGENCYSTATE_MODE'], axis=1)

application_train_dummies.isnull().sum()

#Replace NaN with 0
application_train_dummies = application_train_dummies.replace(np.nan, 0)


# Data Splitting
from sklearn.model_selection import train_test_split

# Membagi data menjadi 80/20 dengan menyamakan distribusi dari bad loans di test set dengan train set.
X = application_train_dummies.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application_train_dummies['TARGET']

# Scalling the features
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X= scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state=42)
y_train.value_counts(normalize=True)

# Balancing the data
from imblearn.over_sampling import SMOTE
from collections import Counter

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE= SMOTE()

# fit and apply the transform 
X_train,y_train= SMOTE.fit_resample(X_train,y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train))

# Distribusi y_test sudah sama persis dengan y_train
y_test.value_counts(normalize=True)

X_train.shape

# Dapat dilakukan print untuk semua unique values kolom, sehingga dapat di cek satu-satu
# unique values apa saja yang kotor.

for col in X_train.select_dtypes(include= ['object','bool']).columns:
    print(col)
    print(X_train[col].unique())
    print()

# Modelling
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

result = pd.DataFrame(list(zip(y_pred,y_test)), columns = ['y_pred', 'y_test'])
result.head()

#Untuk melihat akurasi, cara 1
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Untuk melihat akurasi, cara 2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, plot_confusion_matrix, plot_precision_recall_curve

print("The accuracy of logit model is:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# The accuracy of logit model is: 0.6822431426109296
#               precision    recall  f1-score   support

#            0       0.96      0.69      0.80     56538
#            1       0.15      0.65      0.25      4965

#     accuracy                           0.68     61503
#    macro avg       0.55      0.67      0.52     61503
# weighted avg       0.89      0.68      0.75     61503

#Untuk menampilkan confussion matrix
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('Predicted Label')
plt.ylabel('The Label')

plt.show()

# Plot confusion metrics, cara 2
plot_confusion_matrix(model, X_test, y_test, cmap="Blues_r")

# plot roc_auc curve
plot_precision_recall_curve(model,X_test,y_test)


# # Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()

# Fitting the model
rf.fit(X_train,y_train)

# Predicting the model
pred_rf= rf.predict(X_test)

# Evaluating the model
print("The accuracy of Random Forest model is:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test,pred_rf ))

# The accuracy of logit model is: 0.9187844495390469
#               precision    recall  f1-score   support

#            0       0.92      1.00      0.96     56538
#            1       0.40      0.01      0.02      4965

#     accuracy                           0.92     61503
#    macro avg       0.66      0.51      0.49     61503
# weighted avg       0.88      0.92      0.88     61503

# Plot confusion metrics
plot_confusion_matrix(rf, X_test, y_test, cmap="Blues_r")

# plot pprcision_recall curve
plot_precision_recall_curve(rf,X_test,y_test)

# #XGBoost Classifier
import xgboost as xgb
xgb_clf= xgb.XGBClassifier()

#fitting the model
xgb_clf.fit(X_train,y_train)

## Predicting the model
xgb_predict = xgb_clf.predict(X_test)

# Evaluating the model
print("The accuracy of XGBoost Classifier model is:", accuracy_score(y_test, xgb_predict))
print(classification_report(y_test, xgb_predict))

# The accuracy of XGBoost Classifier model is: 0.9192234525145115
#               precision    recall  f1-score   support

#            0       0.92      1.00      0.96     56538
#            1       0.50      0.03      0.06      4965

#     accuracy                           0.92     61503
#    macro avg       0.71      0.52      0.51     61503
# weighted avg       0.89      0.92      0.89     61503

# Plot confusion metrics
plot_confusion_matrix(xgb_clf, X_test, y_test, cmap="Blues_r")

# plot pprcision_recall curve
plot_precision_recall_curve(xgb_clf,X_test,y_test)

# Hyperparameter tunning
## Hyper Parameter Optimization
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]  
}

# Cross validation

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)

score
# array([0.91935484, 0.91951481, 0.91922214, 0.91974245, 0.91938474,
#        0.91948229, 0.91912458, 0.91941725, 0.91961237, 0.91984001])

score.mean()
# 0.9194695478038479







