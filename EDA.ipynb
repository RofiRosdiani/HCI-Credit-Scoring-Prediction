# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:44:19 2023

@author: admin
"""

# Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import pearsonr

import time
from sklearn import preprocessing

from tqdm import tqdm
tqdm.pandas()
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Import Dataset
application_train = pd.read_csv("application_train.csv")

application_train.columns

# Pisahkan antara feature data dengan target data
feature_df = application_train.drop(['SK_ID_CURR', 'TARGET', ],axis=1)
target_df = application_train['TARGET']

# Melihat feature apa saja yang memiliki missing value lebih dari 30%
missing_values = pd.DataFrame(feature_df.isnull().sum()/feature_df.shape[0])
missing_values = missing_values[missing_values.iloc[:,0] > 0.50]
missing_values.sort_values([0], ascending=False)

# Mengubah index menjadi columns
missing_values.reset_index(inplace=True)
missing_values = missing_values.rename(columns = {'index':'variable'})

# Menampilkan isi column
missing_values['variable'].values

# Drop feature tersebut
feature_df = feature_df.drop(['OWN_CAR_AGE', 'EXT_SOURCE_1', 'APARTMENTS_AVG',
       'BASEMENTAREA_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
       'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
       'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
       'BASEMENTAREA_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
       'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMIN_MODE',
       'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
       'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BUILD_MEDI',
       'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
       'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
       'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
       'WALLSMATERIAL_MODE'], axis=1)

# # Pengecheckan ulang apakah feature tersebut berhasil di drop
# missing_values = pd.DataFrame(feature_df.isnull().sum()/feature_df.shape[0])
# missing_values = missing_values[missing_values.iloc[:,0] > 0.50]
# missing_values.sort_values([0], ascending=False)


# feature_df['BASEMENTAREA_AVG']


feature_df.isnull().sum().sum()

# Change Nan with 0 
feature_df = feature_df.replace(np.nan, 0)

feature_df.info()

# Check Distribution Plot numerical data
# numerical = feature_df.select_dtypes(exclude=['object'])
# col_names = numerical.columns

# fig, ax = plt.subplots(len(col_names), figsize=(16,16))
# fig.subplots_adjust(hspace=.5)

# for i, col_val in enumerate(col_names):

#     sns.distplot(numerical[col_val].dropna(), hist=True, ax=ax[i])
#     ax[i].set_xlabel(col_val, fontsize=12)
#     ax[i].set_ylabel('Count', fontsize=12)

# plt.show()

# Convert data type (Object -> Category)
# Kita melakukan convert tipe data karena computing time 
# untuk data tipe category lebih cepat dibanding object

# Convert object type to category type
for col in feature_df.columns:
  if feature_df[col].dtypes == 'O':
    feature_df[col] = feature_df[col].astype('category')

# Split Numerical and Categorical Dataframe
num_feature_df = feature_df.select_dtypes(exclude=['category'])
cat_feature_df = feature_df.select_dtypes(include=['category'])

# Visualize Target Distribution
ax = sns.countplot(target_df, label="Count", palette = "winter")
L, T = target_df.value_counts()
plt.show()
print("Jumlah LAYAK diberikan pinjaman: ", L)
print("Jumlah TIDAK LAYAK diberikan pinjaman: ", T)

# Concat feature and target dataframe
data_df = pd.concat([feature_df,target_df], axis=1)

# EDA on CATEGORICAL DATA
# Check total unique value for each feature

cat_feature_df.info()
# Data columns (total 13 columns):
#  #   Column                      Non-Null Count   Dtype   
# ---  ------                      --------------   -----   
#  0   NAME_CONTRACT_TYPE          307511 non-null  category
#  1   CODE_GENDER                 307511 non-null  category
#  2   FLAG_OWN_CAR                307511 non-null  category
#  3   FLAG_OWN_REALTY             307511 non-null  category
#  4   NAME_TYPE_SUITE             307511 non-null  category
#  5   NAME_INCOME_TYPE            307511 non-null  category
#  6   NAME_EDUCATION_TYPE         307511 non-null  category
#  7   NAME_FAMILY_STATUS          307511 non-null  category
#  8   NAME_HOUSING_TYPE           307511 non-null  category
#  9   OCCUPATION_TYPE             307511 non-null  category
#  10  WEEKDAY_APPR_PROCESS_START  307511 non-null  category
#  11  ORGANIZATION_TYPE           307511 non-null  category
#  12  EMERGENCYSTATE_MODE         307511 non-null  category

cat_feature_df.columns

cat_feature_df_1 = cat_feature_df[['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']]
cat_feature_df_1.info()

cat_feature_df_2 = cat_feature_df[['NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
'EMERGENCYSTATE_MODE']]

# Check total unique value
for col in cat_feature_df_1.columns:
  print(col,"= ",cat_feature_df_1[col].dropna().unique())
  print('\n')

#Countplot Categorical Data
col_names = cat_feature_df_1.columns

# Check countplot for each categorical data ( For see the most categories for each feature )

fig, ax = plt.subplots(len(col_names), figsize=(20,100))
fig.subplots_adjust(hspace=0.9)

for i, col_val in enumerate(col_names):

    graph = sns.countplot(y=cat_feature_df_1[col_val].dropna(), ax=ax[i], edgecolor='black', palette='winter', orient='v')
    k=0
    for p in graph.patches:
      height = p.get_height()
      # graph.text(p.get_width(), p.get_y()+0.5,cat_feature_df_1[col_val].value_counts()[k],ha="right")
      k+=0
      
    ax[i].set_xlabel('Count', fontsize=10)
    ax[i].set_ylabel(col_val, fontsize=10)

plt.show()

# # visualize cat_feature_df_1['NAME_INCOME_TYPE']
# cat_feature_df_1['NAME_INCOME_TYPE'].value_counts(ascending=False)
# # Working                 158774
# # Commercial associate     71617
# # Pensioner                55362
# # State servant            21703
# # Unemployed                  22
# # Student                     18
# # Businessman                 10
# # Maternity leave              5

# labels_nit = cat_feature_df_1['NAME_INCOME_TYPE'].value_counts().index
# values_nit = cat_feature_df_1['NAME_INCOME_TYPE'].value_counts().values

# plt.figure(figsize = (15, 8))
# ax = sns.barplot(x=labels_nit, y=values_nit)
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2., height + 0.1, cat_feature_df_1['NAME_INCOME_TYPE'].value_counts().sort_index()[i],ha="center")

#     ax.set_xlabel('Count', fontsize=10)
#     ax.set_ylabel('NAME_INCOME_TYPE', fontsize=10)
#     ax.set_title('NAME_INCOME_TYPE', fontsize=10)   
 
# plt.show()

# # visualize cat_feature_df_1 'NAME_EDUCATION_TYPE'
# cat_feature_df_1['NAME_EDUCATION_TYPE'].value_counts(ascending=False)
# # Secondary / secondary special    218391
# # Higher education                  74863
# # Incomplete higher                 10277
# # Lower secondary                    3816
# # Academic degree                     164

# labels_net = cat_feature_df_1['NAME_EDUCATION_TYPE'].value_counts().index
# values_net = cat_feature_df_1['NAME_EDUCATION_TYPE'].value_counts().values

# plt.figure(figsize = (15, 8))
# ax = sns.barplot(x=labels_net, y=values_net)
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2., height + 0.1, cat_feature_df_1['NAME_EDUCATION_TYPE'].value_counts().sort_index()[i],ha="center")

#     ax.set_xlabel('Count', fontsize=10)
#     ax.set_ylabel('NAME_EDUCATION_TYPE', fontsize=10)
#     ax.set_title('NAME_EDUCATION_TYPE', fontsize=10)   
 
# plt.show()

# Check total unique value
for col in cat_feature_df_2.columns:
  print(col,"= ",cat_feature_df_2[col].dropna().unique())
  print('\n')

#Countplot Categorical Data
col_names = cat_feature_df_2.columns

# Check countplot for each categorical data ( For see the most categories for each feature )

fig, ax = plt.subplots(len(col_names), figsize=(30,200))
fig.subplots_adjust(hspace=0.2)

for i, col_val in enumerate(col_names):

    graph = sns.countplot(y=cat_feature_df_2[col_val].dropna(), ax=ax[i], edgecolor='black', palette='winter', orient='v')
    j=0
    for p in graph.patches:
      height = p.get_height()
      # graph.text(p.get_width(), p.get_y()+0.5,cat_feature_df_2[col_val].value_counts()[j],ha="right")
      j+=0
      
    ax[i].set_xlabel('Count', fontsize=10)
    ax[i].set_ylabel(col_val, fontsize=10)

plt.show()

# EDA on NUMERICAL DATA

num_feature_df.describe().head(10)

# Visualize Gender categories proportion
feature_df.CODE_GENDER.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(5, 5))

# Visualize family status categories proportion
feature_df.NAME_FAMILY_STATUS.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(5, 5))

# Visualize Type of Loans categories proportion
feature_df["NAME_CONTRACT_TYPE"].value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(5, 5))

# Visualize NAME_HOUSING_TYPE categories proportion
feature_df.NAME_HOUSING_TYPE.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(20, 10), subplots=True, legend = True)

# Visualize Type of Income categories proportion
feature_df.NAME_INCOME_TYPE.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(20, 10), subplots=True, legend = True)

# Visualize Education categories proportion
feature_df.NAME_EDUCATION_TYPE.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(20, 10), subplots=True, legend = True)
















