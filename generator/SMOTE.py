#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
from sklearn.impute import KNNImputer
import time
from collections import Counter
import os
import numpy as np
from imblearn.over_sampling import SMOTENC


# In[83]:


directory = 'data/training/'
files = os.listdir(directory)

for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        if 'national' in file:
            df = df.drop(columns=['X', 'row_index'])
            df['state'].fillna(df['state'].mode()[0], inplace=True)
            discrete_columns = ['state', 'Parties_Description', 'EthnicGroups_EthnicGroup1Desc', 'Ethnic_Description',
                    'Residence_HHParties_Description', 'CommercialData_PropertyType', 'voted', 'Voters_Gender', 'nonpartisan_donation']

        else:
            df = df.drop(columns=['X','state','row_index'])
            discrete_columns = ['Parties_Description', 'EthnicGroups_EthnicGroup1Desc', 'Ethnic_Description',
                    'Residence_HHParties_Description', 'CommercialData_PropertyType', 'voted', 'Voters_Gender', 'nonpartisan_donation']
            
        print(file_path)
            
        # For numerical columns
        df['calculated_age'].fillna(df['calculated_age'].median(), inplace=True)
        df['CommercialData_EstimatedIncomeAmount'].fillna(df['CommercialData_EstimatedIncomeAmount'].median(), inplace=True)
        df['CommercialData_EstHomeValue'].fillna(df['CommercialData_EstHomeValue'].median(), inplace=True)

        # For categorical columns
        df['Voters_Gender'].fillna(df['Voters_Gender'].mode()[0], inplace=True)
        df['Parties_Description'].fillna(df['Parties_Description'].mode()[0], inplace=True)
        df['EthnicGroups_EthnicGroup1Desc'].fillna(df['EthnicGroups_EthnicGroup1Desc'].mode()[0], inplace=True)
        df['Ethnic_Description'].fillna(df['Ethnic_Description'].mode()[0], inplace=True)
        df['Residence_HHParties_Description'].fillna(df['Residence_HHParties_Description'].mode()[0], inplace=True)
        df['CommercialData_PropertyType'].fillna(df['CommercialData_PropertyType'].mode()[0], inplace=True)

        for i in range(5):  # Running the model five times with different seeds
            start_time = time.time()
            
            synthetic_data = []
            rs = i*5
            
            while len(synthetic_data) < 0.999*len(df):

                X_real = df
                y_label = np.random.permutation([0] * (len(df) // 4) + [1] * (len(df) - len(X_real) // 4))

                smote = SMOTENC(categorical_features = discrete_columns, sampling_strategy = 'all', random_state = rs)
                X_resampled, y_resampled = smote.fit_resample(X_real, y_label)

                X_real = df
                y_label2 = np.random.permutation([0] * (len(df) // 4) + [1] * (len(df) - len(X_real) // 4))

                smote = SMOTENC(categorical_features = discrete_columns, sampling_strategy = 'all', random_state = rs+2)
                X_resampled2, y_resampled2 = smote.fit_resample(X_real, y_label2)

                stacked_df = pd.concat([X_resampled.iloc[len(df):, :], X_resampled2.iloc[len(df):, :]])
                synthetic_data = stacked_df.reset_index(drop=True)
                dedup_df = synthetic_data.drop_duplicates()
                
                rs +=1
                print('random_state: ', rs)
            
            print('The real_df is ', len(df))
            print('The fake_df is ', len(synthetic_data))
            print('The dedup fake_df is ', len(dedup_df))
            synthetic_data.to_csv(f'output/SMOTE/'+str(file[:-4])+'_Run_'+str(i)+'.csv')

            end_time = time.time()
            print(f"Run {i+1}: Time taken for modeling - {end_time - start_time} seconds")
        
