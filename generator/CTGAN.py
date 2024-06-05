#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.impute import KNNImputer
import time


# In[4]:

df = pd.read_csv('data/training_20k.csv')
#df = pd.read_csv('data/train.csv')
#df = df.dropna()

# In[5]:

# Data Imputation
# For numerical columns
df['calculated_age'].fillna(df['calculated_age'].median(), inplace=True)
df['CommercialData_EstimatedIncomeAmount'].fillna(df['CommercialData_EstimatedIncomeAmount'].median(), inplace=True)
df['CommercialData_EstHomeValue'].fillna(df['CommercialData_EstHomeValue'].median(), inplace=True)

# For categorical columns
df['state'].fillna(df['state'].mode()[0], inplace=True)
df['Parties_Description'].fillna(df['Parties_Description'].mode()[0], inplace=True)
df['EthnicGroups_EthnicGroup1Desc'].fillna(df['EthnicGroups_EthnicGroup1Desc'].mode()[0], inplace=True)
df['Ethnic_Description'].fillna(df['Ethnic_Description'].mode()[0], inplace=True)
df['Residence_HHParties_Description'].fillna(df['Residence_HHParties_Description'].mode()[0], inplace=True)
df['CommercialData_PropertyType'].fillna(df['CommercialData_PropertyType'].mode()[0], inplace=True)


# In[7]:


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU acceleration can be used.")
else:
    print("CUDA is not available. GPU acceleration is not supported.")


# In[9]:


from ctgan import CTGAN

discrete_columns = ['state', 'Parties_Description', 'EthnicGroups_EthnicGroup1Desc', 'Ethnic_Description',
                    'Residence_HHParties_Description', 'CommercialData_PropertyType', 'voted', 'Voters_Gender', 'nonpartisan_donation']

#df = df.drop('row_index', axis=1)

for i in range(5):  # Running the model five times with different seeds
    start_time = time.time()

    # Initialize CTGAN Synthesizer 
    ctgan = CTGAN(epochs=300, cuda=True, batch_size=40960)

    # Train the model
    ctgan.fit(df, discrete_columns=discrete_columns)

    # Save the trained CTGAN model
    model_path = f"model/ctgan_model_impute_all_run_{i+1}.pth"
    ctgan.save(model_path)

    # Generate and save synthetic data
    synthetic_data = ctgan.sample(len(df))
    synthetic_data.to_csv(f'output/CTGAN_Impute_All_Run_{i+1}.csv')

    end_time = time.time()
    print(f"Run {i+1}: Time taken for modeling - {end_time - start_time} seconds")
    print(f"Run {i+1}: CTGAN model saved at: {model_path}")

