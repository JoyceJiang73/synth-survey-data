#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata

import pandas as pd
import torch
import os
import time


# In[22]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# In[23]:


directory = 'data/training/'
files = os.listdir(directory)

for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        if 'national' in file:
            df = df.drop(columns=['X', 'row_index'])
        else:
            df = df.drop(columns=['X','state','row_index'])
           
        print(file_path)
        metadata = Metadata.detect_from_dataframe(data=df)
            
        for i in range(5):  
            start_time = time.time()

            # Initialize TVAESynthesizer with a specific seed for each run
            TVAE = TVAESynthesizer(metadata, cuda=torch.cuda.is_available(),batch_size=len(df))
            # Train the model
            TVAE.fit(df)

            # Generate and save synthetic data
            synthetic_data = TVAE.sample(len(df))
            synthetic_data.to_csv(f'output/TVAE/'+str(file[:-4])+'_Run_'+str(i)+'.csv')

            end_time = time.time()
            print(f"Run {i+1}: Time taken for modeling - {end_time - start_time} seconds")

