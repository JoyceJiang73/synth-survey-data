import pandas as pd
from realtabformer import REaLTabFormer
import time
import torch
import os

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU acceleration can be used.")
else:
    print("CUDA is not available. GPU acceleration is not supported.")

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

        for i in range(1):  # Running the model five times with different seeds
            start_time = time.time()

            rtf_model = REaLTabFormer(
                model_type="tabular",
                gradient_accumulation_steps=4,
                logging_steps=100,
                random_state=i)    

            rtf_model.fit(df)
            synthetic_data = rtf_model.sample(n_samples=len(df)) # type is dataframe
            synthetic_data.to_csv(f'output/LLMs/'+str(file[:-4])+'_Run_'+str(i)+'.csv')

            end_time = time.time()
            print(f"Run {i+1}: Time taken for modeling - {end_time - start_time} seconds")

