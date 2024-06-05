import pandas as pd
from realtabformer import REaLTabFormer

if __name__ == "__main__":
    df = pd.read_csv("foo.csv")

    rtf_model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=4,
        logging_steps=100)

    rtf_model.fit(df)
    # save model
    # rtf_model.save("rtf_model/")

    # Generate synthetic data with the same
    # number of observations as the real dataset.
    samples = rtf_model.sample(n_samples=len(df)) # type is dataframe
    df.to_csv('/home/alien/Documents/REaLTabFormer/data/gen_1.csv', sep='\t')




