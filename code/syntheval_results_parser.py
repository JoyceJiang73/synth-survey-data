import pandas as pd
import re
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Evaluation results', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 14)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_df(self, df):
        self.set_font('Arial', '', 12)
        # Replace unsupported characters
        df_str = df.to_string().replace('\u2013', '-')
        self.multi_cell(0, 10, df_str)
        self.ln()

def parse_results_classification(filename):
    results = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'\|\s*(\w+)\s*:\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)(\s*(\d+\.\d+))?', line)
            if match:
                classifier = match.group(1)
                acc_r = float(match.group(2))
                acc_f = float(match.group(3))
                diff = float(match.group(4))
                error = float(match.group(6)) if match.group(6) else None
                results.append([classifier, acc_r, acc_f, diff, error])
    return pd.DataFrame(results, columns=['Classifier', 'acc_r', 'acc_f', 'diff', 'error'])

def parse_results_main_and_classification(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    utility_metrics = []
    classification_accuracy = []
    privacy_metrics = []

    for line in lines:
        # Utility metrics
        match = re.match(r'\|\s*(.*?)\s*:\s*(\d+\.\d+)(\s*(\d+\.\d+))?', line)
        if match:
            metric = match.group(1)
            value = float(match.group(2))
            error = float(match.group(4)) if match.group(4) else None
            utility_metrics.append([metric, value, error])

        # Classification accuracy
        match = re.match(r'\|\s*(\w+)\s*:\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)(\s*(\d+\.\d+))?', line)
        if match:
            classifier = match.group(1)
            acc_r = float(match.group(2))
            acc_f = float(match.group(3))
            diff = float(match.group(4))
            error = float(match.group(6)) if match.group(6) else None
            classification_accuracy.append([classifier, acc_r, acc_f, diff, error])

        # Privacy metrics
        match = re.match(r'\|\s*(.*?)\s*:\s*(\d+\.\d+)(\s*(\d+\.\d+))?', line)
        if match:
            metric = match.group(1)
            value = float(match.group(2))
            error = float(match.group(4)) if match.group(4) else None
            privacy_metrics.append([metric, value, error])

    all_df = pd.DataFrame(utility_metrics, columns=['Metric', 'Value', 'Error'])
    classification_df = pd.DataFrame(classification_accuracy, columns=['Classifier', 'acc_r', 'acc_f', 'diff', 'error'])
    privacy_df = pd.DataFrame(privacy_metrics, columns=['Metric', 'Value', 'Error'])

    return all_df, classification_df

def process_files_classification(prefix, target, file_count):
    class_acc_frames = []
    holdout_frames = []

    model = prefix.capitalize()

    for i in range(1, file_count + 1):
        file_name = f'../output/syntheval/{model}/{target}/{prefix}_evaluation_results_{i}_{target}.txt'
        results = parse_results_classification(file_name)

        class_acc_frames.append(results[:5])
        holdout_frames.append(results[5:])

    class_acc_mean = pd.concat(class_acc_frames).groupby('Classifier').mean()
    holdout_mean = pd.concat(holdout_frames).groupby('Classifier').mean()

    return class_acc_mean, holdout_mean

def process_files_main_and_classification(prefix, target, file_count):
    utility_frames = []
    privacy_frames = []
    class_acc_frames = []
    holdout_frames = []

    model = prefix.capitalize()

    for i in range(1, file_count + 1):
        file_name = f'../output/syntheval/{model}/{target}/{prefix}_evaluation_results_{i}_{target}.txt'
        all_results, classification_results = parse_results_main_and_classification(file_name)

        utility_frames.append(all_results[:9])
        privacy_frames.append(all_results[19:])
        class_acc_frames.append(classification_results[:5])
        holdout_frames.append(classification_results[5:])

    utility_mean = pd.concat(utility_frames).groupby('Metric').mean()
    privacy_mean = pd.concat(privacy_frames).groupby('Metric').mean()
    class_acc_mean = pd.concat(class_acc_frames).groupby('Classifier').mean()
    holdout_mean = pd.concat(holdout_frames).groupby('Classifier').mean()

    return utility_mean, privacy_mean, class_acc_mean, holdout_mean

ctgan_utility_mean, ctgan_privacy_mean, ctgan_class_acc_mean_voted, ctgan_holdout_mean_voted = process_files_main_and_classification('ctgan', 'voted', 5)
ctgan_class_acc_mean_nonpartisan_donation, ctgan_holdout_mean_nonpartisan_donation = process_files_classification('ctgan', 'nonpartisan_donation', 5)
ctgan_class_acc_mean_Residence_HHParties_Description, ctgan_holdout_mean_Residence_HHParties_Description = process_files_classification('ctgan', 'Residence_HHParties_Description', 5)

ctgan_utility_mean.to_csv('../output/syntheval/CTGAN/ctgan_utility_mean.csv')
ctgan_privacy_mean.to_csv('../output/syntheval/CTGAN/ctgan_privacy_mean.csv')
ctgan_class_acc_mean_voted.to_csv('../output/syntheval/CTGAN/ctgan_class_acc_mean_voted.csv')
ctgan_holdout_mean_voted.to_csv('../output/syntheval/CTGAN/ctgan_holdout_mean_voted.csv')
ctgan_class_acc_mean_nonpartisan_donation.to_csv('../output/syntheval/CTGAN/ctgan_class_acc_mean_nonpartisan_donation.csv')
ctgan_holdout_mean_nonpartisan_donation.to_csv('../output/syntheval/CTGAN/ctgan_holdout_mean_nonpartisan_donation.csv')
ctgan_class_acc_mean_Residence_HHParties_Description.to_csv('../output/syntheval/CTGAN/ctgan_class_acc_mean_Residence_HHParties_Description.csv')
ctgan_holdout_mean_Residence_HHParties_Description.to_csv('../output/syntheval/CTGAN/ctgan_holdout_mean_Residence_HHParties_Description.csv')

pdf = PDF()

pdf.add_page()
pdf.chapter_title('CTGAN - Utility mean')
pdf.add_df(ctgan_utility_mean)

pdf.add_page()
pdf.chapter_title('CTGAN - Privacy mean')
pdf.add_df(ctgan_privacy_mean)

pdf.add_page()
pdf.chapter_title('CTGAN - Classification accuracy mean (Voted)')
pdf.add_df(ctgan_class_acc_mean_voted)

pdf.add_page()
pdf.chapter_title('CTGAN - Holdout mean (Voted)')
pdf.add_df(ctgan_holdout_mean_voted)

pdf.add_page()
pdf.chapter_title('CTGAN - Classification accuracy mean (Nonpartisan donation)')
pdf.add_df(ctgan_class_acc_mean_nonpartisan_donation)

pdf.add_page()
pdf.chapter_title('CTGAN - Holdout mean (Nonpartisan donation)')
pdf.add_df(ctgan_holdout_mean_nonpartisan_donation)

pdf.add_page()
pdf.chapter_title('CTGAN - Classification accuracy mean (Residence HHParties Description)')
pdf.add_df(ctgan_class_acc_mean_Residence_HHParties_Description)

pdf.add_page()
pdf.chapter_title('CTGAN - Holdout mean (Residence HHParties Description)')
pdf.add_df(ctgan_holdout_mean_Residence_HHParties_Description)

pdf.output('../output/syntheval/ctgan_evaluation_results.pdf', 'F')

llm_utility_mean, llm_privacy_mean, llm_class_acc_mean_voted, llm_holdout_mean_voted = process_files_main_and_classification('llm', 'voted', 5)
llm_class_acc_mean_nonpartisan_donation, llm_holdout_mean_nonpartisan_donation = process_files_classification('llm', 'nonpartisan_donation', 5)
llm_class_acc_mean_Residence_HHParties_Description, llm_holdout_mean_Residence_HHParties_Description = process_files_classification('llm', 'Residence_HHParties_Description', 5)

llm_utility_mean.to_csv('../output/syntheval/LLM/llm_utility_mean.csv')
llm_privacy_mean.to_csv('../output/syntheval/LLM/llm_privacy_mean.csv')
llm_class_acc_mean_voted.to_csv('../output/syntheval/LLM/llm_class_acc_mean_voted.csv')
llm_holdout_mean_voted.to_csv('../output/syntheval/LLM/llm_holdout_mean_voted.csv')
llm_class_acc_mean_nonpartisan_donation.to_csv('../output/syntheval/LLM/llm_class_acc_mean_nonpartisan_donation.csv')
llm_holdout_mean_nonpartisan_donation.to_csv('../output/syntheval/LLM/llm_holdout_mean_nonpartisan_donation.csv')
llm_class_acc_mean_Residence_HHParties_Description.to_csv('../output/syntheval/LLM/llm_class_acc_mean_Residence_HHParties_Description.csv')
llm_holdout_mean_Residence_HHParties_Description.to_csv('../output/syntheval/LLM/llm_holdout_mean_Residence_HHParties_Description.csv')

pdf = PDF()

pdf.add_page()
pdf.chapter_title('LLM - Utility mean')
pdf.add_df(llm_utility_mean)

pdf.add_page()
pdf.chapter_title('LLM - Privacy mean')
pdf.add_df(llm_privacy_mean)

pdf.add_page()
pdf.chapter_title('LLM - Classification accuracy mean (Voted)')
pdf.add_df(llm_class_acc_mean_voted)

pdf.add_page()
pdf.chapter_title('LLM - Holdout mean (Voted)')
pdf.add_df(llm_holdout_mean_voted)

pdf.add_page()
pdf.chapter_title('LLM - Classification accuracy mean (Nonpartisan donation)')
pdf.add_df(llm_class_acc_mean_nonpartisan_donation)

pdf.add_page()
pdf.chapter_title('LLM - Holdout mean (Nonpartisan donation)')
pdf.add_df(llm_holdout_mean_nonpartisan_donation)

pdf.add_page()
pdf.chapter_title('LLM - Classification accuracy mean (Residence HHParties Description)')
pdf.add_df(llm_class_acc_mean_Residence_HHParties_Description)

pdf.add_page()
pdf.chapter_title('LLM - Holdout mean (Residence HHParties Description)')
pdf.add_df(llm_holdout_mean_Residence_HHParties_Description)

pdf.output('../output/syntheval/llm_evaluation_results.pdf', 'F')

synthpop_utility_mean, synthpop_privacy_mean, synthpop_class_acc_mean_voted, synthpop_holdout_mean_voted = process_files_main_and_classification('synthpop', 'voted', 5)
synthpop_class_acc_mean_nonpartisan_donation, synthpop_holdout_mean_nonpartisan_donation = process_files_classification('synthpop', 'nonpartisan_donation', 5)
synthpop_class_acc_mean_Residence_HHParties_Description, synthpop_holdout_mean_Residence_HHParties_Description = process_files_classification('synthpop', 'Residence_HHParties_Description', 5)

synthpop_utility_mean.to_csv('../output/syntheval/SYNTHPOP/synthpop_utility_mean.csv')
synthpop_privacy_mean.to_csv('../output/syntheval/SYNTHPOP/synthpop_privacy_mean.csv')
synthpop_class_acc_mean_voted.to_csv('../output/syntheval/SYNTHPOP/synthpop_class_acc_mean_voted.csv')
synthpop_holdout_mean_voted.to_csv('../output/syntheval/SYNTHPOP/synthpop_holdout_mean_voted.csv')
synthpop_class_acc_mean_nonpartisan_donation.to_csv('../output/syntheval/SYNTHPOP/synthpop_class_acc_mean_nonpartisan_donation.csv')
synthpop_holdout_mean_nonpartisan_donation.to_csv('../output/syntheval/SYNTHPOP/synthpop_holdout_mean_nonpartisan_donation.csv')
synthpop_class_acc_mean_Residence_HHParties_Description.to_csv('../output/syntheval/SYNTHPOP/synthpop_class_acc_mean_Residence_HHParties_Description.csv')
synthpop_holdout_mean_Residence_HHParties_Description.to_csv('../output/syntheval/SYNTHPOP/synthpop_holdout_mean_Residence_HHParties_Description.csv')

pdf = PDF()

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Utility mean')
pdf.add_df(synthpop_utility_mean)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Privacy mean')
pdf.add_df(synthpop_privacy_mean)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Classification accuracy mean (Voted)')
pdf.add_df(synthpop_class_acc_mean_voted)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Holdout mean (Voted)')
pdf.add_df(synthpop_holdout_mean_voted)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Classification accuracy mean (Nonpartisan donation)')
pdf.add_df(synthpop_class_acc_mean_nonpartisan_donation)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Holdout mean (Nonpartisan donation)')
pdf.add_df(synthpop_holdout_mean_nonpartisan_donation)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Classification accuracy mean (Residence HHParties Description)')
pdf.add_df(synthpop_class_acc_mean_Residence_HHParties_Description)

pdf.add_page()
pdf.chapter_title('SYNTHPOP - Holdout mean (Residence HHParties Description)')
pdf.add_df(synthpop_holdout_mean_Residence_HHParties_Description)

pdf.output('../output/syntheval/synthpop_evaluation_results.pdf', 'F')

# Combined Train/Test Results

combined_test_train_results = pd.DataFrame(columns=["Model", 
                                                    "F1 Vote Train", 
                                                    "F1 Diff Vote Train",
                                                    "F1 Vote Test", 
                                                    "F1 Diff Vote Test",
                                                    "F1 Donation Train", 
                                                    "F1 Diff Donation Train",
                                                    "F1 Donation Test", 
                                                    "F1 Diff Donation Test", 
                                                    "F1 Party Train", 
                                                    "F1 Diff Part Train",
                                                    "F1 Party Test", 
                                                    "F1 Diff Party Test"])

models = ["CTGAN", "LLM", "Synthpop"]

dataframes = [ctgan_class_acc_mean_voted, 
              ctgan_holdout_mean_voted, 
              ctgan_class_acc_mean_nonpartisan_donation, 
              ctgan_holdout_mean_nonpartisan_donation, 
              ctgan_class_acc_mean_Residence_HHParties_Description, 
              ctgan_holdout_mean_Residence_HHParties_Description, 
              llm_class_acc_mean_voted, 
              llm_holdout_mean_voted, 
              llm_class_acc_mean_nonpartisan_donation, 
              llm_holdout_mean_nonpartisan_donation, 
              llm_class_acc_mean_Residence_HHParties_Description, 
              llm_holdout_mean_Residence_HHParties_Description, 
              synthpop_class_acc_mean_voted,
              synthpop_holdout_mean_voted,
              synthpop_class_acc_mean_nonpartisan_donation,
              synthpop_holdout_mean_nonpartisan_donation,
              synthpop_class_acc_mean_Residence_HHParties_Description,
              synthpop_holdout_mean_Residence_HHParties_Description]

results = []

for i in range(0, len(dataframes), 6):
    model_name = models[i // 6]
    voted_f1_train = dataframes[i].loc['Average', 'acc_f']
    voted_f1_diff_train = dataframes[i].loc['Average', 'diff']
    voted_f1_test = dataframes[i+1].loc['Average', 'acc_f']
    voted_f1_diff_test = dataframes[i+1].loc['Average', 'diff']
    donation_f1_train = dataframes[i+2].loc['Average', 'acc_f']
    donation_f1_diff_train = dataframes[i+2].loc['Average', 'diff']
    donation_f1_test = dataframes[i+3].loc['Average', 'acc_f']
    donation_f1_diff_test = dataframes[i+3].loc['Average', 'diff']
    party_f1_train = dataframes[i+4].loc['Average', 'acc_f']
    party_f1_diff_train = dataframes[i+4].loc['Average', 'diff']
    party_f1_test = dataframes[i+5].loc['Average', 'acc_f']
    party_f1_diff_test = dataframes[i+5].loc['Average', 'diff']

    results.append(pd.DataFrame({"Model": [model_name], 
                                 "F1 Vote Train": [voted_f1_train], 
                                 "F1 Diff Vote Train": [voted_f1_diff_train],
                                 "F1 Vote Test": [voted_f1_test], 
                                 "F1 Diff Vote Test": [voted_f1_diff_test],
                                 "F1 Donation Train": [donation_f1_train], 
                                 "F1 Diff Donation Train": [donation_f1_diff_train],
                                 "F1 Donation Test": [donation_f1_test], 
                                 "F1 Diff Donation Test": [donation_f1_diff_test],
                                 "F1 Party Train": [party_f1_train], 
                                 "F1 Diff Party Train": [party_f1_diff_train],
                                 "F1 Party Test": [party_f1_test],
                                 "F1 Diff Party Test": [party_f1_diff_test]}))

combined_test_train_results = pd.concat(results, ignore_index=True)

# round combined_test_train_results to 4 decimal places

combined_test_train_results = combined_test_train_results.round(4)

combined_test_train_results

# export combined_results to csv

combined_test_train_results.to_csv('../output/syntheval/combined_test_train_results.csv', index=False)

ctgan_utility_mean

models = ["CTGAN", "LLM", "Synthpop"]
dataframes = [ctgan_utility_mean, llm_utility_mean, synthpop_utility_mean]

results = []

for i in range(len(dataframes)):
    model_name = models[i]
    pval = dataframes[i].loc['-> average combined p-value', 'Value']
    stat = dataframes[i].loc['-> average combined statistic', 'Value']
    pMSE_class_acc = dataframes[i].loc['-> average pMSE classifier accuracy', 'Value']
    KS_dist = dataframes[i].loc['-> avg. Kolmogorov–Smirnov dist.', 'Value']
    total_var_dist = dataframes[i].loc['-> avg. Total Variation Distance', 'Value']
    sig_test_frac = dataframes[i].loc['-> fraction of significant tests', 'Value']
    mixed_cor_dist = dataframes[i].loc['Mixed correlation matrix difference', 'Value']
    mut_inf_dif = dataframes[i].loc['Pairwise mutual information difference', 'Value']
    pMSE = dataframes[i].loc['Propensity mean squared error (pMSE)', 'Value']

    results.append(pd.DataFrame({"Model": [model_name], 
                                 "avg combined p-value": [pval], 
                                 "avg combined statistic": [stat], 
                                 "avg pMSE classifier accuracy": [pMSE_class_acc], 
                                 "avg Kolmogorov-Smirnov dist.": [KS_dist], 
                                 "avg Total Variation Distance": [total_var_dist], 
                                 "Fraction of significant tests": [sig_test_frac], 
                                 "Mixed correlation matrix difference": [mixed_cor_dist], 
                                 "Pairwise mutual information difference": [mut_inf_dif], 
                                 "Propensity mean squared error (pMSE)": [pMSE]}))

combined_utility_results = pd.concat(results, ignore_index=True)

# round combined_utility_results to 4 decimal places

combined_utility_results = combined_utility_results.round(4)

combined_utility_results

combined_utility_results.to_csv('../output/syntheval/combined_utility_results.csv', index=False)

models = ["CTGAN", "LLM", "Synthpop"]
dataframes = [ctgan_privacy_mean, llm_privacy_mean, synthpop_privacy_mean]

results = []

for i in range(len(dataframes)):
    model_name = models[i]
    precision = dataframes[i].loc['-> Precision', 'Value']
    recall = dataframes[i].loc['-> Recall', 'Value']
    disc_risk = dataframes[i].loc['Attr. disclosure risk (acc. with holdout)', 'Value']
    eps_risk = dataframes[i].loc['Epsilon identifiability risk', 'Value']
    med_closest = dataframes[i].loc['Median distance to closest record', 'Value']
    mia_f1 = dataframes[i].loc['Membership inference attack Classifier F1', 'Value']

    results.append(pd.DataFrame({"Model": [model_name],
                                    "Precision": [precision],
                                    "Recall": [recall],
                                    "Attribute disclosure risk (acc, with holdout)": [disc_risk],
                                    "Epsilon identifiability risk": [eps_risk],
                                    "Median distance to closest record": [med_closest],
                                    "Membership inference attack Classifier F1": [mia_f1]}))
    
combined_privacy_results = pd.concat(results, ignore_index=True)

# round combined_privacy_results to 4 decimal places

combined_privacy_results = combined_privacy_results.round(4)

combined_privacy_results

combined_privacy_results.to_csv('../output/syntheval/combined_privacy_results.csv', index=False)