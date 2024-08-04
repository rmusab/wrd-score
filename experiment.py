import pandas as pd
from scipy.stats import spearmanr, pearsonr
from rouge_score import rouge_scorer
from wrdscore import WRDScore
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_ind

wrd_score = WRDScore(model='codebert')
# r = ['calculate', 'total', 'amount']
# p = ['compute', 'aggregate', 'value']
# p = ['send', 'file', 'to', 'internet']
# pr, rc, f1 = wrd_score.wrdscore(r, p)
# print(f"Precision: {pr}, Recall: {rc}, F1: {f1}")

# Load CSV files
files = ['set1_res.csv', 'set2_res.csv', 'set3_res.csv']
dfs = [pd.read_csv(file, header=None, names=['Reference', 'Predicted', 'Human_score']) for file in files]

# Concatenate dataframes
data = pd.concat(dfs, ignore_index=True)

# Define a function to calculate ROUGE-1 score
def calculate_rouge_1(ref, pred):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    score = scorer.score(ref, pred)
    return score['rouge1'].fmeasure

# Calculate ROUGE-1 scores
data['ROUGE-1'] = data.apply(lambda row: calculate_rouge_1(row['Reference'], row['Predicted']), axis=1)

# Calculate WRDScore
data['WRDScore'] = data.apply(lambda row: wrd_score.wrdscore(row['Reference'].split(), row['Predicted'].split()), axis=1)

# Calculate BERTScore
data['BERTScore'] = data.apply(lambda row: wrd_score.bertscore(row['Reference'].split(), row['Predicted'].split()), axis=1)

# Calculate Spearman correlation for ROUGE-1
rouge_corr_sp, rouge_p_value_sp = spearmanr(data['Human_score'], data['ROUGE-1'])

# Calculate Spearman correlation for WRDScore
wrd_corr_sp, wrd_p_value_sp = spearmanr(data['Human_score'], data['WRDScore'])

# Calculate Spearman correlation for BERTScore
bert_corr_sp, bert_p_value_sp = spearmanr(data['Human_score'], data['BERTScore'])

# Calculate Pearson correlation for ROUGE-1
rouge_corr_pr, rouge_p_value_pr = pearsonr(data['Human_score'], data['ROUGE-1'])

# Calculate Pearson correlation for WRDScore
wrd_corr_pr, wrd_p_value_pr = pearsonr(data['Human_score'], data['WRDScore'])

# Calculate Pearson correlation for BERTScore
bert_corr_pr, bert_p_value_pr = pearsonr(data['Human_score'], data['BERTScore'])

# Print results
print(f"Spearman correlation for ROUGE-1: {rouge_corr_sp}, p-value: {rouge_p_value_sp}")
print(f"Spearman correlation for WRDScore: {wrd_corr_sp}, p-value: {wrd_p_value_sp}")
print(f"Spearman correlation for BERTScore: {bert_corr_sp}, p-value: {bert_p_value_sp}")

print(f"Pearson correlation for ROUGE-1: {rouge_corr_pr}, p-value: {rouge_p_value_pr}")
print(f"Pearson correlation for WRDScore: {wrd_corr_pr}, p-value: {wrd_p_value_pr}")
print(f"Pearson correlation for BERTScore: {bert_corr_pr}, p-value: {bert_p_value_pr}")

# Normalize Human_score by dividing by 10
data['Normalized_Human_score'] = data['Human_score'] / 10.0

# Calculate mean squared error for ROUGE-1
mse_rouge = mean_squared_error(data['Normalized_Human_score'], data['ROUGE-1'])

# Calculate mean squared error for WRDScore
mse_wrd = mean_squared_error(data['Normalized_Human_score'], data['WRDScore'])

# Calculate mean squared error for BERTScore
mse_bert = mean_squared_error(data['Normalized_Human_score'], data['BERTScore'])

# Print MSE results
print(f"Mean Squared Error for ROUGE-1: {mse_rouge}")
print(f"Mean Squared Error for WRDScore: {mse_wrd}")
print(f"Mean Squared Error for BERTScore: {mse_bert}")

# Calculate mean absolute error for ROUGE-1
mae_rouge = mean_absolute_error(data['Normalized_Human_score'], data['ROUGE-1'])

# Calculate mean absolute error for WRDScore
mae_wrd = mean_absolute_error(data['Normalized_Human_score'], data['WRDScore'])

# Calculate mean absolute error for BERTScore
mae_bert = mean_absolute_error(data['Normalized_Human_score'], data['BERTScore'])

# Print MSE results
print(f"Mean Absolute Error for ROUGE-1: {mae_rouge}")
print(f"Mean Absolute Error for WRDScore: {mae_wrd}")
print(f"Mean Absolute Error for BERTScore: {mae_bert}")

# Function to calculate p-values for T-test
def calculate_p_values(data1, data2):
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    return p_value

# Calculate p-values for T-test
p_value_rouge = calculate_p_values(data['Normalized_Human_score'], data['ROUGE-1'])
p_value_wrd = calculate_p_values(data['Normalized_Human_score'], data['WRDScore'])
p_value_bert = calculate_p_values(data['Normalized_Human_score'], data['BERTScore'])

# Print p-values
print(f"P-value for ROUGE-1: {p_value_rouge}")
print(f"P-value for WRDScore: {p_value_wrd}")
print(f"P-value for BERTScore: {p_value_bert}")