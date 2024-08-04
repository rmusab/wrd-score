# Word Rotator's Distance Score (WRDScore)
© Mussabayev Ravil, 2024

[![arXiv](https://img.shields.io/badge/arXiv-2405.19220-B31B1B)](https://arxiv.org/abs/2405.19220)

## Overview

This repository contains the implementation of Word Rotator's Distance Score (WRDScore), a novel metric for multi-purpose evaluation of natural language generation (NLG) models. WRDScore leverages word embeddings, like Word2Vec and CodeBERT, to compute the distance between reference and predicted sequences. Additionally, it includes a method to compute BERTScore, a popular metric for evaluating text generation models, as well as scripts for training Word2Vec embeddings on code corpora.

## Requirements

The following Python packages are required:
- pandas
- scipy
- rouge-score
- wrdscore
- scikit-learn
- numpy
- POT
- transformers
- torch
- gensim

You can install these dependencies using the provided `requirements.txt` file:
```sh
pip install -r requirements.txt
```

## WRDScore Class

The WRDScore class provides methods to compute WRDScore and BERTScore. It supports both CodeBERT and Word2Vec models for generating word embeddings.

### Initialization

The class can be initialized with either CodeBERT or Word2Vec model:

```python
wrd_score = WRDScore(model='codebert')
# or
wrd_score = WRDScore(model='word2vec')
```

### Methods

- `tokens_to_embeddings(tokens, model)`: Converts a sequence of tokens into embeddings.
- `compute_embeddings_and_weights(r, p)`: Computes embeddings and weight vectors for reference and predicted sequences.
- `word_rotator_distance(r_context_normalized, p_context_normalized, r_weights, p_weights)`: Computes the Word Rotator's Distance.
- `compute_precision(r_context_normalized, p_context_normalized, r_weights, p_weights)`: Computes precision for WRDScore.
- `wrdscore(r, p)`: Computes WRDScore for reference and predicted sequences.
- `bertscore(r, p)`: Computes BERTScore for reference and predicted sequences.

### Usage Example

```python
wrd_score = WRDScore(model='codebert')
reference = ['calculate', 'total', 'amount']
predicted = ['compute', 'aggregate', 'value']
score = wrd_score.wrdscore(reference, predicted)
print(f"WRDScore: {score}")
```

## Evaluation Script

The repository includes a script to evaluate the performance of WRDScore, BERTScore, and ROUGE-1 using a dataset of reference and predicted sequences along with human scores.

### Loading Data

CSV files containing the reference, predicted, and human scores are loaded and concatenated:

```python
files = ['set1_res.csv', 'set2_res.csv', 'set3_res.csv']
dfs = [pd.read_csv(file, header=None, names=['Reference', 'Predicted', 'Human_score']) for file in files]
data = pd.concat(dfs, ignore_index=True)
```

### Calculating Scores

ROUGE-1, WRDScore, and BERTScore are calculated for each reference-predicted pair:

```python
data['ROUGE-1'] = data.apply(lambda row: calculate_rouge_1(row['Reference'], row['Predicted']), axis=1)
data['WRDScore'] = data.apply(lambda row: wrd_score.wrdscore(row['Reference'].split(), row['Predicted'].split()), axis=1)
data['BERTScore'] = data.apply(lambda row: wrd_score.bertscore(row['Reference'].split(), row['Predicted'].split()), axis=1)
```

### Statistical Analysis

The script computes Spearman and Pearson correlations, Mean Squared Error (MSE), and Mean Absolute Error (MAE) between the human scores and the calculated scores.

## Training Word2Vec Model

The repository includes a script to train a Word2Vec model on a collection of Java files:

```python
directory_path = 'Data/java-med'
trained_model = train_word2vec_model(directory_path, tokenizer, window_size=2)
```

## Citing Our Work

If you find our research useful in your work, please cite our work:

```bibtex
@misc{mussabayev2024wrdscorenewmetricevaluation,
      title={WRDScore: New Metric for Evaluation of Natural Language Generation Models}, 
      author={Ravil Mussabayev},
      year={2024},
      eprint={2405.19220},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.19220}, 
}
```