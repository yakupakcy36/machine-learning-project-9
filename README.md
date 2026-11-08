# Titanic Survival Prediction

Small learning project. The notebook `tıtanic_survival .ipynb` trains logistic regression.

Data: Titanic passenger data (train (2).csv), predicting survival. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares logistic regression, random forest and HistGradientBoosting with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (891 samples, 8 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | accuracy | f1 | roc_auc |
|---|---|---|---|
| Logistic Regression (L2) | 0.8283 | 0.6087 | 0.8589 |
| Random Forest (200 Trees) | 0.8373 | 0.6154 | 0.8556 |
| HistGradientBoosting | 0.8339 | 0.639 | 0.8474 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `tıtanic_survival .ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
