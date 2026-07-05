# Introduction to Machine Learning

A learning project: every major ML algorithm outside KNN/Naive Bayes/Decision
Trees (which you already know from studying on paper) implemented on the
exact same 2-row dataset, organized by category, so they can be compared
side by side. KNN/Naive Bayes/Decision Trees are included too (in
`classification/`) purely so the comparison is complete in one place.

## The dataset

**`data/loan_data.csv`** is the *only* dataset used anywhere in this
project -- every script loads it from the same relative path
(`../data/loan_data.csv`). It has exactly 2 rows:

```
age,income,credit_score,years_employed,existing_debt,approved
25,80,750,3,5,1
55,20,580,20,40,0
```

This is intentionally too small to do "real" machine learning with. The
point isn't predictive accuracy -- it's watching each algorithm's
internal mechanics operate on data simple enough to trace by hand, and
seeing exactly where each algorithm's assumptions break down when there
isn't enough data to support them (documented in-script wherever it
happens, rather than papered over).

## Setup

```
pip install -r requirements.txt
```

Every script is self-contained and run from inside its own folder:

```
cd classification
python 01_knn.py
```

## Folder structure

```
introduction-to-machine-learning/
├── data/loan_data.csv
├── classification/            predict `approved` (0/1)
├── regression/                predict `income` (a number)
├── clustering/                group rows, ignoring `approved` entirely
├── dimensionality_reduction/   compress the 5 feature columns to 2D
├── ensemble_methods/          the "combine multiple models" idea
└── neural_networks/           perceptron -> MLP, plus a CNN/RNN note
```

## classification/ -- predict `approved`

| File | Algorithm |
|---|---|
| `01_knn.py` | K-Nearest Neighbors -- classify by majority vote among the closest training points |
| `02_naive_bayes.py` | Gaussian Naive Bayes -- classify via per-class, per-feature Gaussian probability, assuming feature independence |
| `03_decision_tree.py` | Decision Tree -- learn a series of if/else feature-threshold splits |
| `04_logistic_regression.py` | Logistic Regression -- linear weighted sum of features, squashed through a sigmoid into a probability |
| `05_svm.py` | Support Vector Machine -- find the maximum-margin separating hyperplane, defined by the closest "support vector" points |
| `06_random_forest.py` | Random Forest -- bagged ensemble of decision trees, each trained on a bootstrap resample with random feature subsets |
| `07_xgboost.py` | XGBoost -- gradient-boosted decision trees, each new tree fit to correct the ensemble's current errors |
| `08_neural_network_mlp.py` | MLP (sklearn) -- a small feedforward neural network trained via backpropagation |

## regression/ -- predict `income`

| File | Algorithm |
|---|---|
| `01_linear_regression.py` | Linear Regression -- fit the best straight-line/hyperplane relationship by minimizing squared error |
| `02_ridge_lasso.py` | Ridge (L2) & Lasso (L1) -- linear regression with a penalty on large coefficients, comparing both side by side |
| `03_decision_tree_regressor.py` | Decision Tree Regressor -- split rules whose leaves predict an averaged numeric value |
| `04_random_forest_regressor.py` | Random Forest Regressor -- bagged ensemble of regression trees |
| `05_svr.py` | Support Vector Regression -- fit within an epsilon-tolerance tube, ignoring points that already fall inside it |
| `06_xgboost_regressor.py` | XGBoost Regressor -- gradient-boosted trees minimizing squared error |

## clustering/ -- group rows without using the `approved` label

| File | Algorithm |
|---|---|
| `01_kmeans.py` | K-Means -- iteratively assign points to the nearest of K centroids, then recompute centroids |
| `02_hierarchical_clustering.py` | Agglomerative Clustering -- repeatedly merge the two closest clusters, building a dendrogram bottom-up |
| `03_dbscan.py` | DBSCAN -- group points by neighborhood density, marking sparse points as noise (genuinely degenerates here: 2 points can never satisfy a minimum-density threshold, explained in-script) |
| `04_gaussian_mixture.py` | Gaussian Mixture Model -- soft-cluster by fitting each cluster as its own Gaussian distribution |

## dimensionality_reduction/ -- compress the 5 feature columns to 2D

| File | Algorithm |
|---|---|
| `01_pca.py` | PCA -- find variance-maximizing orthogonal axes (unsupervised) |
| `02_lda.py` | LDA -- find the axis that best separates classes (**supervised** -- uses the `approved` label, unlike the other three in this folder; genuinely fails to fit here since sklearn requires more samples than classes, explained in-script) |
| `03_tsne.py` | t-SNE -- embed points so that local neighborhood relationships are preserved (unsupervised) |
| `04_autoencoder.py` | Autoencoder (PyTorch) -- a tiny neural net trained to reconstruct its input through a 2-unit bottleneck, whose bottleneck activations become the 2D representation |

## ensemble_methods/ -- the "combine multiple models" idea

| File | Algorithm |
|---|---|
| `01_bagging.py` | Bagging -- average many copies of the same model, each trained on a random bootstrap resample |
| `02_adaboost.py` | AdaBoost -- train models sequentially, reweighting misclassified points so later models focus on them |
| `03_gradient_boosting.py` | Gradient Boosting -- train models sequentially, each fit to the loss gradient (residual error) of the ensemble so far |
| `04_stacking.py` | Stacking -- train a meta-model on top of several base models' predictions (surfaces a real small-data failure mode here: leave-one-out cross-fitting with only 2 rows teaches the meta-model an inverted relationship, explained in-script) |

## neural_networks/

| File | Contents |
|---|---|
| `01_single_perceptron.py` | Hand-rolled in plain numpy (no sklearn) -- `sigmoid(w . x + b)` with manually-coded forward pass, gradient, and update rule |
| `02_multilayer_perceptron.py` | MLP via sklearn -- stacks multiple perceptron-style layers for nonlinear decision boundaries |
| `03_note_on_cnn_rnn.md` | Explains why CNNs (need grid/image data) and RNNs (need sequential data) aren't demoed here -- this dataset is single-snapshot tabular data with no spatial or temporal structure, and faking a demo would teach the wrong intuition |

## A note on how the "only 2 rows" constraint was handled

Several scripts hit real algorithmic limits at n=2 (DBSCAN's density
threshold, LDA's sample-vs-class-count requirement, PCA/t-SNE's rank
and perplexity limits, XGBoost's default split threshold, stacking's
cross-validation needing enough data to be representative). In every
case the script runs, prints what actually happens, and explains *why*
in a comment or printed note -- hyperparameters were adjusted only far
enough to avoid a crash (e.g. DBSCAN's `min_samples`, t-SNE's
`perplexity`), never to manufacture a misleadingly clean result, and no
additional data rows were invented anywhere.
