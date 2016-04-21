# Feature Engineering


Create histograms for categorical variables and group/cluster them.

Use Vowpal Wabbit (vw-varinfo) or XGBoost (XGBfi) to quickly check two-way and three-way interactions.

Use statistical tests to discard noisy features (eg: select k best with ANOVA F-score), Benford's Law to detect natural counts (great for logtransforms).

Manually inspect the data and combine features that look similar in structure (both columns contain hashed variables) or expand categorical variables that look like hierarchical codes ("1.11125A" -> 1, 111, 25, A).

Use (progressive/cross-)validation with fast algorithms or on a subset of the data to spot significant changes.

Compute stats about a row of data (nr. of 0's, nr. of NAs, max, min, mean, std)

Transforms: Log, tfidf

Numerical->Categorical: Encoding numerical variables, like 11.25 as categorical variables: eleventwentyfive

Bayesian: Encode categorical variables with its ratio of the target variable in train set.

Reduce the dimensionality / project down with tSNE, MDA, PCA, LDA, RP, RBM, kmeans or expand raw data.

Genetic programming: http://gplearn.readthedocs.org/en/latest/examples.html#example-2-symbolic-tranformer to automatically create non-linear features.

Recursive Feature Elimination: Use all features -> drop the feature that results in biggest CV gain -> repeat till no improvement

Automation: Try to infer type of feature automatically with scripts and create feature engineering pipelines. This scales and can be used even when the features are not anonymized.

# Tuning xgb
nrounds and eta is the last thing I tune. As your cpu has limitations, I would go for 3-fold CV (but would suggest 5-fold). I usually start with eta=0.05 (or even 0.1), tune optimal subsample/colsample_bytree simultaneously, then go for max_depth. and in the end reduce eta 0.025 or 0.01 and pick nrounds by CV (I have never used eta below 0.01 as it usually has no good cpu ROI :).

# Feature importance
- gini, auc, information value
- lasso, tree

# Ensemble
- cor of submissions  
- weighted averaging
- rank averaging
- stacking (oof predictions)
- blending (predict on valid set, then stack on valid set)
- second layer classifier (using only predictions, no other raw features)
- time-based split of data and predict, then combine the output from different subsets

# Todo
- interaction on top features
- oof target mean for categorical vars
- noise
- 30 features
- single model: KNN, ET, xgb(poisson), tsne, kmeans (level 0)
- apply different subsets of rows and cols to train single model (level 0)
- stacking with different models (level 1)
- weighted averaging (level 2)
