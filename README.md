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

# xgb parameters
Gamma is a pseudo-regularization hyperparameter of gradient boosting, it can prove very useful when you drop many correlated+predictive features, but also when too many strong and predictive features are present and cannot be handled properly using colsample_bytree.

For instance, when two features are nearly identical and dominate the whole train set as predictors, a high gamma will not allow both to remain in a single tree as you will need a minimum loss reduction (according to the loss functions you are using). If the minimum loss reduction is not reached (because they are nearly doing the same thing), only one will subsist and other variables will be taken in consideration (while in the contrary it would pick the heavy predictor a second time).

Extremely powerful (and much more than min_child_weight) when it comes to constructing very deep trees without overfitting at a godlike speed (using min_child_weight you are looking at the hessian (derivative of the gradient) unlike the direct loss function for gamma). It can also push xgboost beyond the typical local performance maxima, but it's also a very sensible parameter (a 0.1 difference in gamma can mean a lot, if not a "disaster back-to-start performance untuned" depending on the data set).

## What's Gamma?
The range of that parameter is [0, Infinite[. Finding a "good" gamma is very dependent on both your data set and the other parameters you are using. There is no optimal gamma for a data set, there is only an optimal (real-valued) gamma depending on both the training set + the other parameters you are using.

## Gamma is dependent on both the training set and the other parameters you use.
There is no "good Gamma" for any data set alone
It is a pseudo-regularization hyperparameter in gradient boosting.
Mathematically you call "Gamma" the "Lagrangian multiplier" (complexity control).
The higher Gamma is, the higher the regularization. Default value is 0 (no regularization).
Gamma values around 20 are extremely high, and should be used only when you are using high depth (i.e overfitting blazing fast, not letting the variance/bias tradeoff stabilize for a local optimum) or if you want to control the directly the features which are dominating in the data set (i.e too strong feature engineering).
Gamma Tuning
Always start with 0, use xgb.cv, and look how the train/test are faring. If you train CV skyrocketing over test CV at a blazing speed, this is where Gamma is useful instead of min_child_weight (because you need to control the complexity issued from the loss, not the loss derivative from the hessian weight in min_child_weight). Another choice typical and most preferred choice: step max_depth down :)
If Gamma is useful (i.e train CV skyrockets at godlike speed when test CV can't follow), crank up Gamma. This is where the experience with tuning Gamma is useful (so you lose the lowest amount of time). Depending on what you see between the train/test CV increase speed, you try to find an appropriate Gamma. The higher the Gamma, the lower the difference between train/test CV will happen. If you have no idea of the value to use, put 10 and look what happens.
How to set Gamma values?
If your train/test CV are always lying too close, it means you controlled way too much the complexity of xgboost, and the model can't grow trees without pruning them (due to the loss threshold not reached thanks to Gamma). Lower Gamma (good relative value to reduce if you don't know: cut 20% of Gamma away until you test CV grows without having the train CV frozen).
If your train/test CV are differing too much, it means you did not control enough the complexity of xgboost, and the model grows too many trees without pruning them (due to the loss threshold not reached because of Gamma). Put a higher Gamma (good absolute value to use if you don't know: +2, until your test CV can follow faster your train CV which goes slower, your test CV should be able to peak).
If your train CV is stuck (not increasing, or increasing way too slowly), decrease Gamma: that value was too high and xgboost keeps pruning trees until it can find something appropriate (or it may end in an endless loop of testing + adding nodes but pruning them straight away...).
## Need TIPS about how to tune perfectly Gamma
Tuning Gamma should result in something very close to a U-shaped CV :) - this is not exactly true due to potential differences in the folds, but you should get approximately a U-shaped CV if you were to plot (Gamma, Performance Metric). From there, you know when to minimize and when to maximize :) (and with your experience too!)

## Test yourself
With high depth such as 15 in this data set, you can train yourself using Gamma. You should be able with the following settings to get at least 0.841:

## 4-fold cross-validation
subsample = 0.70
colsample_bytree = ~0.70 (tune this if you don't manage to get 0.841 by tuning Gamma)
max_depth = 10
nrounds = 100000 (use early.stop.round = 50)
eta = 0.05
In case you get a bad fold set, set yourself the seed for folds, and set your own benchmark using max_depth = 5 (which was "the best" found).

At the end, you should be able to push locally by 0.0002 more than the typical "best" found parameters using an appropriate depth. Unfortunately, a Gamma value for a specific max_depth does NOT work the same with a different max_depth. This is also true for all other parameters used.

What to optimize first? Gamma or Depth? What's up with min_child_weight?
It is your choice. Using Gamma will always yield a higher performance than not using Gamma, as long as you found the best set of parameters for Gamma to shine. This is due to the ability to prune a shallow tree using the loss function instead of using the hessian weight (gradient derivative).

Controlling the loss function? (Gamma) => you are the first controller to force pruning of the pure weights! (full momentum)

Controlling the hessian weights? (min_child_weight) => you are the second controller to force pruning using derivatives! (0 momentum)

## When to use Gamma?
Easy question: when you want to use shallow trees because you expect them to do better. Very good hyperparameter also for ensembling / dealing with heavy dominating group of features, much better than min_child_weight.

## What to remember? Too much information! Need TL;DR
If you need to resume what is min_child_weight: the knob which tunes the soft performance difference between the overfitting set (train) and a (potential) test set (minimizes the difference => locally blocking potential interactions at the expense of potentially higher rounds and lower OR better performance).

If you need to resume what is Gamma: the knob which fine-tunes the hard performance difference between the overfitting set (train) and a (potential) test set (minimizes both the difference and the speed at which it is accrued => give more rounds to train at the expense of being stuck at a local minima for the train set, by blocking generalized strong interactions which gives no appropriate gain).

If you need to resume what is Depth: the knob which tunes "roughly" the hard performance difference between the overfitting set (train) and a (potential) test set (maximizes only the speed at which it is accrued => give room for more generalized potential interactions at the expense of less rounds).

Understand by "performance" the word "complexity", i.e how complex (overfitting) a model is, but also how good the complexity for your model is when measured using quantitative measures.

## TL;DR S version
If you understood the four sentences higher ^, you can now understand why tuning Gamma is dependent on all the other hyperparameters you are using, but also the only reasons you should tune Gamma:

Very High depth => high Gamma (like 3? 5? 10? 20? even more?)
Typical depths where you have good CV values => low Gamma (like 0.01? 0.1? going over 1 is useless, you probably badly tuned something else or use the wrong depth!)
I STILL CAN'T GET IT BETWEEN GAMMA AND MIN_CHILD_WEIGHT!!!
Take the following example: you sleep in a room during night, and you need to wake up at a specific time (but you don't know when you will wake up yourself!!!). You know the dependent features of "when I wake up" are: noise, time, cars. Noise is made of 1000 other features.

If you tune Gamma, you will tune how much you can take from these 1000 features in a globalized fashion. For instance, you won't take all immediately, but you will take them slowly. XGBoost will discard most of them, but NOT all everytime :)
If you tune min_child_weight, you will tune what interactions you allow in a localized fashion. For instance, if the interaction between the 1000 "other features" and the features xgboost is trying to use is too low (at 0 momentum, the weight given to the interaction using time as weight), the interaction is discarded (pruned) everytime. :)
That's over-simplified, but it is close to be like that. Remember also that "local" means "dependent on the previous nodes", so a node that should not exist may exist if the previous nodes are allowing it :)

=======

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

