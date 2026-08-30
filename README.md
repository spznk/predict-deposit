# Deposit subscription prediction
This is a Mid term project for ML / DS courses.

Data is taken from kaggle competition:
https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv

The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution.
The classification goal is to predict if the client will subscribe a term deposit (variable y).

## For this task following steps were performed:

1. Exploratory Data Analysis (EDA)
2. Feature encoding
3. Feature engineering
4. Metrics selection
5. Training of different models
6. Models comparison
7. Selection of the best performing model.

## Models used:

1. Logistic Regression
2. k Nearest Neighbours
3. Decision Tree
4. XGBooster
5. LightGBM
6. Autogluon (as a benchmark)

## Comparison table of models evaluation:

| Model | Hyperparameters | AUROC Train | AUROC Test | Recall Train | Recall Test | Comments |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (default) | random_state=987 | 0.7938 | 0.7952 | 0.67 | 0.62 | Basic model, which shows pretty good AUROC values and mediocre recall. Can be improved by changing threshold value |
| Logistic Regression + PolynomialFeatures | degree=5, max_iter=200, random_state=987 | 0.7990 | 0.7992 | 0.69 | 0.62 | Slightly better than the previous one. Also can be improved by moving threshold |
| Logistic Regression + TomekLinks | random_state=987 | 0.7997 | 0.7948 | 0.70 | 0.59 | AUROC is similar to basic model but Recall on Test is worse, so I wouldn't use this model |
| Logistic Regression + Liblinear + L1 | solver='liblinear', l1_ratio=1, random_state=987 | 0.7942 | 0.7954 | 0.66 | 0.62 | Metrics are almost the same as for the basic model. Can be improved by moving threshold |
| Logistic Regression + PolynomialFeatures + Liblinear + L1 | degree=5, max_iter=200, solver='liblinear', l1_ratio=1, random_state=987 | 0.7977 | 0.7984 | 0.68 | 0.64 | Metrics are slightly better than for the basic model. Can be improved by moving threshold |
| kNN (default) | - | 0.9265 | 0.7009 | 0.74 | 0.54 | Metrics values are much better for Train set, which means that this model is overfitted |
| kNN + RandomizedSearchCV | weights='uniform', n_neighbors=199, random_state=987 | 0.8007 | 0.7847 | 0.75 | 0.73 | Pretty good AUROC values, no severe overfit and the best Recall values out of all the models. Good candidate to use as a final model |
| Decision Tree (default) | random_state=987 | 0.9999 | 0.6327 | 1.00 | 0.32 | AUROC value for Train set is almost 1 and Recall value is 1, which means this model is heavily overfitted |
| Decision Tree + RandomizedSearchCV | splitter='best', min_samples_split=7, min_samples_leaf=5, max_leaf_nodes=36, max_depth=33, criterion='entropy', random_state=987 | 0.8076 | 0.7970 | 0.67 | 0.65 | Pretty good metrics values, this model can be used |
| XGBClassifier (default) | random_state=987, eval_metric=auc | 0.9082 | 0.7891 | 0.87 | 0.57 | Metrics values are much better for Train set, which means that this model is overfitted |
| XGBClassifier + RandomizedSearchCV | n_iter=200, random_state=987, scoring='roc_auc' | 0.8323 | 0.8074 | 0.71 | 0.60 | Good AUROC values, mediocre Recall, which can be improved by using loss function related to Recall |
| XGBClassifier + Hyperopt | max_depth=13, min_child_weight=11.8308, learning_rate=0.06499, subsample=0.8244, colsample_bytree=0.8887, gamma=0.9521, reg_alpha=4.9120, reg_lambda=2.7196, n_estimators=1000, early_stopping_rounds=100 | 0.8361 | 0.8093 | 0.71 | 0.63 | Good AUROC values and decent Recall, which can be improved by using loss function related to Recall |
| LGBMClassifier (default) | random_state=987, verbose=-1 | 0.8826 | 0.8017 | 0.78 | 0.63 | Metrics values are much better for Train set, which means that this model is overfitted |
| LGBMClassifier + RandomizedSearchCV | n_iter=200, random_state=987, scoring='roc_auc' | 0.8299 | 0.8100 | 0.72 | 0.64 | Good AUROC values and decent Recall, which can be improved by using loss function related to Recall |
| LGBMClassifier + Hyperopt | learning_rate=0.2618, min_child_samples=15, num_leaves=22, max_depth=13, min_split_gain=0.4344, subsample=0.8686, colsample_bytree=0.8447, reg_alpha=0.00658, reg_lambda=4.1574, n_estimators=1000, subsample_freq=1 | 0.8235 | 0.8117 | 0.70 | 0.64 | Overall the best model. Recall can be improved if needed by business requirements |
| [BENCHMARK] Logistic Regression with 'duration' | random_state=987 | 0.9344 | 0.9352 | 0.68 | 0.67 | This model shows much better results, which is expected using 'duration' feature. Though this model can only be used as a benchmark |
| [BENCHMARK] AutoGluon | presets='high_quality', eval_metric='roc_auc', time_limit=1800 | 0.9772 | 0.9545 | 0.82 | 0.68 | Very high AUROC values and good Recall values. It means there is a potential to create better models manually |

## Model selected
Overall, the best model is **LGBMClassifier with hyperparameters selection using Hyperopt**.
It showed following results:

| Model | Hyperparameters | AUROC Train | AUROC Test | Recall Train | Recall Test |
| --- | --- | --- | --- | --- | --- |
| LGBMClassifier + Hyperopt | learning_rate=0.2618, min_child_samples=15, num_leaves=22, max_depth=13, min_split_gain=0.4344, subsample=0.8686, colsample_bytree=0.8447, reg_alpha=0.00658, reg_lambda=4.1574, n_estimators=1000, subsample_freq=1 | 0.8235 | 0.8117 | 0.70 | 0.64 |

This model has good AUROC values, doesn't show overfit signs. According to importance of finding all the positive predictions, Recall can be improved via introducing loss function bound to Recall value.

## Summary

### What is achieved
- Data was analyzed and new features created
- Trained several models with good results
- Most important features are found

### What can be improved
- Autogluon model showed very good results, which means manual models can be improved
- Data can be analyzed more in details, taking into account the best models feature importances
- Ansemble model can be built using different models which performed best