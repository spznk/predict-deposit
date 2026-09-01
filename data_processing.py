import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_curve, auc
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval


class DataProcessor:
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    input_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    ordered_cols: list[str]
    target_col = 'y'

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    col_transformer: ColumnTransformer
    classifier: BaseEstimator

    # Define X and y
    def set_X_and_y(self):
        self.X_train = self.train_df[self.input_cols].copy()
        self.y_train = self.train_df[self.target_col].copy()
        self.X_test = self.test_df[self.input_cols].copy()
        self.y_test = self.test_df[self.target_col].copy()

    # Class constructor
    def __init__(self, raw_df, columns_to_drop):
        self.input_cols = raw_df.columns.drop(columns_to_drop).to_list()

        # Split to train and test
        train_df, test_df = train_test_split(
            raw_df,
            test_size=0.2,
            stratify=raw_df[self.target_col],
            random_state=15
        )

        # Transform target values to 1 and 0
        target_map = {'yes': 1, 'no': 0}
        train_df[self.target_col] = train_df[self.target_col].map(target_map)
        test_df[self.target_col] = test_df[self.target_col].map(target_map)

        # Set class properties
        self.train_df = train_df
        self.test_df = test_df

        self.set_X_and_y()

    # Function to divide ages into bins
    def age_cat(self, years):
        if years <= 20:
            return '0-20'
        elif years >= 21 and years <= 30:
            return '21-30'
        elif years >= 31 and years <= 35:
            return '31-35'
        elif years >= 36 and years <= 40:
            return '36-40'
        elif years >= 41 and years <= 50:
            return '41-50'
        elif years >= 51 and years <= 60:
            return '51-60'
        elif years >= 61:
            return '61+'

    # Add new column with age ranges
    def add_age_range(self):
        self.X_train['age_range'] = self.X_train['age'].apply(self.age_cat)
        self.X_test['age_range'] = self.X_test['age'].apply(self.age_cat)

    # Add new column for 'campaign' with putting outliers into one bin
    def add_category_for_campaign(self):
        self.X_train['campaign_category'] = [
            str(val)
            if val < 7
            else '7 or more'
            for val in self.X_train['campaign']
        ]
        self.X_test['campaign_category'] = [
            str(val)
            if val < 7
            else '7 or more'
            for val in self.X_test['campaign']
        ]

        campaign_cats = self.X_train['campaign_category'].unique().tolist()
        campaign_cats.sort()
        self.X_train['campaign_category'] = pd.Categorical(
            self.X_train['campaign_category'],
            categories=['0', *campaign_cats],  # added 0 to correspond categories with original values
            ordered=True
        )
        self.X_test['campaign_category'] = pd.Categorical(
            self.X_test['campaign_category'],
            categories=['0', *campaign_cats],
            ordered=True
        )

    # Add new column for 'previous' gathering all values >=2 together
    def add_category_for_previous(self):
        self.X_train['previous_category'] = [
            str(val)
            if val < 2
            else '2 or more'
            for val in self.X_train['previous']
        ]
        self.X_test['previous_category'] = [
            str(val)
            if val < 2
            else '2 or more'
            for val in self.X_test['previous']
        ]

        previous_cats = self.X_train['previous_category'].unique().tolist()
        previous_cats.sort()
        self.X_train['previous_category'] = pd.Categorical(
            self.X_train['previous_category'],
            categories=previous_cats,
            ordered=True
        )
        self.X_test['previous_category'] = pd.Categorical(
            self.X_test['previous_category'],
            categories=previous_cats,
            ordered=True
        )

    # Add new binary column for 'pdays' which eliminates discrepancy
    # between some 'pdays' and 'previous' values
    def add_binary_for_pdays(self):
        self.X_train['pdays_was_contacted'] = (
            ~((self.X_train['pdays'] == 999) & (self.X_train['previous'] == 0))
        ).astype(int)
        self.X_test['pdays_was_contacted'] = (
            ~((self.X_test['pdays'] == 999) & (self.X_test['previous'] == 0))
        ).astype(int)

    # Transform columns with 'oject' type to 'category' type
    def transform_objects_to_categories(self):
        features_to_transform = self.X_train.select_dtypes(include=['object']).columns
        self.X_train[features_to_transform] = self.X_train[features_to_transform].astype('category')
        self.X_test[features_to_transform] = self.X_test[features_to_transform].astype('category')

        # Make sure validation set has the same categories
        for cat in features_to_transform:
            self.X_test[cat] = self.X_test[cat].cat.set_categories(
                self.X_train[cat].cat.categories
            )

    # Redefine input columns and define other types of columns
    def redefine_column_groups(self):
        self.input_cols = self.X_train.columns.to_list()
        self.numeric_cols = self.X_train[self.input_cols].select_dtypes(include='number').columns.to_list()
        self.ordered_cols = ['campaign_category', 'previous_category']
        self.categorical_cols = (
            self.X_train[self.input_cols]
            .select_dtypes(include='category')
            .columns
            .drop(self.ordered_cols)
            .to_list()
        )

    # Call all the functions which add new columns and redefine column groups
    def add_features(self):
        self.add_age_range()
        self.add_category_for_campaign()
        self.add_category_for_previous()
        self.add_binary_for_pdays()
        self.transform_objects_to_categories()
        self.redefine_column_groups()

    # Create column transformer with possibility to customize
    # and transform columns in Train and Test
    def transform_columns(self, add_features=True, num_preprocessor=None, cat_preprocessor=None, ord_preprocessor=None):
        if add_features:
            self.add_features()

        if num_preprocessor:
            num_preprocessor = num_preprocessor
        else:
            num_preprocessor = Pipeline(steps=[
                ('scaler', MinMaxScaler())
            ])

        if cat_preprocessor:
            cat_preprocessor = cat_preprocessor
        else:
            cat_preprocessor = Pipeline(steps=[
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])

        if ord_preprocessor:
            ord_preprocessor = ord_preprocessor
        else:
            ord_preprocessor = Pipeline(steps=[
                ('encoder', OrdinalEncoder())
            ])

        self.col_transformer = ColumnTransformer(transformers=[
            ('num', num_preprocessor, self.numeric_cols),
            ('cat', cat_preprocessor, self.categorical_cols),
            ('ord', ord_preprocessor, self.ordered_cols)
        ]).set_output(transform='pandas')

        self.X_train = self.col_transformer.fit_transform(self.X_train)
        self.X_test = self.col_transformer.transform(self.X_test)

    # Calculate AUROC and optionally show plot with ROC curve
    @staticmethod
    def get_auroc(target, probas, show_plot=True):
        if type(probas) is pd.DataFrame:
            fpr, tpr, thresholds = roc_curve(target, probas['yes'], pos_label='yes')
        else:
            fpr, tpr, thresholds = roc_curve(target, probas[:, 1])

        if show_plot:
            plt.plot(fpr, tpr)
            plt.xlabel('fpr')
            plt.ylabel('tpr')
            plt.title('ROC Curve')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.show()

        auroc = auc(fpr, tpr)

        return auroc

    # Fit classifier, make predictions
    # and show classification report and AUROC
    def predict_and_show_metrics(self, classifier, **fit_params):
        self.classifier = classifier

        classifier.fit(self.X_train, self.y_train, **fit_params)

        train_preds = classifier.predict(self.X_train)
        test_preds = classifier.predict(self.X_test)

        train_probas = classifier.predict_proba(self.X_train)
        test_probas = classifier.predict_proba(self.X_test)

        print('TRAIN')
        print(classification_report(train_preds, self.y_train))
        print('AUROC:', DataProcessor.get_auroc(self.y_train, train_probas))

        print('\n------------------------------------------------------\n')
        print('TEST')
        print(classification_report(test_preds, self.y_test))
        print('AUROC:', DataProcessor.get_auroc(self.y_test, test_probas))

    # Objective function for optimization of Booster models
    @staticmethod
    def objective(model_type, params, X_train, y_train, X_test, y_test):
        match model_type:
            case 'xgb':
                clf = XGBClassifier(
                    max_depth=params['max_depth'],
                    min_child_weight=params['min_child_weight'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    gamma=params['gamma'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    n_estimators=1000,
                    early_stopping_rounds=100,
                    random_state=987
                )
                clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            case 'lgb':
                clf = LGBMClassifier(
                    learning_rate=params['learning_rate'],
                    min_child_samples=params['min_child_samples'],
                    num_leaves=params['num_leaves'],
                    max_depth=params['max_depth'],
                    min_split_gain=params['min_split_gain'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    n_estimators=1000,
                    subsample_freq=1,
                    random_state=987,
                    verbose=-1
                  )
                clf.fit(X_train, y_train, eval_X=X_test, eval_y=y_test,
                        callbacks=[early_stopping(stopping_rounds=100, verbose=False)])

            case _:
                raise ValueError("Undefined model. Available values are: 'xgb', 'lgb'.")

        probas = clf.predict_proba(X_test)

        auroc = DataProcessor.get_auroc(y_test, probas, show_plot=False)

        return {'loss': -auroc, 'status': STATUS_OK}

    # Search for the best parameters
    @staticmethod
    def search_best_params(model_type, param_space, X_train, y_train, X_test, y_test):
        trials = Trials()
        match model_type:
            case 'xgb':
                best_params = fmin(
                    fn=lambda params: DataProcessor.objective(
                        model_type, params,
                        X_train, y_train,
                        X_test, y_test
                    ),
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=200,
                    trials=trials,
                    rstate=np.random.default_rng(2026)
                )
            case 'lgb':
                best_params = fmin(
                    fn=lambda params: DataProcessor.objective(
                        model_type, params,
                        X_train, y_train,
                        X_test, y_test
                    ),
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=200,
                    trials=trials,
                    rstate=np.random.default_rng(2026), 
                )
            case _:
                raise ValueError("Undefined model. Available values are: 'xgb', 'lgb'.")

        best_params = space_eval(param_space, best_params)

        return best_params