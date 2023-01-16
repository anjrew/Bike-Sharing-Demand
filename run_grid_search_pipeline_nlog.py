from functools import reduce
import os
import numpy as np
import pandas as pd
import feature_engineering_functions as fe
import submission_functions as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import, Ridge,  LinearRegression, SGDRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from statistics import mean


num_scaler = MinMaxScaler()
ohe = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')
poly = PolynomialFeatures(degree=2, include_bias=False)


dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(f'{dir_path}/data/train.csv', index_col=0, parse_dates=True)

target = 'count'


feature_configurations = [

        {
        'id': 'remove_rush_hour_and_circulars_not_cos_hour',
        'rm': ['is_rush_hour', 'hour_sin', 'month_sin','month_cos','interaction_work_rush'],
        'drop_features': ['casual', 'registered'],
        'numeric_values': ['atemp',	'humidity',	'windspeed', 'temp', 'hour', 'year', 'cos_hour'],
        'cat_features': ['season', 'weather', 'month', 'day_of_week', 'time_of_day']
    },
]

feature_results = []

for feature_config in feature_configurations:

    X = df.drop(feature_config['drop_features'], axis=1).drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)
    
    # x_train Alone
    X_train = fe.engineer_df(X_train, feature_config['numeric_values'],
                             feature_config['cat_features'], num_scaler, poly, ohe, True, feature_config['rm'])

    # x_test Alone
    X_test = fe.engineer_df(X_test, feature_config['numeric_values'],
                            feature_config['cat_features'], num_scaler, poly, ohe, False, feature_config['rm'])

    grid_searches = [

     GridSearchCV(LinearRegression(), {
            'fit_intercept': [False],
            'normalize': [True],
            'n_jobs': [4]
            }, cv=5, verbose=3),

        GridSearchCV(RandomForestRegressor(), {
            'max_depth': [22 ],
            'n_estimators': [95],
            'n_jobs': [4]
        }, cv=5, verbose=3),

        GridSearchCV(Ridge(), {
            'fit_intercept': [False],
            'normalize': [True],
            'solver': ['auto']
            }, cv=5, verbose=3),

        GridSearchCV(SGDRegressor(), {}, cv=5, verbose=3),

        GridSearchCV(SVR(), {}, cv=5, verbose=3),

        GridSearchCV(BayesianRidge(), {}, cv=5, verbose=3),

        GridSearchCV(GradientBoostingRegressor(), {'learning_rate': [0.05]}, cv=5, verbose=3),

        GridSearchCV(XGBRegressor(), {
            'booster': ['gbtree'],
            'nthread':[4],
            'max_depth': [5]
        }, cv=5, verbose=3
        ),
    ]

    grid_results = []

    for grid in grid_searches:
        grid.fit(X_train, y_train)

        result = {
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_,
            'grid':  grid,
            'best_score': grid.best_score_,
            'test_score': grid.score(X_test, y_test),
            # 'rmse': np.sqrt(mean_squared_log_error(y,  grid.score(X_test, y_test-1)))
        }

        print(f'\n Grid tested', result)

        grid_results.append(result)

    print('\n The grid results are:', grid_results)

    df_test = pd.read_csv(f'{dir_path}/data/test.csv',index_col=0,  parse_dates=True)

    train_df = fe.engineer_df(
        X, feature_config['numeric_values'], feature_config['cat_features'], num_scaler, poly, ohe, True, feature_config['rm'])

    test_df = fe.engineer_df(df_test.copy(), feature_config['numeric_values'],
                             feature_config['cat_features'], num_scaler, poly, ohe, False, feature_config['rm'])
    y = np.log1p(df[target])

    for result in grid_results:
        if(result['test_score'] > 0.80):
            model = result['grid'].estimator
            model.fit(train_df, np.ravel(y))

            predictions = np.exp(model.predict(test_df))-1

            predictions = np.round(predictions, decimals=0)

            predictions = np.where(predictions < 0, 0, predictions)

            model_name = result['grid'].best_estimator_.__class__.__name__

            submission = pd.DataFrame(
                {
                    'datetime': test_df.index.values, 
                    f'{model_name}_count': predictions.astype(int)
                }
                )

            result['submission_df'] = submission

            filename = f'{dir_path}/artifacts/bike_sharing_demand_{model_name}_submission.csv'

            submission.to_csv(filename, index=False)

    feature_results.append(
        {
            'feature_config': feature_config,
            'grid_results': grid_results
        }
    )

best_feature_result = None

print('\n The feature results are', feature_results)

for feature_result in feature_results:

    grid_results = feature_result['grid_results']
    scores = list(map(lambda gr: gr['best_score'], grid_results))
    average_score = mean(scores)

    feature_result['score']= average_score

    if best_feature_result is None or best_feature_result.get('score') is None or best_feature_result.get('score') < average_score:
        best_feature_result = feature_result

print('\n The Best results are',)
print(best_feature_result['feature_config'])

grid_results = best_feature_result['grid_results']

good_enough = list(filter(lambda res: res.get('submission_df') is not None, grid_results))

for result in good_enough:
    print('best_estimator',result['best_estimator'], 'score', result['best_score'])

submissions = list(map(lambda res: res['submission_df'], good_enough))

print('\n The submissions are', submission)

composed_df: pd.DataFrame = reduce(lambda x, y: x.merge(y), submissions)

composed_df = composed_df.set_index('datetime')

voted_df = composed_df.apply(sm.get_mode_or_mean, axis=1)

final_df = voted_df[['count']]


# This is saved in the same directory as your notebook
filename = f'{dir_path}/artifacts/bike_sharing_demand_composed_submission.csv'

final_df.to_csv(filename, index=True)

print('Saved file: ' + filename)
