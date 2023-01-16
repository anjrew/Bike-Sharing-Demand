from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, ElasticNet,PoissonRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import feature_engineering_functions as fe
from create_model import create
import pandas as pd
import numpy as np
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(f'{dir_path}/data/train.csv', index_col=0, parse_dates=True)

target = 'count'

rm = []
drop_features = ['casual', 'registered' ] + rm
numeric_values = ['atemp',	'humidity',	'windspeed', 'temp', 'hour', 'year']
cat_features = ['season', 'weather', 'month', 'day_of_week']


num_scaler = MinMaxScaler()
ohe = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')
poly = PolynomialFeatures(degree=3, include_bias=False)

model = None
best_average = None
best_depth = 15
best_estimators = 5
    
m = RandomForestRegressor(max_depth=15, n_estimators=87, n_jobs=-1)

result = create(m, df.copy(), numeric_values, drop_features, cat_features,
                    target, num_scaler, ohe, poly, test_size=0.2, rnd_state=43)
model = result['model']

print('\nThe result was :', result)

print('\nbest_average', best_average, 'best_depth',best_depth, 'best_depth',best_estimators)

df_test = pd.read_csv(f'{dir_path}/data/test.csv', index_col=0,  parse_dates=True)


x = df.drop(drop_features, axis=1).drop(target, axis=1)
y = df[target]
test_train = fe.engineer_df(x.copy(), numeric_values, cat_features, num_scaler, poly, ohe, False)

m.fit(test_train, np.ravel(y))

engineered_test_data = fe.add_time_columns(df_test.drop(rm, axis=1))

engineered_test_data = fe.transform_features(
    engineered_test_data, numeric_values, num_scaler, fit=False)

engineered_test_data = fe.transform_features(
    engineered_test_data, numeric_values, poly, fit=False)

engineered_test_data = fe.transform_features(
    engineered_test_data, cat_features, ohe, fit=True)
predictions = m.predict(engineered_test_data)

predictions = np.round(predictions, decimals=0)

predictions = np.where(predictions < 0, 0, predictions)

submission = pd.DataFrame(
    {'datetime': engineered_test_data.index.values, 'count': predictions.astype(int)})

# This is saved in the same directory as your notebook
filename = f'{dir_path}/artifacts/bike_sharing_demand_submission.csv'

submission.to_csv(filename, index=False)

print('Saved file: ' + filename)
