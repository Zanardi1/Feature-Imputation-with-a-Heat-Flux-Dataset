# Competitie Kaggle.
# Link-uri: https://www.kaggle.com/competitions/playground-series-s3e15/overview
# https://www.kaggle.com/datasets/saurabhshahane/predicting-heat-flux
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fancyimpute import IterativeImputer
from seaborn import heatmap
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

training_data = pd.read_csv('Data\\data.csv', index_col=0)
ans = pd.read_csv('Data\\sample_submission.csv', index_col=0)
test_data = training_data.loc[training_data.index.isin(ans.index)]
test_data = test_data.drop(columns=['x_e_out [-]']).reset_index(drop=True)
training_data = training_data.drop(index=ans.index).reset_index(drop=True)

training_data.hist()
plt.show()

training_data['chf_exp [MW/m2]'] = np.log(training_data['chf_exp [MW/m2]'])
training_data['D_e [mm]'] = np.log(training_data['D_e [mm]'])
training_data['D_h [mm]'] = np.log(training_data['D_h [mm]'])
training_data['length [mm]'] = np.log(training_data['length [mm]'])

heatmap(data=training_data[training_data.columns[2:]].corr(), annot=True)
plt.show()

training_data.loc[training_data['D_e [mm]'].isna(), 'D_e [mm]'] = training_data.loc[
    training_data['D_e [mm]'].isna(), 'D_h [mm]']
training_data.loc[training_data['D_h [mm]'].isna(), 'D_h [mm]'] = training_data.loc[
    training_data['D_h [mm]'].isna(), 'D_e [mm]']

test_data['chf_exp [MW/m2]'] = np.log(test_data['chf_exp [MW/m2]'])
test_data['D_e [mm]'] = np.log(test_data['D_e [mm]'])
test_data['D_h [mm]'] = np.log(test_data['D_h [mm]'])
test_data['length [mm]'] = np.log(test_data['length [mm]'])

test_data.loc[test_data['D_e [mm]'].isna(), 'D_e [mm]'] = test_data.loc[
    test_data['D_e [mm]'].isna(), 'D_h [mm]']
test_data.loc[test_data['D_h [mm]'].isna(), 'D_h [mm]'] = test_data.loc[
    test_data['D_h [mm]'].isna(), 'D_e [mm]']

print(training_data['author'].value_counts())
print(training_data['geometry'].value_counts())

training_data['author'] = training_data['author'].fillna('Thompson')
training_data['geometry'] = training_data['geometry'].fillna('tube')

test_data['author'] = test_data['author'].fillna('Thompson')
test_data['geometry'] = test_data['geometry'].fillna('tube')

training_data = pd.get_dummies(data=training_data, columns=['author', 'geometry'], drop_first=True)
test_data = pd.get_dummies(data=test_data, columns=['author', 'geometry'], drop_first=True)

imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=6, p=2, metric='euclidean'), max_iter=20)
training_data = pd.DataFrame(data=imputer.fit_transform(training_data), columns=training_data.columns)
test_data = pd.DataFrame(data=imputer.fit_transform(test_data), columns=test_data.columns)

X = pd.DataFrame(data=training_data.drop(columns=['x_e_out [-]']))
y = pd.DataFrame(data=training_data['x_e_out [-]'])

models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(random_state=42),
          AdaBoostRegressor(random_state=42), RandomForestRegressor(random_state=42), BaggingRegressor(random_state=42),
          CatBoostRegressor(random_state=42, verbose=False), Ridge(alpha=1), Lasso(alpha=1),
          SVR(kernel='rbf', C=1, epsilon=0.1), KNeighborsRegressor(n_neighbors=10), BayesianRidge()]
models_name = [LinearRegression().__class__.__name__, KNeighborsRegressor().__class__.__name__,
               DecisionTreeRegressor(random_state=42).__class__.__name__,
               AdaBoostRegressor(random_state=42).__class__.__name__,
               RandomForestRegressor(random_state=42).__class__.__name__,
               BaggingRegressor(random_state=42).__class__.__name__,
               CatBoostRegressor(random_state=42, verbose=False).__class__.__name__,
               Ridge(alpha=1).__class__.__name__,
               Lasso(alpha=1).__class__.__name__, SVR(kernel='rbf', C=1, epsilon=0.1).__class__.__name__,
               KNeighborsRegressor(n_neighbors=10).__class__.__name__, BayesianRidge().__class__.__name__]

score = []
std = []

for model in models:
    try:
        score.append(
            np.mean(cross_val_score(estimator=model, X=X, y=y.values.ravel(), scoring='neg_root_mean_squared_error')))
        std.append(
            np.std(cross_val_score(estimator=model, X=X, y=y.values.ravel(), scoring='neg_root_mean_squared_error')))
        print(model, ': DONE')
    except:
        score.append('Error')
        print(model, ': Error')

tabel = pd.DataFrame(data=models_name)
tabel.columns = ['Models']
tabel['Result'] = score
tabel['Dev'] = std
tabel = tabel.sort_values(by='Result')

print(tabel)

model = CatBoostRegressor(random_state=42, verbose=False)
model.fit(X=X, y=y)
y_calc = model.predict(data=test_data)

ans['x_e_out [-]'] = y_calc
ans.to_csv('submission.csv')
