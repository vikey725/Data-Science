import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from os.path import exists, join, dirname, realpath


def find_best_model(train_X, val_X, train_y, val_y, n_estimators_list):
    train_X.drop(['Id'], axis=1, inplace=True)
    val_X.drop(['Id'], axis=1, inplace=True)
    best_model_error = float('infinity')
    best_estimator_index = -1
    for idx, n_estimators in enumerate(n_estimators_list):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
        model.fit(train_X, train_y)

        preds = model.predict(val_X)
        error = mean_absolute_error(val_y, preds)
        if error < best_model_error:
            best_model_error = error
            best_estimator_index = idx

    return best_estimator_index

def build_model(train_data, train_labels, n_estimators):
    train_data.drop(["Id"], axis=1, inplace=True)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    model.fit(train_data, train_labels)

    return model

def generate_out_csv(train_data, train_labels, test_data, n_estimators):
    model = build_model(train_data, train_labels, n_estimators)
    ids = test_data.Id
    test_data.drop(["Id"], axis=1, inplace=True)
    preds = model.predict(test_data)

    output = pd.DataFrame({'Id': [int(id) for id in ids],
                           'SalePrice': preds})
    output_file_path = join(dirname(realpath(__file__)), "../data/submission.csv")
    output.to_csv(output_file_path, index=False)


