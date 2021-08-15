import os
from os.path import exists, join, dirname, realpath
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def manage_missing_numerical_values(train_data, test_data):
    imputer = SimpleImputer()
    num_train_data = train_data.select_dtypes(exclude=['object'])
    num_test_data = test_data.select_dtypes(exclude=['object'])
    imputed_train_data = pd.DataFrame(imputer.fit_transform(num_train_data))
    imputed_test_data = pd.DataFrame(imputer.fit_transform(num_test_data))

    imputed_train_data.columns = num_train_data.columns
    imputed_test_data.columns = num_test_data.columns

    return imputed_train_data, imputed_test_data

def get_data(train_csv_path, test_csv_path):
    train_data = pd.read_csv(join(dirname(realpath(__file__)), train_csv_path))
    test_data = pd.read_csv(join(dirname(realpath(__file__)), test_csv_path))

    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

    train_labels = train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    imputed_train_data, imputed_test_data = manage_missing_numerical_values(train_data, test_data)
    train_data[imputed_train_data.columns] = imputed_train_data
    test_data[imputed_test_data.columns] = imputed_test_data

    # to keep things simple
    cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]
    train_data.drop(cols_with_missing, axis=1, inplace=True)

    test_data.drop(cols_with_missing, axis=1, inplace=True)

    return train_data, test_data, train_labels

def split_train_data(train_data, train_labels):
    train_X, val_X, train_y, val_y = train_test_split(
        train_data, train_labels, random_state=0
    )

    return train_X, val_X, train_y, val_y

def get_categorical_columns(data):
    s = (data.dtypes == 'object')
    return list(s[s].index)


def get_good_and_bad_label_columns(train_data, test_data, categorical_columns):
    good_label_columns = [col for col in categorical_columns if set(test_data[col]).issubset(set(train_data[col]))]
    bad_label_columns = list(set(categorical_columns) - set(good_label_columns))

    return good_label_columns, bad_label_columns

def get_columns_cardinality(data, categorical_cols):
     columns_with_cardinality = {col: data[col].nunique() for col in categorical_cols}
     return list(columns_with_cardinality.items())

def get_low_cardinality_columns(columns_with_cardinality):
    return [col for col, _ in list(filter(lambda x: x[1] < 10, columns_with_cardinality))]

def get_high_cardinality_columns(columns_with_cardinality):
    return [col for col, _ in list(filter(lambda x: x[1] >= 10, columns_with_cardinality))]


def manage_categorical_columns(train_data, test_data):
    categorical_columns = get_categorical_columns(train_data)

    good_label_columns, bad_label_columns = get_good_and_bad_label_columns(train_data, test_data, categorical_columns)
    columns_with_cardinality = get_columns_cardinality(train_data, good_label_columns)

    low_cardinality_columns = get_low_cardinality_columns(columns_with_cardinality)
    high_cardinality_columns = get_high_cardinality_columns(columns_with_cardinality)

    ordinal_encoder = OrdinalEncoder()
    ordinal_train_data = pd.DataFrame(ordinal_encoder.fit_transform(train_data[high_cardinality_columns]))
    ordinal_test_data = pd.DataFrame(ordinal_encoder.transform(test_data[high_cardinality_columns]))

    ordinal_train_data.index = train_data.index
    ordinal_test_data.index = test_data.index

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_train_data = pd.DataFrame(one_hot_encoder.fit_transform(train_data[low_cardinality_columns]))
    oh_test_data = pd.DataFrame(one_hot_encoder.transform(test_data[low_cardinality_columns]))

    oh_train_data.index = train_data.index
    oh_test_data.index = test_data.index

    num_train_data = train_data.drop(categorical_columns, axis=1)
    num_test_data = test_data.drop(categorical_columns, axis=1)

    train_final = pd.concat([num_train_data, oh_train_data, ordinal_train_data], axis=1)
    test_final = pd.concat([num_test_data, oh_test_data, ordinal_test_data], axis=1)

    return train_final, test_final



