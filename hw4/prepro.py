from sklearn import preprocessing
import pandas as pd
import numpy as np
import pickle

data_path = './data/STT.csv'
window = 15


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(
        df.volume.values.reshape(-1, 1))
    return df


def split_data(stock, window, percent=0.85):

    amount_of_features = len(stock.columns)  # 5
    data = stock.values
    sequence_length = window + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    row = round(percent * data.shape[0])
    result = np.array(result)
    train = result[:int(row), :]

    x_train = train[:, :-1]
    y_train = np.array(train[:, -1][:, -1])

    x_test = result[int(row):, :-1]
    y_test = np.array(result[int(row):, -1][:, -1])

    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


if __name__ == "__main__":
    df = pd.read_csv(data_path, index_col=0)
    target_df = df[df.symbol == 'STT'].copy()

    target_df.drop(['symbol'], 1, inplace=True)

    target_df_normalized = normalize(target_df)

    x_train, y_train, x_test, y_test = split_data(
        target_df_normalized, window)

    with open('./data/train.pickle', 'wb') as f:
        pickle.dump((x_train, y_train), f)

    with open('./data/test.pickle', 'wb') as f:
        pickle.dump((x_test, y_test), f)
