import pandas as pd
from sklearn import impute



if __name__ == "__main__":
    imputer = impute.KNNImputer(copy=False)
    df_train = pd.read_csv("../input/train.csv")
    train_len = df_train.shape[0]
    df_test = pd.read_csv("../input/test.csv")
    test_len = df_test.shape[0]
    df = pd.concat([df_train, df_test], ignore_index=True)

    for c in df.drop(['id', 'target'], axis=1).columns:
        df.loc[:, c] = df.loc[:, c] 
