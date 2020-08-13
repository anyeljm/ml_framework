import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, metrics, model_selection
import xgboost as xgb
import sys

import dispatcher
import joblib
from gridSearch import param_grid

from categorical_features import  CategoricalFeatures
from create_folds import createFold

folds = [0, 1, 2, 3, 4]

mdl = 'xgboost'

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df_train = pd.read_csv("../input/train.csv")
    train_len = df_train.shape[0]
    df_test = pd.read_csv("../input/test.csv")
    test_len = df_test.shape[0]
    df = pd.concat([df_train, df_test], ignore_index=True)

    label_encoders = []
    one_hot = []
    drops = []
    for c in df.drop(['id','target'], axis=1).columns:#c=df.drop(['id','target'], axis=1).columns[0]
        if df[c].nunique()==1:
            drops.append(c)
        elif df[c].nunique()==2:
            label_encoders.append(c)
        elif df[c].unique()<=12:
            one_hot.append(c)
        else:
            label_encoders.append(c)

    df = df.drop(drops, axis=1).copy()
    df_encode = CategoricalFeatures(df=df, 
                                    categorical_features=label_encoders, 
                                    encoding_type='label')
    df_lbl_encode_ = df_encode.processing()
        
    df_encode = CategoricalFeatures(df=df_lbl_encode_, 
                                    categorical_features=one_hot, 
                                    encoding_type='ohe')
    df_encode_ = df_encode.processing()

    train_df_encode_ = df_encode_.loc[:train_len-1]
    test_df_encode_ = df_encode_.loc[train_len:]

    train_df_encode_.to_csv("../input/train_encode.csv", index=False)
    test_df_encode_.to_csv("../input/test_encode.csv", index=False)

    createFold(target_name='target')

    df = pd.read_csv("../input/train_fold.csv")
    for fold in folds:#fold=0
        train_df = df[df.kfold.isin(FOLD_MAPPPING.get(fold))].reset_index(drop=True)
        valid_df = df[df.kfold==fold].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(['id', 'kfold', 'target'], axis=1)
        valid_df = valid_df.drop(['id', 'kfold', 'target'], axis=1)

        model = dispatcher.MODELS[mdl]
        model.fit(train_df, ytrain)
        preds = model.predict(valid_df)

        preds = model.predict_proba(valid_df)[:, 1]
        auc_score = metrics.roc_auc_score(y_score=preds, y_true=yvalid)

        joblib.dump(model, f"../models/{mdl}_{fold}_model.pkl")
        joblib.dump(train_df.columns, f"../models/{mdl}_{fold}_columns.pkl")




            





