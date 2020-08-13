import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
import joblib, os
import numpy as np

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    predictions = None

    for fold in range(5):#fold=0
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{fold}_columns.pkl"))
        
        model = joblib.load(os.path.join(model_path, f"{model_type}_{fold}_model.pkl"))
        
        df = df[cols]
        preds = model.predict_proba(df)[:, 1]

        if fold == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub    

if __name__ == "__main__":
    submission = predict(test_data_path="../input/test_encode.csv",
                         model_type="xgboost",
                         model_path="../models/")
    submission.loc[:, 'id'] = submission.loc[:, 'id'].astype(int)
    submission.to_csv("../models/submission_xgboost.csv", index=False)
