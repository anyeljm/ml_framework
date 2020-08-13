import pandas as pd
from sklearn import model_selection

def createFold(target_name):
    df = pd.read_csv("../input/train_encode.csv")
    df['kfold'] = -1
    df = df.rename(columns={target_name:'target'})

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv("../input/train_fold.csv", index=False)


if __name__ == "__main__":
    createFold(target_name='target')

    