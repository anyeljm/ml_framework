import pandas as pd
from sklearn import preprocessing

class CategoricalFeatures():
    """
    df: pandas DataFrame 
    categorical_features: list of columns to encode
    encoding_type: str -> 'label' or 'ohe'
    """
    def __init__(self, df, categorical_features, encoding_type):
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type

        for c in self.df.columns:
            self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("None")

    def processing(self):
        if self.enc_type == 'label':
            for c in self.cat_feats:
                lbl = preprocessing.LabelEncoder()
                self.df.loc[:, c] = lbl.fit_transform(self.df.loc[:, c].values)
            return self.df

        if self.enc_type == 'ohe':
            for c in self.cat_feats:                
                ohe = preprocessing.OneHotEncoder()
                array = pd.DataFrame(ohe.fit_transform(self.df.loc[:, c].values.reshape(-1,1)).toarray())
                self.df = pd.concat([self.df.drop(c, axis=1), array], axis=1)
            return self.df
        else:
            raise Exception(f"Enconding type not implemented: {self.enc_type}")
