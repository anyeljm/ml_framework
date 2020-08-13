from sklearn import ensemble
from xgboost import XGBClassifier

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators=300, min_samples_split=50, max_depth=5) ,
    'xgboost' : XGBClassifier(eta=0.3, max_depth=3, objective='multi:softprob', num_class=3, n_estimators=250)
}