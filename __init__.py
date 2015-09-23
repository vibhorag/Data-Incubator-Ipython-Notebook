from __future__ import absolute_import

import toolz
import imp
import typecheck
import fellow
from .data import test_json
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import gzip
import ujson
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator,RegressorMixin,TransformerMixin
import dill
imp.load_source('FeatureSelection_CityEstimator','./ml/FeatureSelection_CityEstimator.py')
from FeatureSelection_CityEstimator import LatLongModelTransformer
from FeatureSelection_CityEstimator import CategoryModelTransformer
from FeatureSelection_CityEstimator import AttrKNNModelTransformer

attr_vectorizer = dill.load(open("./ml/vector-attr.pkl"))
attr_tfidf =      dill.load(open("./ml/tfidf-attr.pkl"))
attri_linear =    dill.load(open("./ml/linear-latt.pkl"))

full = dill.load(open("./ml/featureunion.pkl"))
linearfitmodel = dill.load(open("./ml/linearfit-model.pkl"))


##Nested Dictioanry function

def dfs_selector(attrbs):
    res = {}
    for att in attrbs:
        if type(attrbs[att]) in [int,float,bool]:
            res[att] = attrbs[att]
        elif type(attrbs[att]) ==  dict:
            tmp = dfs_selector(attrbs[att])
            for key in tmp:
                res[att + '_' + key] = tmp[key]
        elif type(attrbs[att]) in [str, unicode]:
            res[att + '_' + attrbs[att]] = 1
        else:
            print '\nsomethingwrong'
    return res

catmodel = dill.load(open("./ml/dill-categories.pkl"))
dict_vectorizer = dill.load(open("./ml/dict-vectorizer_dill.pkl"))
tdidf_transformer = dill.load(open("./ml/tdidf-vectorizer_dill.pkl"))



def pick(whitelist, dicts):
    return [toolz.keyfilter(lambda k: k in whitelist, d)
            for d in dicts]

def exclude(blacklist, dicts):
    return [toolz.keyfilter(lambda k: k not in blacklist, d)
            for d in dicts]


@fellow.batch(name="ml.city_model")
@typecheck.test_cases(record=pick({"city"}, test_json))
@typecheck.returns("number")
def city_model(record):
    import numpy as np
    with open("./ml/citymodel-dill.pkl","rb") as citystr:
         citymodel = dill.load(citystr)
    y_pred = citymodel.predict(record['city'])
    #pred = citymodel.predict([record["city"]])
    return y_pred


@fellow.batch(name="ml.lat_long_model")
@typecheck.test_cases(record=pick({"longitude", "latitude"}, test_json))
@typecheck.returns("number")
def lat_long_model(record):
    class ColumnSectionTransformer(BaseEstimator,TransformerMixin):
        def __init__(self,keys):
            self.keys = keys

        def fit(self,X,y=None):
            return self

        def transform(self,X):
            import pandas as pd
            X_trans = pd.DataFrame(columns=(self.keys))
            cnt = len(X)
            for i in xrange(cnt):
                X_trans.loc[i] = [X[i][key] for key in self.keys]
            return X_trans

    with open('./ml/lat_long_model.pkl', 'rb') as in_strm:
        latlongmodel = dill.load(in_strm)

    transformer = ColumnSectionTransformer(keys=(u'latitude',u'longitude'))


    record_trans = transformer.fit_transform([record])
    res = float(latlongmodel.predict(record_trans))
    return res


@fellow.batch(name="ml.category_model")
@typecheck.test_cases(record=pick({"categories"}, test_json))
@typecheck.returns("number")
def category_model(record):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn import linear_model
    class ColumnSelectTransformer(BaseEstimator,TransformerMixin):
        def __init__(self,keys):
            self.keys = keys

        def fit(self,X,y=None):
            return self
        def transform(self,X):
            import pandas as pd
            count = len(X)
            print count
            if type(self.keys) == tuple:
                X_trans = pd.DataFrame(columns=(self.keys))
                for i in xrange(count):
                    X_trans.loc[i] = [X[i][key] for key in self.keys]
                return X_trans
            else:
                X_trans = pd.Series(name=self.keys)
                for i in xrange(count):
                    if self.keys in X[i]:
                        X_trans.loc[i] = X[i][self.keys]
                    else:
                        return -1
                return X_trans

    select_transform = ColumnSelectTransformer(keys=('categories'))
    record_selection = select_transform.fit_transform([record])
    record_dictionary = []
    Dicti = {}
    for label in record_selection[0]:
        Dicti[label] = 1
    record_dictionary.append(Dicti)
    record_vector = dict_vectorizer.transform(record_dictionary)
    record_tdidf = tdidf_transformer.transform(record_vector)
    res = float(catmodel.predict(record_tdidf))
    return res


@fellow.batch(name="ml.attribute_knn_model")
@typecheck.test_cases(record=pick({"attributes"}, test_json))
@typecheck.returns("number")
def attribute_knn_model(record):

    reco_flat = dfs_selector(record[u'attributes'])
    reco_vector = attr_vectorizer.transform(reco_flat)
    reco_tfidf = attr_tfidf.transform(reco_vector)
    pres = float(attri_linear.predict(reco_tfidf))

    return pres


@fellow.batch(name="ml.full_model")
@typecheck.test_cases(record=exclude({"stars"}, test_json))
@typecheck.returns("number")
def full_model(record):
    full_model_record = full.transform([record])
    res = linearfitmodel.predict(full_model_record)
    if type(res)==list:
        return res[0]
    else:
        return float(res)
