import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
pd.options.mode.chained_assignment = None
  
class MappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_column, mapping_dict:dict):
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column
    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X
    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'
        column_set = set(X[self.mapping_column])
        keys_not_found = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        keys_absent = column_set -  set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_ = X.copy()
        X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result
      
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_list, action="drop"):
        assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
        self.column_list=column_list
        self.action=action

    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        column_set=set(self.column_list)-set(X.columns.to_list())
        if self.action == 'keep':
            assert column_set == set(), f'{self.__class__.__name__}.transform does not contain these columns to keep: {column_set} .'
            X_=X.copy()
            X_ = X[self.column_list]
        if self.action == 'drop':
            X_=X.copy()
            if column_set != set():
              print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {column_set}.\n")
            X_= X_.drop(columns=self.column_list, errors ='ignore')
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result
      
class OHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, dummy_na=False, drop_first=False):  
        self.target_column = target_column
        self.dummy_na=dummy_na
        self.drop_first=drop_first

    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'
        X_ = X.copy()
        X_=pd.get_dummies(X,prefix=self.target_column,prefix_sep='_', columns=[self.target_column],dummy_na=self.dummy_na, drop_first=self.drop_first)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result
      
      
class Sigma3Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, column_name):
        self.column_name = column_name  

    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.column_name in X.columns.to_list(), f'unknown column {self.column_name}'
        assert all([isinstance(v, (int, float)) for v in X[self.column_name].to_list()])

        m=X[self.column_name].mean()
        sig=X[self.column_name].std()
        s3min=(m-3*sig)
        s3max=(m+3*sig)
        X_=X.copy()
        X_[self.column_name] = X_[self.column_name].clip(lower=s3min, upper=s3max)
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result
      
      
class TukeyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, target_column,fence):
        self.target_column = target_column 
        self.fence=fence

    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X

    def transform(self, X):
        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3-q1
        outer_low = q1-3*iqr
        outer_high = q3+3*iqr

        inner_low = q1-1.5*iqr
        inner_high = q3+1.5*iqr

        X_=X.copy()
    
        if self.fence == 'inner':
            X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
            return X_
        if self.fence == 'outer':
            X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
            return X_
        

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result
      
      
class MinMaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass 

    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X

    def transform(self, X):
        X_=X.copy()
        min=X_.min()
        max=X_.max()
        denom=max-min
        X_ -=min
        X_ /=denom
        return X_

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result


class KNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,n_neighbors=5, weights="uniform"):
        from sklearn.impute import KNNImputer
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.KNNImputer=KNNImputer
     

    def fit(self, X, y = None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return X
      
    def transform(self, X):
        imputer=self.KNNImputer(n_neighbors=self.n_neighbors,weights=self.weights,add_indicator=False)
        column_name=X.columns.to_list()
        imputer_df = pd.DataFrame(imputer.fit_transform(X), columns = column_name)
        return imputer_df

    def fit_transform(self, X, y = None):
        result = self.transform(X)
        return result
      
 def find_random_state(features_df, labels, n=200):
    model = KNeighborsClassifier(n_neighbors=5)
    var = []  
    for i in range(1, n):
        train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
        model.fit(train_X, train_y) 
        train_pred = model.predict(train_X)           
        test_pred = model.predict(test_X)             
        train_f1 = f1_score(train_y, train_pred)   
        test_f1 = f1_score(test_y, test_pred)      
        f1_ratio = test_f1/train_f1          
        var.append(f1_ratio)

    rs_value = sum(var)/len(var)
    rs_id = np.array(abs(var - rs_value)).argmin()
    return rs_id 
      
      
