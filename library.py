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
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        column_name=X_.columns.to_list()
        scaler_df=pd.DataFrame(scaler.fit_transform(X_),columns = column_name)
        return scaler_df

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
        train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,random_state=i, stratify=labels)
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
  
def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
 
  features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()
  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)
                                    
  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')),
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')),
    ('minmax', MinMaxTransformer()), 
    ('imputer', KNNTransformer()) 
    ], verbose=True)

  
def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  label='Survived'
  return dataset_setup(titanic_table, label, transformer, rs, ts)

customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),  #you may need to add an action if you have no default
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)
  
def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  label='Rating'
  return dataset_setup(customer_table, label, transformer, rs, ts)
      
      
