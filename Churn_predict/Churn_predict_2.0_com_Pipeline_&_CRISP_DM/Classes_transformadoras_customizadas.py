
'''AQUI CONTEM AS CLASSES TRANSFORMADAORAS USADAS NO PIPELINE DO MODELO,
 DEVEM SER DEPOSITADO JUNTO DA PASTA DO PKL DO MODELO PAR QUE POSSAM SER ACESSADAS
'''

# Importando as bibliotecas necessárias
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
import numpy as np


# === REMOVE COLUNAS ===
class RemoveColunas(BaseEstimator, TransformerMixin):
    def __init__(self, colunas_para_remover=None):
        self.colunas_para_remover = colunas_para_remover or []

    def fit(self, X, y=None):
        self.feature_names_out_ = [col for col in X.columns if col not in self.colunas_para_remover]
        return self

    def transform(self, X):
        return X.drop(columns=self.colunas_para_remover, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# === ONE HOT ENCODER TRANSFORMER ===
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        self.feature_names_out_ = self.encoder.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X)
        return pd.DataFrame(encoded, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# === OUTLIER TRANSFORMER ===
class OutlierPercentilFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, limites=(0.01, 0.01)):
        self.limites = limites
        self.percentis_ = None

    def fit(self, X, y=None):
        self.percentis_ = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            p_inf = X[col].quantile(self.limites[0])
            p_sup = X[col].quantile(1 - self.limites[1])
            self.percentis_[col] = (p_inf, p_sup)
        self.feature_names_out_ = list(X.columns) + [f"{col}_outlier" for col in self.percentis_]
        return self

    def transform(self, X):
        X_outliers = X.copy()
        for col, (p_inf, p_sup) in self.percentis_.items():
            nova_coluna = f'{col}_outlier'
            X_outliers[nova_coluna] = ((X_outliers[col] < p_inf) | (X_outliers[col] > p_sup)).astype(int)
        return X_outliers

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# === ORDINAL ENCODER TRANSFORMER ===
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X)
        self.feature_names_out_ = X.columns
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X)
        return pd.DataFrame(encoded, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
    

# === PassThrough - variaveis sem tratamento ===
class PassThroughTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Retorna apenas as colunas selecionadas, sem alteração
        return X[self.columns].copy()
    
    def get_feature_names_out(self, input_features=None):
        return self.columns    


# === FEATURE ENGINEERING ===
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_transformed = self.transform(X.copy())  # aplica as transformações
        self.feature_names_out_ = X_transformed.columns
        return self

    def transform(self, X):
        X = X.copy()

        X['CreditScore_Bin'] = pd.cut(X['CreditScore'],bins=[0, 584, 718, np.inf],labels=['Baixo', 'Médio', 'Alto'])
        X['Age_Bin'] = pd.cut(X['Age'], bins=[18, 30, 50, 80, np.inf], labels=['Jovem', 'Adulto', 'Idoso', 'Muito Idoso'], right=True, include_lowest=True)
        X['Score_padron_z_score_por_faixa'] = X.groupby('Age_Bin')['CreditScore'].transform(lambda x: (x - x.mean()) / x.std())
        #X['Tenure_Age_Ratio_zscore'] = X.groupby('Age_Bin')['Tenure'].transform(lambda x: (x - x.mean()) / x.std())
        #X['Balance_flag'] = (X['Balance'] > 0).astype(int)
        X['Salary_Level'] = pd.cut(X['EstimatedSalary'], bins=[0, 50000, 100000, 150000, 200000], labels=['Baixo', 'Médio', 'Alto', 'Muito Alto'], right=True, include_lowest=True)
        #X['Balance_z_score_por_SalaryLevel'] = X.groupby('Salary_Level')['Balance'].transform(lambda x: (x - x.mean()) / x.std())
        X['High_Product_Active'] = ((X['NumOfProducts'] >= 2) & (X['IsActiveMember'] == 1)).astype(int)
        X['Complain_NoSolution'] = ((X['Complain'] == 1) & (X['Satisfaction Score'] < 5)).astype(int)
        X['Points_per_Product_zscored'] = X.groupby('NumOfProducts')['Point Earned'].transform(lambda x: (x - x.mean()) / x.std())

        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
