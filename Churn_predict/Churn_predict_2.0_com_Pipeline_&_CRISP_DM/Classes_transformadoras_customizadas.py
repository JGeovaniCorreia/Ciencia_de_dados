
'''AQUI CONTEM AS CLASSES TRANSFORMADAORAS USADAS NO PIPELINE DO MODELO,
 DEVEM SER DEPOSITADO JUNTO DA PASTA DO PKL DO MODELO PARA QUE POSSAM SER ACESSADAS
'''

# bibliotecas necessarias para as classes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


# === REMOVE COLUNAS ===
class RemoveColunas(BaseEstimator, TransformerMixin):

    '''
    Classe que remove colunas específicas do DataFrame, conforme lista informada.

    '''
    def __init__(self, colunas_para_remover=None):
        self.colunas_para_remover = colunas_para_remover or []

    def fit(self, X, y=None):
        X = self._check_dataframe(X)
        self.feature_names_out_ = [col for col in X.columns if col not in self.colunas_para_remover]
        return self

    def transform(self, X):
        X = self._check_dataframe(X)
        return X.drop(columns=self.colunas_para_remover, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _check_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Este transformador espera um DataFrame como entrada.")
        return X


# === ONE HOT ENCODER TRANSFORMER ===
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):

    '''
    Classe que aplica One-Hot Encoding, transformando categorias em colunas binárias (0 ou 1).

    '''
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        self.feature_names_out_ = self.encoder.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X)
        return pd.DataFrame(encoded, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# === ORDINAL ENCODER TRANSFORMER ===
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):

    '''
    Classe que aplica codificação ordinal em variáveis categóricas, atribuindo números inteiros às categorias. 
    Útil para variáveis categóricas que guardam informação ordinal relevante.
    
    Parâmetro:
    - categories: lista de listas, onde cada lista define a ordem das categorias para cada variável.
      Exemplo: categories=[['Baixo', 'Médio', 'Alto', 'Muito Alto']]
    '''

    def __init__(self, categories=None):
        self.categories = categories
        self.encoder = OrdinalEncoder(categories=categories,
                                      handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        X = self._check_dataframe(X)
        self.encoder.fit(X)
        self.feature_names_out_ = X.columns
        return self

    def transform(self, X):
        X = self._check_dataframe(X)
        encoded = self.encoder.transform(X)
        return pd.DataFrame(encoded, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _check_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OrdinalEncoderTransformer espera um DataFrame como entrada.")
        return X


# === OUTLIER FLAG TRANSFORMER ===
class OutlierPercentilFlagTransformer(BaseEstimator, TransformerMixin):

    '''
    Classe que cria flags binárias de outliers com base em percentis definidos via parametro limites=(). 
    Útil para sinalizar valores extremos sem alterar os dados originais.
    ''' 

    def __init__(self, limites=(0.01, 0.01)):
        self.limites = limites
        self.percentis_ = None

    def fit(self, X, y=None):
        X = self._check_dataframe(X)
        self.percentis_ = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            p_inf = X[col].quantile(self.limites[0])
            p_sup = X[col].quantile(1 - self.limites[1])
            self.percentis_[col] = (p_inf, p_sup)
        self.feature_names_out_ = list(X.columns) + [f"{col}_outlier" for col in self.percentis_]
        return self

    def transform(self, X):
        X = self._check_dataframe(X)
        X_outliers = X.copy()
        for col, (p_inf, p_sup) in self.percentis_.items():
            nova_coluna = f'{col}_outlier'
            X_outliers[nova_coluna] = ((X_outliers[col] < p_inf) | (X_outliers[col] > p_sup)).astype(int)
        return X_outliers

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _check_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OutlierPercentilFlagTransformer espera um DataFrame como entrada.")
        return X


# === PassThrough - variáveis sem transformação ===
class PassThroughTransformer(BaseEstimator, TransformerMixin):

    ''' 
   Classe que aplica PassThrough, ou seja, passa os valores sem alteraçao para o modelo, 
   serve para variaveis que nao se deseja aplicar nenhuma trasnformacao via pipeline

    '''
        
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X = self._check_dataframe(X)
        return self

    def transform(self, X):
        X = self._check_dataframe(X)
        return X[self.columns].copy()

    def get_feature_names_out(self, input_features=None):
        return self.columns

    def _check_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PassThroughTransformer espera um DataFrame como entrada.")
        return X


# === SIMPLE IMPUTER TRANSFORMER ===
class SimpleImputerTransformer(BaseEstimator, TransformerMixin):

    ''' 
   Classe que aplica simple imputer, técnica de pré-processamento que preenche 
   valores ausentes em um dataset com uma estratégia simples, atraves do parametro strategy, que no caso é median
    '''

    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        X = self._check_dataframe(X)
        self.imputer.fit(X)
        self.feature_names_out_ = X.columns
        return self

    def transform(self, X):
        X = self._check_dataframe(X)
        imputed_array = self.imputer.transform(X)
        return pd.DataFrame(imputed_array, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _check_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SimpleImputerTransformer espera um DataFrame como entrada.")
        return X


# === FEATURE ENGINEERING ===
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    '''
    classe que aplica engenharia de features a partir das variavies ja existentes

    '''


    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X = self._check_dataframe(X)
        X_transformed = self.transform(X.copy())
        self.feature_names_out_ = X_transformed.columns
        return self

    def transform(self, X):
        X = self._check_dataframe(X).copy()
        
        X['CreditScore_Bin'] = pd.cut(X['CreditScore'],bins=[0, 584, 718, np.inf],labels=['Baixo', 'Médio', 'Alto'])

        ''' 
        aqui sao variaveis que foram criadas e estudadas, mas que nao apresentaram ganho na predicao do evento em estudo; 
        foram descartadas. Seus valores se restrigem a analise descritiva doevento apenas. 

        '''
        #X['Age_Bin'] = pd.cut(X['Age'], bins=[18, 30, 50, 80, np.inf], labels=['Jovem', 'Adulto', 'Idoso', 'Muito Idoso'], right=True, include_lowest=True)
        #X['Score_padron_z_score_por_faixa'] = X.groupby('Age_Bin')['CreditScore'].transform(lambda x: (x - x.mean()) / x.std())
        #X['Tenure_Age_Ratio_zscore'] = X.groupby('Age_Bin')['Tenure'].transform(lambda x: (x - x.mean()) / x.std())
        X['Balance_flag'] = (X['Balance'] > 0).astype(int)
        X['Salary_Level'] = pd.cut(X['EstimatedSalary'], bins=[0, 50000, 100000, 150000, 200000], labels=['Baixo', 'Médio', 'Alto', 'Muito Alto'], right=True, include_lowest=True)
        #X['Balance_z_score_por_SalaryLevel'] = X.groupby('Salary_Level')['Balance'].transform(lambda x: (x - x.mean()) / x.std())
        X['High_Product_Active'] = ((X['NumOfProducts'] >= 2) & (X['IsActiveMember'] == 1)).astype(int)
        #X['Complain_NoSolution'] = ((X['Complain'] == 1) & (X['Satisfaction Score'] <=3)).astype(int)
        #X['Points_per_Product_zscored'] = X.groupby('NumOfProducts')['Point Earned'].transform(lambda x: (x - x.mean()) / x.std())

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _check_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineeringTransformer espera um DataFrame como entrada.")
        return X





    


