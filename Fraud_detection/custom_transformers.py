
'''MESMO CODIGO DO NOTEBOOK DO MODELO, SE ALTERAR CLASSES LA DEVE-SE ALTERAR AQUI E VICE VERSA 
(AQUI CABE UMA MELHORIA PARA QUE SEJA SALVO DE LA AQUI AUTOMATICAMENTE)
'''

# Importando as bibliotecas necessárias
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, OrdinalEncoder
from category_encoders.woe import WOEEncoder
import pandas as pd
import numpy as np


# Classe para transformar com Target Encoder
class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = None

    def fit(self, X, y):
        col = X.columns[0]
        self.mapping_ = X[col].to_frame().join(pd.Series(y, name='target')).groupby(col)['target'].mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        col = X.columns[0]
        X_transformed[col] = X_transformed[col].map(self.mapping_)
        X_transformed[col].fillna(0, inplace=True)
        return X_transformed


# Classe para transformação One Hot Encoder
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X)
        col_names = self.encoder.get_feature_names_out(X.columns)
        return pd.DataFrame(encoded, columns=col_names, index=X.index)


# Classe para transformação Yeo-Johnson
class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variaveis=None):
        """
        Classe para aplicar a transformação Yeo-Johnson nas variáveis específicas.
         lista de variáveis para aplicar a transformação. Se None, aplica a todas as variáveis numéricas.
        """
        self.variaveis = variaveis
        self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)  # Yeo-Johnson sem padronização
    
    def fit(self, X, y=None):
        """
        Ajusta o transformer nas variáveis especificadas.
        :param X: DataFrame com os dados de treino.
        :param y: Ignorado (necessário para compatibilidade com pipeline).
        """
        # Se a lista de variáveis for None, aplica a transformação em todas as colunas numéricas
        if self.variaveis is None:
            self.variaveis = X.select_dtypes(include=['number']).columns.tolist()

        # Verifica se todas as variáveis existem no DataFrame
        missing_vars = [var for var in self.variaveis if var not in X.columns]
        if missing_vars:
            raise ValueError(f"As seguintes variáveis não existem no DataFrame: {', '.join(missing_vars)}")

        # Verifica e remove valores ausentes (NaN)
        if X[self.variaveis].isnull().any().any():
            raise ValueError("Existem valores ausentes nas variáveis selecionadas. Por favor, trate-os antes de aplicar a transformação.")
        
        # Ajusta o transformer com os dados das variáveis selecionadas
        self.transformer.fit(X[self.variaveis])
        return self
    
    def transform(self, X):
        """
        Aplica a transformação Yeo-Johnson nas variáveis especificadas.
        :param X: DataFrame com os dados para transformação.
        :return: DataFrame com as variáveis transformadas.
        """
        # Verifica se as variáveis foram previamente definidas
        if not self.variaveis:
            raise ValueError("Nenhuma variável definida para transformação.")
        
        X_transformed = X.copy()

        # Aplica a transformação nas variáveis selecionadas
        X_transformed[self.variaveis] = self.transformer.transform(X[self.variaveis])
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Ajusta e aplica a transformação Yeo-Johnson nas variáveis especificadas.
        :param X: DataFrame com os dados de treino.
        :param y: Ignorado (necessário para compatibilidade com pipeline).
        :return: DataFrame com as variáveis transformadas.
        """
        self.fit(X, y)
        return self.transform(X)


# Classe para flag de outliers com base em percentis
class OutlierPercentilFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, limites=(0.01, 0.01)):
        """
        Inicializa o transformador com limites inferiores e superiores para outliers, baseados em percentis.
        
        Parâmetros:
            limites (tuple): Limites inferiores e superiores para identificar outliers com base em percentis (ex: (0.01, 0.01) = 1%).
        """
        self.limites = limites
        self.percentis_ = None

    def fit(self, X, y=None):
        """
        Ajusta o transformador aos dados de entrada, calculando os percentis para cada variável numérica.
        
        Parâmetros:
            X (pd.DataFrame): Dados de entrada para ajuste (não é utilizado no cálculo, mas necessário para o fit).
        
        Retorna:
            self: O próprio transformador.
        """
        # Calculando os percentis para cada variável numérica
        self.percentis_ = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            p_inf = X[col].quantile(self.limites[0])
            p_sup = X[col].quantile(1 - self.limites[1])
            self.percentis_[col] = (p_inf, p_sup)

        return self

    def transform(self, X):
        """
        Transforma os dados de entrada, criando flags binárias para os outliers com base em percentis.
        
        Parâmetros:
            X (pd.DataFrame): Dados a serem transformados, com os quais as flags de outliers serão geradas.
        
        Retorna:
            pd.DataFrame: DataFrame com as novas colunas de outliers adicionadas.
        """
        X_outliers = X.copy()
        
        for col, (p_inf, p_sup) in self.percentis_.items():
            nova_coluna = f'{col}_outlier'
            X_outliers[nova_coluna] = ((X_outliers[col] < p_inf) | (X_outliers[col] > p_sup)).astype(int)
        
        return X_outliers


# Classe para transformação WOE (Weight of Evidence)
class WoeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        """
        cols: lista das colunas para aplicar o WOEEncoder.
        """
        self.cols = cols
        self.encoders_ = {}

    def fit(self, X, y):
        for col in self.cols:
            encoder = WOEEncoder()
            encoder.fit(X[col], y)
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols:
            encoder = self.encoders_[col]
            X_transformed[col] = encoder.transform(X_transformed[col])
        return X_transformed


# Classe para transformação Ordinal Encoder
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X)
        return pd.DataFrame(encoded, columns=X.columns, index=X.index)
