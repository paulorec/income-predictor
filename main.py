# -*- coding: utf-8 -*-
# Importe as bibliotecas necessárias para o projeto.
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Permite a utilização da função display() para DataFrames.
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

# Carregando os dados do Censo
data = pd.read_csv("census.csv")

#display(data.head(n=10))

n_records = data.size

# Número de registros com remuneração anual superior à $50,000
n_greater_50k = len(filter(lambda x : x == '>50K',data.income))

# O número de registros com remuneração anual até $50,000
n_at_most_50k = len(filter(lambda x : x == '<=50K' ,data.income))

# O percentual de indivíduos com remuneração anual superior à $50,000
greater_percent = (n_greater_50k*100) /n_records

# Exibindo os resultados
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Aplicando a transformação de log nos registros distorcidos.
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Exibindo um exemplo de registro com a escala aplicada
display(features_log_minmax_transform.head(n=5))
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Faça o encode da coluna 'income_raw' para valores numéricos
income = income_raw.map(lambda x : x == ">50K" )

# Exiba o número de colunas depois do one-hot encoding
encoded = list(features_final.columns)
#print "{} total features after one-hot encoding.".format(len(encoded))
#print encoded

# Dividir os 'atributos' e 'income' entre conjuntos de treinamento e de testes.
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])