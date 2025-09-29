import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Seção 3

base_census = pd.read_csv('analisecenso/census.csv')
print("Primeiros 5 valores: \n", base_census.head(5))
print("\nDados: \n", base_census.describe())

plt.hist(base_census['age'], bins=30, color='blue', alpha=0.7)
plt.title('Distribuição de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

LabelEncoderTeste = LabelEncoder()
teste = LabelEncoderTeste.fit_transform(x_census[:, 1])

print(teste)

LabelEncoderWorkclass = LabelEncoder()
LabelEncoderEducation = LabelEncoder()
LabelEncoderMaritalStatus = LabelEncoder()
LabelEncoderOccupation = LabelEncoder() 
LabelEncoderRelationship = LabelEncoder()
LabelEncoderRace = LabelEncoder()
LabelEncoderSex = LabelEncoder()
LabelEncoderNativeCountry = LabelEncoder()

x_census[:, 1] = LabelEncoderWorkclass.fit_transform(x_census[:, 1])
x_census[:, 3] = LabelEncoderEducation.fit_transform(x_census[:, 3])
x_census[:, 5] = LabelEncoderMaritalStatus.fit_transform(x_census[:, 5])
x_census[:, 6] = LabelEncoderOccupation.fit_transform(x_census[:, 6])
x_census[:, 7] = LabelEncoderRelationship.fit_transform(x_census[:, 7])
x_census[:, 8] = LabelEncoderRace.fit_transform(x_census[:, 8])
x_census[:, 9] = LabelEncoderSex.fit_transform(x_census[:, 9])
x_census[:, 13] = LabelEncoderNativeCountry.fit_transform(x_census[:, 13])

OneHotEncoderCensus = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
x_census = OneHotEncoderCensus.fit_transform(x_census).toarray()

scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

with open('census.pkl', 'wb') as f:
    pickle.dump([x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste], f)