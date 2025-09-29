import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Seção 3

base_credit = pd.read_csv('analisecredito/credit_data.csv')
print("Primeiros 5 valores: \n", base_credit.head(5))
print("\nDados: \n", base_credit.describe())

sns.countplot(x='default', data=base_credit)
plt.show()  

plt.hist(base_credit['age'], bins=50)
plt.show()  

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')

base_credit = base_credit[base_credit['age'] >= 0]
print("\nApós remoção de idades negativas: \n", base_credit.describe())

media = base_credit['age'][base_credit['age'] >= 0].mean()

print("\nMédia das idades: ", media)

x_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

print(x_credit[:, 0].min())

scaler_credit = StandardScaler()
x_credit = scaler_credit.fit_transform(x_credit)

x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)

with open('credit.pkl', 'wb') as f:
    pickle.dump([x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste], f)