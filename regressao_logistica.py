import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from yellowbrick.regressor import ResidualsPlot

dados = pd.read_csv('Eleicao.csv',sep=';')
print(dados)
#Grafico de disperssão
plt.scatter(dados.SITUACAO, dados.DESPESAS)
plt.show()
# Descrição dos Dados
print(dados.describe())
#correlação
correlacao = np.corrcoef(dados.SITUACAO, dados.DESPESAS)
print(correlacao)
#Verificado correlação de 0.81218717, positiva forte.
# Definindo as variaveis do modelo
x = dados.iloc[:,2].values
#transformação de x para o formato de matriz adicionada um novo eixo(newaxis)
x = x[:, np.newaxis]
y = dados.iloc[:,1].values
# #Redefinindo a vartiavel x
# x = x.reshape(-1, 1)

#Criação do modelo com Regressão Logistica
# fit : treina o modelo
modelo = LogisticRegression()
modelo.fit(x, y)

#Coeficiente e interceptação do modelo
print(f'Coeficiente: {modelo.coef_}\ninterceptação: {modelo.intercept_}')

plt.scatter(x,y)
plt.show()
#criação de novos dados para gerar a função sigmoide
x_teste = np.linspace(10, 3000, 100)

#implementação de função sigmoide
def model(x):
    return 1 / (1 + np.exp(-x))

#Geração de previsões (variavel r) e visualização dos resultados
r = model(x_teste * modelo.coef_ + modelo.intercept_).ravel()
plt.plot(x_teste, r, color='red')
plt.show()


#novo dataset
base_previsao = pd.read_csv('NovosCandidatos.csv', sep=';')
print(base_previsao)

#mudança dos dados para formato de matriz
despesas = base_previsao.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)
# previsoes e geração de nova base de dados com os valores originais e as previsoes
previsoes_teste = modelo.predict(despesas)
print(previsoes_teste)
# np.column_stack() empilha arrays ao longo das colunas, ou seja, 
# junta várias colunas em uma matriz 2D. aqui no caso, base_previsao e previsoes_teste
base_previsao = np.column_stack((base_previsao, previsoes_teste))
print(base_previsao)




