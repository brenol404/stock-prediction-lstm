# Stock Market Price Prediction with LSTM Neural Networks

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Deep%20Learning-success)

Este repositório contém uma aplicação desenvolvida em Python que aplica técnicas de Deep Learning para a análise e previsão de séries temporais financeiras. Utilizando uma arquitetura de Rede Neural Recorrente (RNN) do tipo Long Short-Term Memory (LSTM), o modelo analisa dados históricos de ativos para prever seu valor de fechamento subsequente.

## Funcionalidades Principais
- **Integração de Dados Dinâmica**: Consumo em tempo real da API do Yahoo Finance (`yfinance`), permitindo a extração de dados de ações globais, índices e criptomoedas.
- **Modelagem Preditiva**: Implementação de redes LSTM focadas em capturar dependências de longo prazo e identificar padrões de volatilidade no mercado.
- **Pipeline de Pré-processamento**: Estruturação de dados em tensores tridimensionais com janelas deslizantes (sliding windows) de 60 dias e normalização via `MinMaxScaler`.
- **Avaliação de Performance**: Cálculo de erro preditivo baseado na métrica RMSE (Root Mean Squared Error) e geração de representações visuais comparativas entre dados reais e previsões.

## Tecnologias e Dependências
- **Python 3.12+**
- **TensorFlow & Keras**: Construção, treinamento e inferência do modelo de Deep Learning.
- **Scikit-Learn**: Escalonamento de dados e avaliação de métricas preditivas.
- **Pandas & NumPy**: Manipulação, limpeza e estruturação de matrizes multidimensionais.
- **Matplotlib**: Plotagem gráfica para análise exploratória e validação visual.
- **yfinance**: Interface de comunicação e coleta de dados do mercado.

## Instruções de Uso

### 1. Configuração do Ambiente
Certifique-se de possuir o Python (64-bits) instalado e execute o comando abaixo para instalar as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 2. Rodando o Projeto
Abra o terminal na pasta do projeto e execute:
```bash
python previsao.py
```

### 3. Exemplos de Tickers para Pesquisar
Ao rodar o programa, ele perguntará qual ação você deseja analisar. Você pode digitar:
* **Ações Internacionais:** `AAPL` (Apple), `TSLA` (Tesla), `NVDA` (Nvidia).
* **Ações Brasileiras (Adicione .SA):** `PETR4.SA` (Petrobras), `VALE3.SA` (Vale), `ITUB4.SA` (Itaú).
* **Criptomoedas:** `BTC-USD` (Bitcoin).
* **Índices:** `^BVSP` (Ibovespa).

## Como o Modelo Funciona (Arquitetura)
1. **Coleta:** O histórico desde 2012 é baixado.
2. **Separação:** 80% dos dados mais antigos são usados para *Treino*, e os 20% mais recentes para *Teste* (Validação).
3. **Moldagem (Shape):** Os dados são agrupados em "blocos de 60 dias". A IA tenta descobrir qual será o preço do 61º dia baseando-se no comportamento dos 60 dias anteriores.
4. **Treinamento:** A rede passa por 5 *epochs* (ciclos completos de aprendizado), ajustando seus pesos através do otimizador `adam` para reduzir a função de perda (`mean_squared_error`).
5. **Avaliação:** Calculamos o RMSE (Root Mean Squared Error) cruzando a previsão da IA com a vida real no bloco de testes para saber, em dólares/reais, a média de erro do modelo.
