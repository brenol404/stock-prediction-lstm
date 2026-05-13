"""
Previsão de Preço de Ações com LSTM

Este script realiza o download de dados históricos do Yahoo Finance,
pré-processa as séries temporais e treina uma Rede Neural Recorrente (LSTM)
para prever o preço de fechamento do próximo dia útil.
"""

import os
# Desativar avisos do TensorFlow para deixar o terminal limpo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import contextlib
import io
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# --- CONFIGURAÇÕES GLOBAIS ---
# Clean Code: Evitar "Magic Numbers" (números espalhados pelo código)
TIME_STEPS = 60
EPOCHS = 5
BATCH_SIZE = 32

# Garantir Reprodutibilidade (Resultados consistentes em diferentes execuções)
np.random.seed(42)
tf.random.set_seed(42)

# 1. Definir o ticker da ação (AAPL = Apple. Para Petrobras, seria 'PETR4.SA')
print("\n==================================================")
print(" 📈 BEM-VINDO AO PREVISOR DE AÇÕES COM IA (LSTM)")
print("==================================================")

while True:
    ticker = input("\nDigite o ticker do ativo desejado (ex: AAPL, TSLA, PETR4.SA, BTC-USD): ").strip().upper()
    if not ticker:
        print("⚠️ AVISO: O ticker não pode estar vazio. Por favor, insira um código válido.")
        continue

    print(f"Buscando dados históricos para '{ticker}'... Aguarde.")

    # 2. Baixar os dados de forma silenciosa para evitar erros sujos no terminal
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dados = yf.download(ticker, start='2012-01-01', progress=False)

    # --- TRATAMENTO DE ERRO (ROBUSTEZ) ---
    if dados.empty:
        print(f"❌ ERRO: O ativo '{ticker}' não foi encontrado na base de dados.")
        print("   -> Dica: Verifique o código exato no site oficial do Yahoo Finance (finance.yahoo.com).")
        print("   -> Dica: Mercados fora dos EUA geralmente exigem um sufixo (ex: .SA para Brasil, .L para Londres, .T para Tóquio).\n")
        continue # Volta para o início do loop (pede o ticker de novo)
    
    # --- VALIDAÇÃO DE HISTÓRICO MÍNIMO (EX: AÇÕES COM IPO RECENTE) ---
    if len(dados) < TIME_STEPS * 2:
        print(f"❌ ERRO: O ativo '{ticker}' possui histórico insuficiente para a Inteligência Artificial.")
        print(f"   -> A IA exige pelo menos {TIME_STEPS * 2} pregões para aprender, mas '{ticker}' tem apenas {len(dados)}.\n")
        continue

    break # Se encontrou os dados com sucesso, sai do loop e continua o código principal!

# 3. Mostrar as primeiras linhas no terminal para confirmar que deu certo
print("\n--- Primeiras linhas dos dados ---")
print(dados.head())

# --- NOVO CÓDIGO: PRÉ-PROCESSAMENTO ---
print("\n--- Iniciando o Pré-processamento dos Dados ---")

# Filtrar apenas a coluna 'Close'
# O .dropna() remove valores nulos/defeituosos (NaN) que a API possa retornar acidentalmente
dados_fechamento = dados['Close'].dropna()

# Blindagem contra atualizações do yfinance: garante compatibilidade estrutural futura
if isinstance(dados_fechamento, pd.Series):
    dados_fechamento = dados_fechamento.to_frame(name=ticker)

dataset = dados_fechamento.values.reshape(-1, 1)

# Separar 80% dos dados para treinamento
tamanho_treinamento = int(np.ceil(len(dataset) * .80))
print(f"Total de dados: {len(dataset)} | Dados de treino: {tamanho_treinamento}")

# --- NORMALIZAÇÃO SEM VAZAMENTO DE DADOS (DATA LEAKAGE) ---
scaler = MinMaxScaler(feature_range=(0, 1))
# Ajusta o scaler APENAS com os dados de treino (para a IA não "espiar" o futuro)
scaler.fit(dataset[0:tamanho_treinamento, :])
# Aplica a escala no dataset inteiro baseado no que aprendeu com o treino
dados_escalados = scaler.transform(dataset)

# Criar as sequências de treinamento (usando a janela definida)
dados_treino = dados_escalados[0:tamanho_treinamento, :]
X_train, y_train = [], []

for i in range(TIME_STEPS, len(dados_treino)):
    X_train.append(dados_treino[i-TIME_STEPS:i, 0])
    y_train.append(dados_treino[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"\nDados preparados para a LSTM!")
print(f"Formato de X_train (amostras, dias, features): {X_train.shape}")
print(f"Formato de y_train (amostras): {y_train.shape}")

# --- NOVO CÓDIGO: CONSTRUINDO E TREINANDO A REDE NEURAL ---
print("\n--- Construindo a Rede Neural LSTM ---")

modelo = Sequential()

# Camada de Entrada
modelo.add(Input(shape=(X_train.shape[1], 1)))
# Primeira camada LSTM (aprende as sequências temporais)
modelo.add(LSTM(50, return_sequences=True))

# Segunda camada LSTM
modelo.add(LSTM(50, return_sequences=False))

# Camadas Densas (Processamento final)
modelo.add(Dense(25))
modelo.add(Dense(1)) # Camada de saída: queremos prever apenas 1 valor (o preço de amanhã)

# Compilar o modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

print("Arquitetura do modelo pronta!")
modelo.summary()

print("\n--- Iniciando o Treinamento (isso pode levar alguns minutos) ---")
modelo.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# --- NOVO CÓDIGO: TESTE E AVALIAÇÃO DO MODELO ---
print("\n--- Preparando dados de Teste e Fazendo Previsões ---")

# Criar o conjunto de dados de teste
# Pegamos os últimos dias de treino para começar a prever o primeiro dia de teste
dados_teste = dados_escalados[tamanho_treinamento - TIME_STEPS: , :]

X_test = []
y_test = dataset[tamanho_treinamento:, :] # Valores reais (sem escala) para compararmos depois

for i in range(TIME_STEPS, len(dados_teste)):
    X_test.append(dados_teste[i-TIME_STEPS:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Fazer as previsões
previsoes = modelo.predict(X_test)

# Reverter a normalização (transformar a escala 0 a 1 de volta para Dólares)
previsoes = scaler.inverse_transform(previsoes)

# Calcular o RMSE (Erro Quadrático Médio) - quanto menor esse valor, mais precisa é a IA
rmse = np.sqrt(np.mean(((previsoes - y_test) ** 2)))
print(f"\n[AVALIAÇÃO] Margem de Erro (RMSE): {rmse}")

# --- NOVO CÓDIGO: VISUALIZAÇÃO FINAL ---
print("\nGerando gráfico final com os resultados...")
treino = dados_fechamento.iloc[:tamanho_treinamento].copy()
valido = dados_fechamento.iloc[tamanho_treinamento:].copy()
valido['Previsoes'] = previsoes

# Definir a moeda do gráfico e terminal baseada no Ticker
moeda = "R$" if ticker.endswith(".SA") else "US$"

plt.figure(figsize=(16, 8))
plt.title(f'Previsão de Preço de Ações usando LSTM - {ticker}', fontsize=18)
plt.xlabel('Data', fontsize=14)
plt.ylabel(f'Preço de Fechamento ({moeda})', fontsize=14)
plt.plot(treino[ticker], color='blue', label='Dados de Treino (Passado)')
plt.plot(valido[ticker], color='green', label='Preço Real (Validação)')
plt.plot(valido['Previsoes'], color='red', label='Previsão do Modelo')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)

# --- NOVO CÓDIGO: PREVISÃO PARA AMANHÃ ---
print("\n--- Calculando a Previsão para o Próximo Dia ---")

# Pegar os últimos dias do dataset inteiro
ultimos_dias = dataset[-TIME_STEPS:]

# Normalizar esses últimos dias com o MESMO scaler
ultimos_dias_escalados = scaler.transform(ultimos_dias)

# Moldar para a LSTM (Amostras, Dias, Features)
X_amanha = np.array([ultimos_dias_escalados])
X_amanha = np.reshape(X_amanha, (X_amanha.shape[0], X_amanha.shape[1], 1))

# Prever e reverter a escala de Dólar
previsao_amanha = modelo.predict(X_amanha)
previsao_amanha = scaler.inverse_transform(previsao_amanha)

print(f"\n🔮 PREVISÃO: O preço estimado da ação {ticker} para o próximo dia útil é: {moeda} {previsao_amanha[0][0]:.2f}\n")

# --- PRÁTICA DE MLOps: SALVAR O MODELO TREINADO ---
nome_arquivo_modelo = f"modelo_{ticker.replace('^', '').replace('-', '_')}.keras"
modelo.save(nome_arquivo_modelo)
print(f"💾 Modelo salvo localmente como '{nome_arquivo_modelo}' (Ignorado pelo Git).\n")

# Exibir o gráfico por último (pois isso pausa a execução do terminal)
print("📊 Abrindo o gráfico interativo... Feche a janela para finalizar o programa.")
plt.show()
