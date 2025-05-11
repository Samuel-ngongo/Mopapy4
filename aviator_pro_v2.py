
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

st.set_page_config(page_title="Aviator PRO - IA Adaptativa Total", layout="centered")
st.title("Aviator PRO - IA Inteligente com Padrões, Confiança e Histórico")

# Histórico
if "valores" not in st.session_state:
    st.session_state.valores = []

if "historico_completo" not in st.session_state:
    st.session_state.historico_completo = []

# Entrada
novo = st.text_input("Insira um valor (ex: 2.31):")
if st.button("Adicionar") and novo:
    try:
        valor = float(novo)
        st.session_state.valores.append(valor)
        st.session_state.historico_completo.append((valor, datetime.now().strftime("%d/%m/%Y %H:%M")))
        st.success("Valor adicionado.")
    except:
        st.error("Formato inválido.")

# Previsão com IA adaptativa
def prever_valores(dados):
    if len(dados) < 5:
        return 1.40, 1.80, 2.10, 30

    media = np.mean(dados)
    pesos = np.linspace(1, 2, len(dados))
    media_pond = np.average(dados, weights=pesos)

    if LinearRegression and len(dados) >= 6:
        X = np.arange(len(dados)).reshape(-1, 1)
        y = np.array(dados)
        modelo = LinearRegression()
        modelo.fit(X, y)
        pred = modelo.predict(np.array([[len(dados) + 1]]))[0]
    else:
        pred = media_pond

    final = (media + media_pond + pred) / 3
    desvio = np.std(dados[-10:]) if len(dados) >= 10 else np.std(dados)

    inferior = round(final - desvio, 2)
    superior = round(final + desvio, 2)
    confianca = round(max(5, min(99, 100 - desvio * 90)), 1)

    return round(inferior, 2), round(final, 2), round(superior, 2), confianca

# Mudança brusca
def detectar_transicao(dados):
    if len(dados) < 10:
        return False
    ultimos = np.array(dados[-5:])
    anteriores = np.array(dados[-10:-5])
    diff_media = abs(np.mean(ultimos) - np.mean(anteriores))
    diff_std = abs(np.std(ultimos) - np.std(anteriores))
    return diff_media > 1.0 or diff_std > 1.0

# Análise de padrões
def analisar_padroes(dados):
    alertas = []
    if len(dados) >= 3:
        ultimos3 = dados[-3:]
        if all(v < 1.5 for v in ultimos3):
            alertas.append(("Queda contínua detectada", 70))
        if all(v > 2.5 for v in ultimos3):
            alertas.append(("Alta contínua detectada", 65))
        if len(set(np.sign(np.diff(ultimos3)))) > 1:
            alertas.append(("Alternância instável", 60))
    return alertas

# Visualização
def mostrar_graficos(valores):
    df = pd.DataFrame({
        'Índice': list(range(1, len(valores) + 1)),
        'Valor': valores
    })

    st.subheader("Mini Gráfico de Barras (últimos 10)")
    st.bar_chart(df.tail(10).set_index('Índice'))

    st.subheader("Evolução da Média Móvel")
    df['Média Móvel'] = df['Valor'].rolling(window=3, min_periods=1).mean()
    st.line_chart(df.set_index('Índice')[['Valor', 'Média Móvel']])

# Exibição
if st.session_state.valores:
    st.subheader("Histórico (últimos 30)")
    for valor, data in st.session_state.historico_completo[-30:]:
        cor = "🟥" if valor < 1.5 else "🟩" if valor > 2.5 else "⬜"
        st.write(f"{cor} {valor:.2f}x - {data}")

    mostrar_graficos(st.session_state.valores)

    st.subheader("Previsão e Análise Inteligente")
    inf, media, sup, conf = prever_valores(st.session_state.valores)

    st.info(f"Próxima estimativa: entre **{inf}x** e **{sup}x**")
    st.success(f"Estimativa principal: **{media}x**")
    st.warning(f"Nível de confiança: **{conf}%**")

    # Mensagem por faixa de confiança
    if conf >= 80:
        st.success("Alta confiança – possível padrão estável ou repetição.")
    elif conf >= 60:
        st.info("Boa confiança – observar com atenção.")
    elif conf >= 40:
        st.warning("Confiança fraca – sinais de instabilidade.")
    else:
        st.error("Alta incerteza – possível transição de padrão ou comportamento anômalo.")

    if detectar_transicao(st.session_state.valores):
        st.warning("Transição de padrão detectada. A IA está se ajustando...")

    for alerta, chance in analisar_padroes(st.session_state.valores):
        st.info(f"Alerta: {alerta} ({chance}% de chance)")

# Limpar
if st.button("Limpar dados"):
    st.session_state.valores = []
    st.session_state.historico_completo = []
    st.success("Histórico limpo.")
