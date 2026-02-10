import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real vs Modelo (X,Y) + Métricas", layout="wide")
st.title("Comparação: X/Y Real vs X/Y Modelo + Correlação do Resultante")

st.markdown("""
Carregue um CSV contendo as colunas:
- **X_real, Y_real**
- **X_modelo, Y_modelo**
(opcional: **Timestamp**)
""")

uploaded = st.file_uploader("📄 Carregue o CSV", type=["csv"])

# ---- Helpers ----
def compute_resultant(x, y):
    return np.sqrt(np.square(x) + np.square(y))

def pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 1:
        return np.nan
    d = a[mask] - b[mask]
    return float(np.sqrt(np.mean(d * d)))

def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 1:
        return np.nan
    return float(np.mean(np.abs(a[mask] - b[mask])))

if uploaded is None:
    st.info("Carregue um arquivo para começar.")
    st.stop()

df = pd.read_csv(uploaded)

required = ["X_real", "Y_real", "X_modelo", "Y_modelo"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Faltam colunas obrigatórias: {missing}")
    st.write("Colunas encontradas:", list(df.columns))
    st.stop()

# Timestamp (opcional)
time_col = "Timestamp" if "Timestamp" in df.columns else None

with st.sidebar:
    st.header("Opções")
    use_time = st.checkbox("Usar eixo do tempo (Timestamp)", value=(time_col is not None))
    time_in_seconds = st.checkbox("Converter Timestamp para segundos (relativo ao início)", value=True)
    centralizar = st.checkbox("Centralizar trajetos (subtrair 1º ponto de cada série)", value=False)
    show_head = st.checkbox("Mostrar primeiras linhas", value=False)

if show_head:
    st.subheader("Prévia dos dados")
    st.dataframe(df.head(20), use_container_width=True)

# Eixo do tempo
if use_time and time_col is not None:
    t = df[time_col].astype(float).to_numpy()
    if time_in_seconds:
        # Assumindo ms -> s (se já estiver em s, desmarque e use “relativo ao início” abaixo)
        t = (t - t[0]) / 1000.0
        t_label = "Tempo (s)"
    else:
        t = t - t[0]
        t_label = "Tempo (relativo ao início)"
else:
    t = np.arange(len(df), dtype=float)
    t_label = "Amostra (índice)"

x_r = df["X_real"].astype(float).to_numpy()
y_r = df["Y_real"].astype(float).to_numpy()
x_m = df["X_modelo"].astype(float).to_numpy()
y_m = df["Y_modelo"].astype(float).to_numpy()

if centralizar:
    x_r, y_r = x_r - x_r[0], y_r - y_r[0]
    x_m, y_m = x_m - x_m[0], y_m - y_m[0]

# Slider de intervalo (trecho)
n = len(df)
i0, i1 = st.sidebar.slider("Trecho para análise (índices)", 0, n - 1, (0, n - 1))
if i1 <= i0:
    st.error("Trecho inválido: o índice final precisa ser maior que o inicial.")
    st.stop()

sl = slice(i0, i1 + 1)

t_s = t[sl]
x_r_s, y_r_s = x_r[sl], y_r[sl]
x_m_s, y_m_s = x_m[sl], y_m[sl]

# Resultantes ao (0,0)
R_real = compute_resultant(x_r_s, y_r_s)
R_modelo = compute_resultant(x_m_s, y_m_s)

# Erro euclidiano ponto-a-ponto (real vs modelo)
E = compute_resultant(x_r_s - x_m_s, y_r_s - y_m_s)

# Correlações
corr_R = pearson_corr(R_real, R_modelo)
corr_X = pearson_corr(x_r_s, x_m_s)
corr_Y = pearson_corr(y_r_s, y_m_s)

# Erros e métricas
rmse_x = rmse(x_r_s, x_m_s)
rmse_y = rmse(y_r_s, y_m_s)
rmse_e = float(np.sqrt(np.mean(E[np.isfinite(E)] ** 2))) if np.isfinite(E).sum() else np.nan

mae_x = mae(x_r_s, x_m_s)
mae_y = mae(y_r_s, y_m_s)
mae_e = float(np.mean(np.abs(E[np.isfinite(E)]))) if np.isfinite(E).sum() else np.nan

# ---- Layout ----
st.subheader("Métricas principais")
m1, m2, m3, m4 = st.columns(4)
m1.metric("corr(R_real, R_modelo)", f"{corr_R:.4f}" if np.isfinite(corr_R) else "NaN")
m2.metric("corr(X_real, X_modelo)", f"{corr_X:.4f}" if np.isfinite(corr_X) else "NaN")
m3.metric("corr(Y_real, Y_modelo)", f"{corr_Y:.4f}" if np.isfinite(corr_Y) else "NaN")
m4.metric("RMSE erro euclidiano E(t)", f"{rmse_e:.4f}" if np.isfinite(rmse_e) else "NaN")

m5, m6, m7, m8, m9, m10 = st.columns(6)
m5.metric("RMSE(X)", f"{rmse_x:.4f}" if np.isfinite(rmse_x) else "NaN")
m6.metric("RMSE(Y)", f"{rmse_y:.4f}" if np.isfinite(rmse_y) else "NaN")
m7.metric("MAE(X)", f"{mae_x:.4f}" if np.isfinite(mae_x) else "NaN")
m8.metric("MAE(Y)", f"{mae_y:.4f}" if np.isfinite(mae_y) else "NaN")
m9.metric("MAE(E)", f"{mae_e:.4f}" if np.isfinite(mae_e) else "NaN")
m10.metric("N amostras (trecho)", f"{len(t_s)}")
c1,c2,c3 = st.columns(3)

with c2:
    st.subheader("Trajeto XY (Real vs Modelo)")
    fig, ax = plt.subplots()
    ax.plot(x_r_s, y_r_s, label="Real")
    ax.plot(x_m_s, y_m_s, label="Modelo")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with st.expander("Exportar resultados (trecho selecionado)"):
    out = pd.DataFrame({
        "t": t_s,
        "X_real": x_r_s,
        "Y_real": y_r_s,
        "X_modelo": x_m_s,
        "Y_modelo": y_m_s,
        "R_real": R_real,
        "R_modelo": R_modelo,
        "E_erro_euclidiano": E
    })
    st.dataframe(out.head(30), use_container_width=True)
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV do trecho com R e E", data=csv,
                       file_name="trecho_real_modelo_R_E.csv", mime="text/csv")
