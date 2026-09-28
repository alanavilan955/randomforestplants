"""Random Forest para proyectar los meses restantes del año por planta (Streamlit).

Ejecutar local:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import time

import numpy as np
import pandas as pd
import streamlit as st

from src import charts as CH
from src import config as C
from src import model as M
from src.data_loader import load_dataset, t_label
from src.features import GROUPS, prepare

st.set_page_config(page_title="Random Forest por planta", layout="wide")


# ---------------------------------------------------------------- caché
@st.cache_data(show_spinner="Leyendo el archivo…", max_entries=3)
def cached_dataset(file_bytes: bytes):
    ds = load_dataset(io.BytesIO(file_bytes))
    return ds, prepare(ds)


@st.cache_resource(show_spinner=False, max_entries=6)
def cached_models(_prep, key):
    _, last_real_t, groups, n_trees, depth = key
    return M.train_all(_prep, last_real_t, groups, n_trees, depth)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_validation(_prep, key):
    _, last_real_t, groups, n_trees, depth = key
    return M.walk_forward(_prep, last_real_t, groups, n_trees, depth, 24 - last_real_t)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_importance(_prep, key):
    _, last_real_t, groups, n_trees, depth = key
    return M.grouped_permutation_importance(_prep, last_real_t, groups, n_trees, depth)


def fmt(v: float) -> str:
    return f"{v:,.0f}"


# ---------------------------------------------------------------- barra lateral: datos
st.sidebar.header("Datos")
up = st.sidebar.file_uploader("Performance Model (.xlsb)", type=["xlsb"])
st.title("Random Forest: proyección de los meses restantes por planta")
if up is None:
    st.info("Sube el archivo .xlsb del Performance Model en la barra lateral. La app lee las hojas "
            "Actuals, PY y PM Database de 11 plantas; el archivo no se guarda en el repositorio.")
    st.stop()

file_bytes = up.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()
ds, prep = cached_dataset(file_bytes)
detected = ds.detected_cut
version = f"{detected}+{12 - detected}"

# ---------------------------------------------------------------- barra lateral: controles
st.sidebar.header("Corte")
cut_opts = list(range(min(C.MIN_CUT_MONTH, detected), detected + 1))
cut = st.sidebar.selectbox("Último mes real", cut_opts, index=len(cut_opts) - 1,
                           format_func=lambda m: f"{C.MONTHS_ES[m - 1]}-{str(C.CY_YEAR)[2:]}",
                           help=f"Detectado con las marcas YTD/YTG de Actuals: {C.MONTHS_ES[detected - 1]}. "
                                "No puede ir más allá, porque los meses siguientes son forecast.")
T0 = 12 + cut
future_t = list(range(T0 + 1, 25))
months_txt = f"{t_label(future_t[0])} a {t_label(future_t[-1])}"

st.sidebar.header("Volumen de los meses futuros")
vol_mode = st.sidebar.radio("Escenario", list(M.VOL_MODES), format_func=lambda k: M.VOL_MODES[k].replace("8+4", version))
pct = {}
with st.sidebar.expander("Ajuste de volumen por planta (%)"):
    for p in C.PLANT_ORDER:
        pct[p] = st.slider(p, -30, 30, 0, 1, key=f"vol_{p}")

st.sidebar.header("Clasificación")
ref_mode = st.sidebar.radio("Referencia", list(M.REF_MODES), format_func=lambda k: M.REF_MODES[k].replace("8+4", version))
thr_pct = st.sidebar.slider("Umbral %", 1.0, 20.0, C.DEFAULT_PCT, 0.5)
thr_amt = st.sidebar.slider("Monto mínimo (USD 000's)", 5.0, 250.0, C.DEFAULT_AMT, 5.0)

st.sidebar.header("Variables")
groups = tuple(g for g, label in GROUPS.items() if st.sidebar.checkbox(label, value=True, key=f"g_{g}"))

st.sidebar.header("Random Forest")
n_trees = st.sidebar.slider("Número de árboles", 100, 1000, C.DEFAULT_TREES, 50)
depth_opt = st.sidebar.selectbox("Profundidad máxima", ["Sin límite"] + list(range(3, 21)))
depth = None if depth_opt == "Sin límite" else int(depth_opt)
st.sidebar.caption(f"Fijos: {C.MIN_LEAF} filas mínimas por hoja, fracción de variables {C.MAX_FEATURES}, "
                   f"semilla {C.SEED}.")

if not groups:
    st.error("Prende al menos un grupo de variables.")
    st.stop()

key = (file_hash, T0, groups, n_trees, depth)
ref_col = "ref_84" if ref_mode == "84" else "ref_py"
ref_label = M.REF_MODES[ref_mode].replace("8+4", version)

# ---------------------------------------------------------------- entrenamiento y predicción
t0 = time.time()
with st.spinner(f"Entrenando {24 - T0} modelos (uno por mes hacia adelante)…"):
    models = cached_models(prep, key)
Vp = M.scenario_volume(prep, T0, vol_mode, pct)
pred, T = M.predict_future(prep, models, T0, Vp)
net = M.aggregate_net(pred, T)
with st.spinner("Validando con cortes sucesivos (la primera vez tarda cerca de un minuto)…"):
    wf = cached_validation(prep, key)
    imp, imp_base = cached_importance(prep, key)
elapsed = time.time() - t0

pred["clase"] = M.classify(pred["pred"], pred[ref_col], thr_pct, thr_amt)
net["clase"] = M.classify(net["pred"], net[ref_col], thr_pct, thr_amt)

# ---------------------------------------------------------------- avisos
st.caption(f"USD 000's · convención (Income)/Expense: más alto es más costo · último mes real "
           f"{t_label(T0)} · proyección {months_txt} · referencia de clases: {ref_label}")
for n in ds.notes:
    st.warning(n)
if cut < detected:
    st.info(f"El corte está antes de {C.MONTHS_ES[detected - 1]}: la referencia {version} de los meses entre el "
            "corte y el último mes real es el dato real.")
warn = M.volume_range_warnings(prep, Vp, T0)
if len(warn):
    with st.expander(f"Aviso: {len(warn)} combinaciones planta-mes con volumen fuera del rango histórico "
                     "(el Random Forest no extrapola; la predicción se aplana)"):
        w = warn.assign(Mes=warn["Mes"].map(t_label))
        st.dataframe(w.style.format({c: "{:,.0f}" for c in w.columns[2:]}), hide_index=True)

# ---------------------------------------------------------------- KPIs
tot_pred, tot_ref = net["pred"].sum(), net[ref_col].sum()
cum = pred.groupby(["plant", "series"])[["pred", ref_col]].sum().reset_index()
cum["clase"] = M.classify(cum["pred"], cum[ref_col], thr_pct, thr_amt)
s_all = M.validation_summary(wf, [])
k1, k2, k3, k4 = st.columns(4)
k1.metric("Spend neto RF, 11 plantas", fmt(tot_pred),
          help=f"Suma de {months_txt}, USD 000's.",
          delta=f"{tot_pred - tot_ref:+,.0f} vs {ref_label}", delta_color="inverse")
k2.metric(ref_label, fmt(tot_ref), help=f"Suma de {months_txt} de las 11 plantas, USD 000's.")
k3.metric("Riesgo / Oportunidad",
          f"{(cum['clase'] == 'Riesgo').sum()} / {(cum['clase'] == 'Oportunidad').sum()}",
          help="Combinaciones planta × serie (121) en Riesgo y en Oportunidad, acumulado de los meses proyectados.")
best = min(s_all["WAPE Prom. 3m"].iloc[0], s_all["WAPE PY×YTD"].iloc[0])
k4.metric("WAPE del RF en validación", f"{s_all['WAPE RF'].iloc[0]:.1%}",
          help="Error de la validación con cortes sucesivos, comparado con el mejor de los dos métodos simples.",
          delta=f"{(s_all['WAPE RF'].iloc[0] - best) * 100:+.1f} pts vs método simple ({best:.1%})", delta_color="inverse")

tabs = st.tabs(["Serie mensual", "Cascada", "Mapa de clases", "Validación", "Importancia", "Tabla y descarga"])

# ---------------------------------------------------------------- 1. Serie mensual
with tabs[0]:
    c1, c2 = st.columns(2)
    plant_sel = c1.selectbox("Planta", ["11 plantas"] + C.PLANT_ORDER, index=1)
    series_sel = c2.selectbox("Serie", [C.NET] + C.SERIES_ORDER)
    rows = np.ones(len(prep.keys), bool)
    if plant_sel != "11 plantas":
        rows &= (prep.keys["plant"] == plant_sel).to_numpy()
    if series_sel != C.NET:
        rows &= (prep.keys["series"] == series_sel).to_numpy()
    hist = prep.Y[rows].sum(axis=0)
    src = net if series_sel == C.NET else pred[pred["series"] == series_sel]
    if plant_sel != "11 plantas":
        src = src[src["plant"] == plant_sel]
    p = src.groupby("t")[["pred", "p10", "p90"]].sum().reset_index()
    band = plant_sel != "11 plantas" and series_sel != C.NET
    st.plotly_chart(CH.line_chart(hist, p, T0, f"{plant_sel} · {series_sel}", f"Forecast {version} (hoja Actuals)",
                                  band), theme="streamlit")
    if not band:
        st.caption("La dispersión entre árboles se muestra solo para una planta y una serie: al sumar series, "
                   "la dispersión de cada árbol se acumula y deja de ser una lectura útil.")
    else:
        st.caption("La banda es la dispersión entre árboles (percentiles 10 y 90). No es un intervalo de "
                   "confianza calibrado.")

# ---------------------------------------------------------------- 2. Cascada
with tabs[1]:
    wplant = st.selectbox("Planta ", ["11 plantas"] + C.PLANT_ORDER, key="wf_plant")
    d = pred if wplant == "11 plantas" else pred[pred["plant"] == wplant]
    rows_w = d.groupby("series")[["pred", ref_col]].sum().reset_index()
    st.plotly_chart(CH.waterfall_chart(rows_w, ref_col, ref_label,
                                       f"{wplant} · spend neto {months_txt}: de {ref_label} a Random Forest"),
                    theme="streamlit")
    st.caption("Rojo: la serie agrega costo contra la referencia. Verde: lo reduce. En Recovery, más negativo "
               "es más absorción. El eje vertical no inicia en cero para que los pasos se puedan leer.")

# ---------------------------------------------------------------- 3. Mapa de clases
with tabs[2]:
    per_opts = ["Acumulado"] + future_t
    per = st.selectbox("Periodo", per_opts, format_func=lambda v: f"Acumulado {months_txt}" if v == "Acumulado" else t_label(v))
    dp = pred if per == "Acumulado" else pred[pred["t"] == per]
    g = dp.groupby(["plant", "series"])[["pred", ref_col]].sum().reset_index()
    gn = g.groupby("plant")[["pred", ref_col]].sum().reset_index().assign(series=C.NET)
    ga = g.groupby("series")[["pred", ref_col]].sum().reset_index().assign(plant="11 plantas")
    gan = ga[["pred", ref_col]].sum().to_frame().T.assign(plant="11 plantas", series=C.NET)
    allg = pd.concat([g, gn, ga, gan], ignore_index=True)
    allg["diff"] = allg["pred"] - allg[ref_col]
    allg["clase"] = M.classify(allg["pred"], allg[ref_col], thr_pct, thr_amt)
    allg["z"] = allg["clase"].map({"Oportunidad": -1, "En línea": 0, "Riesgo": 1})
    rows_o, cols_o = C.PLANT_ORDER + ["11 plantas"], C.SERIES_ORDER + [C.NET]
    piv = lambda c: allg.pivot(index="plant", columns="series", values=c).reindex(index=rows_o, columns=cols_o)
    st.plotly_chart(CH.class_heatmap(piv("z"), piv("diff"), piv("clase"),
                                     f"Clase contra {ref_label}: Random Forest − referencia (USD 000's)"),
                    theme="streamlit")
    st.caption(f"▲ Riesgo (rojo) · ● En línea (gris) · ▼ Oportunidad (verde). Regla: diferencia mayor a "
               f"{thr_pct:.1f}% de la referencia y a USD {thr_amt:,.0f}K. La misma regla se aplica a cada celda, "
               "al total de planta y a la fila de 11 plantas.")

# ---------------------------------------------------------------- 4. Validación
with tabs[3]:
    st.markdown(f"Se simularon {wf['cut'].nunique()} cortes (último mes real de {t_label(int(wf['cut'].min()))} "
                f"a {t_label(int(wf['cut'].max()))}); {len(wf):,} predicciones evaluadas. En la simulación se "
                "usa el volumen real del mes, así que el error mide al modelo con el volumen conocido.")
    sh = M.validation_summary(wf, ["h"])
    st.plotly_chart(CH.validation_bars(sh), theme="streamlit")
    fmt_v = {c: "{:.1%}" for c in ["WAPE RF", "WAPE Prom. 3m", "WAPE PY×YTD"]}
    fmt_v.update({c: "{:,.0f}" for c in ["MAE RF", "MAE Prom. 3m", "MAE PY×YTD", "n"]})
    c1, c2 = st.columns(2)
    ss = M.validation_summary(wf, ["series"]).rename(columns={"series": "Serie"})
    c1.markdown(f"Por serie: el RF gana en {int(ss['RF gana'].sum())} de {len(ss)}")
    c1.dataframe(ss.style.format(fmt_v), hide_index=True)
    sp = M.validation_summary(wf, ["plant"]).rename(columns={"plant": "Planta"})
    c2.markdown(f"Por planta: el RF gana en {int(sp['RF gana'].sum())} de {len(sp)}")
    c2.dataframe(sp.style.format(fmt_v), hide_index=True)
    sps = M.validation_summary(wf, ["plant", "series"])
    with st.expander(f"Planta × serie: el RF gana en {int(sps['RF gana'].sum())} de {len(sps)}"):
        st.dataframe(sps.rename(columns={"plant": "Planta", "series": "Serie"}).style.format(fmt_v),
                     hide_index=True)
    cm = M.confusion(wf, thr_pct, thr_amt)
    acc = np.trace(cm.values) / cm.values.sum()
    c3, c4 = st.columns([3, 2])
    c3.plotly_chart(CH.confusion_heatmap(cm), theme="streamlit")
    c4.metric("Aciertos de clase contra PY", f"{acc:.1%}")
    c4.caption("La clasificación de la validación solo se puede medir contra PY: el archivo no trae el forecast "
               "que existía antes de cada corte.")

# ---------------------------------------------------------------- 5. Importancia
with tabs[4]:
    st.plotly_chart(CH.importance_bars(imp), theme="streamlit")
    st.caption(f"Modelo de 1 mes entrenado hasta {t_label(T0 - 3)} y evaluado en {t_label(T0 - 2)} a {t_label(T0)} "
               f"(MAE base {imp_base:,.1f}). 5 permutaciones por grupo; la desviación estándar aparece al pasar el cursor.")
    st.dataframe(imp.style.format({"Aumento MAE": "{:,.1f}", "Desv.": "{:,.1f}"}), hide_index=True)

# ---------------------------------------------------------------- 6. Tabla y descarga
with tabs[5]:
    out = pd.concat([pred.drop(columns=["scale"]), net], ignore_index=True)
    out["dif_vs_ref"] = out["pred"] - out[ref_col]
    out["Mes"] = out["t"].map(t_label)
    out = out.rename(columns={"plant": "Planta", "series": "Serie", "h": "Meses adelante", "pred": "RF",
                              "p10": "P10 árboles", "p90": "P90 árboles", "ref_84": f"Forecast {version}",
                              "ref_py": "PY", "dif_vs_ref": f"RF − {ref_label}", "clase": "Clase"})
    cols = ["Planta", "Serie", "Mes", "Meses adelante", "RF", "P10 árboles", "P90 árboles", f"Forecast {version}",
            "PY", f"RF − {ref_label}", "Clase"]
    out = out[cols]
    st.dataframe(out.style.format({c: "{:,.0f}" for c in cols[4:10]}), hide_index=True,
                 height=460)
    conf = pd.DataFrame({"Parámetro": ["Archivo", "Último mes real", "Escenario de volumen", "Ajustes de volumen %",
                                       "Referencia", "Umbral %", "Monto mínimo USD 000's", "Variables",
                                       "Árboles", "Profundidad", "Filas mínimas por hoja", "Fracción de variables",
                                       "Semilla"],
                         "Valor": [up.name, t_label(T0), M.VOL_MODES[vol_mode], str({k: v for k, v in pct.items() if v}),
                                   ref_label, thr_pct, thr_amt, ", ".join(groups), n_trees, depth_opt, C.MIN_LEAF,
                                   C.MAX_FEATURES, C.SEED]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Predicciones", index=False)
        ss.to_excel(xw, sheet_name="Validación por serie", index=False)
        sp.to_excel(xw, sheet_name="Validación por planta", index=False)
        sps.to_excel(xw, sheet_name="Validación planta-serie", index=False)
        imp.to_excel(xw, sheet_name="Importancia", index=False)
        conf.to_excel(xw, sheet_name="Configuración", index=False)
    st.download_button("Descargar Excel", buf.getvalue(), file_name=f"rf_prediccion_{version}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption(f"Tiempo de esta corrida: {elapsed:.1f} s (los resultados quedan en caché mientras no cambien el "
           "archivo, el corte, las variables o los parámetros del Random Forest).")
