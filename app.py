"""Random Forest para proyectar los meses restantes del año por planta (Streamlit).

Archivo único: todo el código vive aquí para que el repositorio solo necesite app.py y requirements.txt.
Ejecutar local:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# ==========================================================================================
# CONFIGURACIÓN
# ==========================================================================================
# Plantas en alcance. Llave = nombre normalizado (mayúsculas, sin espacios extra).
PLANTS = {
    "MOUNT ROYAL": "Mount Royal",
    "BEAVER DAM": "Beaver Dam",
    "CHAMPAIGN": "Champaign",
    "LOWVILLE": "Lowville",
    "FREMONT": "Fremont",
    "MUSCATINE": "Muscatine",
    "JACKSONVILLE (HEINZ)": "Jacksonville (Heinz)",
    "CEDAR RAPIDS": "Cedar Rapids",
    "MASON": "Mason",
    "ESCALON": "Escalon",
    "HOLLAND": "Holland",
}
PLANT_ORDER = list(PLANTS.values())

# Cost buckets tal como vienen en la columna "Database Cost Bucket" (normalizados a mayúsculas).
BUCKETS = {
    "LABOR": "Labor",
    "PEOPLE COMPENSATION": "People Compensation",
    "MAINTENANCE": "Maintenance",
    "OTHER FIXED": "Other Fixed",
    "UTILITIES": "Utilities",
    "OTHER VIC": "Other VIC",
    "MUV": "MUV",
    "OTHER NICC": "Other NICC",
    "FAULTY": "Faulty",
    "SUPPLY CHAIN LOSSES": "Supply Chain Losses",
}
RECOVERY = "Recovery"
SERIES_ORDER = list(BUCKETS.values()) + [RECOVERY]
NET = "Spend neto"

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_PMDB = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

PY_YEAR = 2025
CY_YEAR = 2026

# Modelo
SCALE_FLOOR = 10.0          # USD 000's: piso del denominador del índice (USD 10K)
N_LAGS = 3                  # meses reales previos usados como variables y como base del índice
MIN_LEAF = 3                # fijo por configuración aprobada
MAX_FEATURES = 0.5          # fijo por configuración aprobada
SEED = 42                   # fijo por configuración aprobada
DEFAULT_TREES = 500
MIN_CUT_MONTH = 5           # último mes real más temprano permitido en el selector (mayo)

# Clasificación
DEFAULT_PCT = 5.0           # %
DEFAULT_AMT = 25.0          # USD 000's (= USD 25K)
CLASSES = ["Riesgo", "En línea", "Oportunidad"]

# Colores (paleta validada con el validador de la skill dataviz, modo claro)
C_REAL = "#2a78d6"      # azul: real
C_RF = "#eb6834"        # naranja: Random Forest
C_84 = "#1baf7a"        # aqua: forecast 8+4
C_PY = "#898781"        # gris: PY (contexto)
C_B1 = "#4a3aa7"        # violeta: promedio 3 meses
C_B2 = "#eda100"        # amarillo: PY x variación YTD
C_RISK = "#d03b3b"      # estado crítico
C_OK = "#c3c2b7"        # neutro
C_OPP = "#0ca30c"       # estado bueno
CLASS_COLORS = {"Riesgo": C_RISK, "En línea": C_OK, "Oportunidad": C_OPP}


# ==========================================================================================
# LECTURA DEL ARCHIVO
# ==========================================================================================
SHEETS = ["Actuals", "PY", "PM Database"]


@dataclass
class Dataset:
    values: pd.DataFrame          # plant, series, t, value (t=1..24; 2026 incluye meses de forecast)
    volume: pd.DataFrame          # plant, t, volume (FG Production Volume)
    plan: pd.DataFrame            # plant, series, t, infl, gs (PM Database, t=13..24)
    detected_cut: int             # último mes real detectado en Actuals (1..12)
    notes: list[str] = field(default_factory=list)


def t_index(year: int, month: int) -> int:
    """Índice mensual continuo: ene-2025 = 1 ... dic-2026 = 24."""
    return (year - PY_YEAR) * 12 + month


def t_label(t: int) -> str:
    year = PY_YEAR + (t - 1) // 12
    return f"{MONTHS_ES[(t - 1) % 12]}-{str(year)[2:]}"


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def _find_header(raw: pd.DataFrame, month_label: str) -> int:
    for i in range(min(15, len(raw))):
        row = set(_norm(raw.iloc[i]).tolist())
        if "PLANT" in row and month_label.upper() in row:
            return i
    raise ValueError(f"No se encontró la fila de encabezado (Plant / {month_label}).")


def _detect_cut(raw: pd.DataFrame, header_row: int) -> int | None:
    """Cuenta las marcas YTD sobre las columnas de meses (hoja Actuals)."""
    hdr = _norm(raw.iloc[header_row])
    month_cols = [j for j, v in enumerate(hdr) if v in [m.upper() for m in MONTHS_EN]][:12]
    for i in range(header_row):
        flags = _norm(raw.iloc[i, month_cols])
        if flags.isin(["YTD", "YTG"]).sum() >= 10:
            return int((flags == "YTD").sum())
    return None


def _cost_sheet(raw: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    h = _find_header(raw, "Jan")
    df = raw.iloc[h + 1:].copy()
    df.columns = _norm(raw.iloc[h]).tolist()
    df = df[df["PLANT"].notna()]
    df["PLANT_N"] = _norm(df["PLANT"])
    df = df[df["PLANT_N"].isin(PLANTS.keys())].copy()
    months = [m.upper() for m in MONTHS_EN]
    for m in months:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)
    df["DT"] = _norm(df["DATA TYPE"])
    df["BUCKET"] = _norm(df["DATABASE COST BUCKET"])
    df["CT"] = _norm(df["COST TYPE"])

    spend = df[df["DT"] == "ACTUALS - ADJUSTED CCTR"].copy()
    spend = spend[spend["BUCKET"].isin(BUCKETS.keys())]
    spend["series"] = spend["BUCKET"].map(BUCKETS)
    rec = df[df["DT"] == "RECOVERY"].copy()
    rec["series"] = RECOVERY
    vals = pd.concat([spend, rec])
    vals = vals.groupby(["PLANT_N", "series"])[months].sum().reset_index()
    vals = vals.melt(id_vars=["PLANT_N", "series"], var_name="m", value_name="value")

    vol = df[(df["DT"] == "VOLUME") & (df["CT"] == "FG PRODUCTION VOLUME")]
    vol = vol.groupby("PLANT_N")[months].sum().reset_index()
    vol = vol.melt(id_vars=["PLANT_N"], var_name="m", value_name="volume")

    for d in (vals, vol):
        d["t"] = d["m"].map({m: t_index(year, i + 1) for i, m in enumerate(months)})
        d["plant"] = d["PLANT_N"].map(PLANTS)
    return vals[["plant", "series", "t", "value"]], vol[["plant", "t", "volume"]], h


def _pmdb(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    h = _find_header(raw, "January")
    df = raw.iloc[h + 1:].copy()
    df.columns = _norm(raw.iloc[h]).tolist()
    df = df[df["PLANT"].notna()].copy()
    scen = ", ".join(sorted(set(_norm(df["SCENARIO"]).tolist()) - {""}))
    df["PLANT_N"] = _norm(df["PLANT"])
    df = df[df["PLANT_N"].isin(PLANTS.keys())].copy()
    df["BBP"] = _norm(df["BBP BUCKET"])
    df["BUCKET"] = _norm(df["COST BUCKET"])
    df = df[df["BBP"].isin(["INFLATION", "GROSS SAVINGS"]) & df["BUCKET"].isin(BUCKETS.keys())]
    months = [m.upper() for m in MONTHS_PMDB]
    for m in months:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)
    g = df.groupby(["PLANT_N", "BUCKET", "BBP"])[months].sum().reset_index()
    g = g.melt(id_vars=["PLANT_N", "BUCKET", "BBP"], var_name="m", value_name="v")
    g["t"] = g["m"].map({m: t_index(CY_YEAR, i + 1) for i, m in enumerate(months)})
    p = g.pivot_table(index=["PLANT_N", "BUCKET", "t"], columns="BBP", values="v", aggfunc="sum").reset_index()
    p = p.rename(columns={"INFLATION": "infl", "GROSS SAVINGS": "gs"})
    for c in ("infl", "gs"):
        if c not in p:
            p[c] = 0.0
    p["plant"] = p["PLANT_N"].map(PLANTS)
    p["series"] = p["BUCKET"].map(BUCKETS)
    return p[["plant", "series", "t", "infl", "gs"]], scen


def load_dataset(file) -> Dataset:
    """Lee el .xlsb (ruta o archivo subido) y devuelve las series de las 11 plantas."""
    raw = pd.read_excel(file, sheet_name=SHEETS, engine="pyxlsb", header=None)
    notes: list[str] = []

    act_vals, act_vol, h_act = _cost_sheet(raw["Actuals"], CY_YEAR)
    py_vals, py_vol, _ = _cost_sheet(raw["PY"], PY_YEAR)
    plan, scen = _pmdb(raw["PM Database"])

    cut = _detect_cut(raw["Actuals"], h_act)
    if cut is None:
        cut = 8
        notes.append("No se encontraron las marcas YTD/YTG en Actuals; se asume agosto como último mes real.")
    label_a1 = str(raw["Actuals"].iloc[0, 0]).strip()
    if label_a1 and label_a1.lower() != "nan" and scen and label_a1 not in scen:
        notes.append(f"Versión inconsistente: Actuals!A1 dice '{label_a1}' y PM Database dice '{scen}'. "
                     f"Se usa el corte de las marcas YTD/YTG ({MONTHS_ES[cut - 1]}). Requiere validación.")

    hdr = _norm(raw["Actuals"].iloc[h_act]).tolist()
    plant_raw = raw["Actuals"].iloc[h_act + 1:, hdr.index("PLANT")].dropna().astype(str).str.strip()
    for key, name in PLANTS.items():
        variants = sorted(set(plant_raw[_norm(plant_raw) == key]))
        if len(variants) > 1:
            notes.append(f"{name} aparece escrita como {variants}; se consolidan en una sola planta.")

    values = pd.concat([py_vals, act_vals], ignore_index=True)
    volume = pd.concat([py_vol, act_vol], ignore_index=True)
    # Rejilla completa planta x serie x mes (rellena con 0 si una combinación no existe)
    grid = pd.MultiIndex.from_product([PLANT_ORDER, SERIES_ORDER, range(1, 25)],
                                      names=["plant", "series", "t"]).to_frame(index=False)
    values = grid.merge(values, how="left", on=["plant", "series", "t"]).fillna({"value": 0.0})
    vgrid = pd.MultiIndex.from_product([PLANT_ORDER, range(1, 25)], names=["plant", "t"]).to_frame(index=False)
    volume = vgrid.merge(volume, how="left", on=["plant", "t"]).fillna({"volume": 0.0})

    missing = sorted(set(PLANT_ORDER) - set(act_vals["plant"]))
    if missing:
        notes.append(f"Plantas sin datos en Actuals: {missing}.")
    return Dataset(values=values, volume=volume, plan=plan, detected_cut=cut, notes=notes)


# ==========================================================================================
# VARIABLES DEL MODELO
# ==========================================================================================
GROUPS = {
    "base": "Base: planta, serie, mes, volumen, 3 meses previos, tamaño",
    "anual": "Anual: valor y volumen PY del mismo mes",
    "plan": "Plan: Inflation y Gross Savings (PM Database)",
}

# Columnas por grupo (las de planta y serie se agregan como one-hot)
BASE_NUM = ["mes", "vol_log", "vol_ratio", "lag0", "lag1", "lag2", "tamano_log", "signo"]
ANUAL = ["py_idx", "py_vol_ratio"]
PLAN = ["infl_idx", "gs_idx"]
PLANT_COLS = [f"planta_{p}" for p in PLANT_ORDER]
SERIES_COLS = [f"serie_{s}" for s in SERIES_ORDER]

# Grupos para la importancia por permutación
IMPORTANCE_GROUPS = {
    "Planta": PLANT_COLS,
    "Serie": SERIES_COLS,
    "Mes": ["mes"],
    "Volumen del mes": ["vol_log", "vol_ratio"],
    "3 meses previos": ["lag0", "lag1", "lag2"],
    "Tamaño de la serie": ["tamano_log", "signo"],
    "Valor PY mismo mes": ["py_idx"],
    "Volumen PY mismo mes": ["py_vol_ratio"],
    "Inflation (PM Database)": ["infl_idx"],
    "Gross Savings (PM Database)": ["gs_idx"],
}


def feature_columns(groups: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    if "base" in groups:
        cols += PLANT_COLS + SERIES_COLS + BASE_NUM
    if "anual" in groups:
        cols += ANUAL
    if "plan" in groups:
        cols += PLAN
    return cols


@dataclass
class Prepared:
    keys: pd.DataFrame        # plant, series por fila (121 filas)
    Y: np.ndarray             # (121, 25) valores; columna 0 sin uso, t = 1..24
    V: np.ndarray             # (121, 25) volumen FG de la planta de cada fila
    INFL: np.ndarray          # (121, 25) NaN en 2025 y en Recovery
    GS: np.ndarray            # (121, 25)
    plant_idx: np.ndarray     # índice de planta por fila
    series_idx: np.ndarray    # índice de serie por fila
    V_plant: np.ndarray       # (11, 25) volumen por planta


def prepare(ds: Dataset) -> Prepared:
    keys = pd.MultiIndex.from_product([PLANT_ORDER, SERIES_ORDER], names=["plant", "series"]).to_frame(index=False)
    Yw = ds.values.pivot_table(index=["plant", "series"], columns="t", values="value", aggfunc="sum")
    Yw = Yw.reindex(pd.MultiIndex.from_frame(keys)).reindex(columns=range(1, 25)).fillna(0.0)
    Y = np.zeros((len(keys), 25))
    Y[:, 1:] = Yw.to_numpy()

    Vp = ds.volume.pivot_table(index="plant", columns="t", values="volume", aggfunc="sum")
    Vp = Vp.reindex(PLANT_ORDER).reindex(columns=range(1, 25)).fillna(0.0)
    V_plant = np.zeros((len(PLANT_ORDER), 25))
    V_plant[:, 1:] = Vp.to_numpy()
    plant_idx = keys["plant"].map({p: i for i, p in enumerate(PLANT_ORDER)}).to_numpy()
    series_idx = keys["series"].map({s: i for i, s in enumerate(SERIES_ORDER)}).to_numpy()
    V = V_plant[plant_idx]

    INFL = np.full((len(keys), 25), np.nan)
    GS = np.full((len(keys), 25), np.nan)
    if len(ds.plan):
        pl = ds.plan.set_index(["plant", "series", "t"])
        for r, (p, s) in enumerate(zip(keys["plant"], keys["series"])):
            if s == RECOVERY:
                continue
            for t in range(13, 25):
                if (p, s, t) in pl.index:
                    INFL[r, t] = pl.at[(p, s, t), "infl"]
                    GS[r, t] = pl.at[(p, s, t), "gs"]
                else:
                    INFL[r, t] = 0.0
                    GS[r, t] = 0.0
    return Prepared(keys, Y, V, INFL, GS, plant_idx, series_idx, V_plant)


def series_scale(Y: np.ndarray, o: int) -> tuple[np.ndarray, np.ndarray]:
    m = Y[:, o - N_LAGS + 1:o + 1].mean(axis=1)
    scale = np.maximum(np.abs(m), SCALE_FLOOR)
    sign = np.where(m < 0, -1.0, 1.0)
    return scale, sign


def rows_for_origin(prep: Prepared, o: int, h: int, V: np.ndarray | None = None) -> pd.DataFrame:
    """Filas (121) desde el origen o para el mes objetivo o + h. `V` permite sustituir el volumen."""
    V = prep.V if V is None else V
    Y = prep.Y
    tau = o + h
    n = Y.shape[0]
    scale, sign = series_scale(Y, o)
    vbase = np.maximum(V[:, o - N_LAGS + 1:o + 1].mean(axis=1), 1.0)
    d = {
        "plant": prep.keys["plant"].to_numpy(),
        "series": prep.keys["series"].to_numpy(),
        "origin": np.full(n, o),
        "t": np.full(n, tau),
        "h": np.full(n, h),
        "scale": scale,
        "y": Y[:, tau] if tau <= 24 else np.full(n, np.nan),
        "mes": np.full(n, (tau - 1) % 12 + 1, dtype=float),
        "vol_log": np.log1p(np.maximum(V[:, tau], 0.0)),
        "vol_ratio": V[:, tau] / vbase,
        "lag0": Y[:, o] / scale,
        "lag1": Y[:, o - 1] / scale,
        "lag2": Y[:, o - 2] / scale,
        "tamano_log": np.log10(scale),
        "signo": sign,
    }
    if tau - 12 >= 1:
        d["py_idx"] = Y[:, tau - 12] / scale
        d["py_vol_ratio"] = V[:, tau] / np.maximum(V[:, tau - 12], 1.0)
    else:
        d["py_idx"] = np.full(n, np.nan)
        d["py_vol_ratio"] = np.full(n, np.nan)
    d["infl_idx"] = prep.INFL[:, tau] / scale
    d["gs_idx"] = prep.GS[:, tau] / scale
    df = pd.DataFrame(d)
    for i, p in enumerate(PLANT_ORDER):
        df[PLANT_COLS[i]] = (prep.plant_idx == i).astype(float)
    for i, s in enumerate(SERIES_ORDER):
        df[SERIES_COLS[i]] = (prep.series_idx == i).astype(float)
    df["y_idx"] = df["y"] / df["scale"]
    return df


def training_rows(prep: Prepared, h: int, last_real_t: int) -> pd.DataFrame:
    """Todas las filas con mes objetivo <= último mes real (sin fuga de información)."""
    frames = [rows_for_origin(prep, o, h) for o in range(N_LAGS, last_real_t - h + 1)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ==========================================================================================
# MODELO, CLASIFICACIÓN Y VALIDACIÓN
# ==========================================================================================
VOL_MODES = {"84": "Forecast 8+4 de la planta", "py": "PY del mismo mes", "ytd": "PY × variación YTD de la planta"}
REF_MODES = {"84": "Forecast 8+4", "py": "PY del mismo mes"}


# ---------------------------------------------------------------- Random Forest
def make_rf(n_trees: int, max_depth: int | None) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=MIN_LEAF,
                                 max_features=MAX_FEATURES, random_state=SEED, n_jobs=-1)


def fit_horizon(prep: Prepared, h: int, last_real_t: int, groups: tuple[str, ...],
                n_trees: int, max_depth: int | None):
    tr = training_rows(prep, h, last_real_t)
    cols = feature_columns(groups)
    rf = make_rf(n_trees, max_depth)
    rf.fit(tr[cols].to_numpy(dtype=float), tr["y_idx"].to_numpy(dtype=float))
    return rf, cols, len(tr)


def train_all(prep: Prepared, last_real_t: int, groups, n_trees, max_depth) -> dict:
    H = 24 - last_real_t
    out = {}
    for h in range(1, H + 1):
        rf, cols, n = fit_horizon(prep, h, last_real_t, groups, n_trees, max_depth)
        out[h] = {"model": rf, "cols": cols, "n_rows": n}
    return out


def tree_matrix(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return np.stack([est.predict(X) for est in rf.estimators_], axis=1)


# ---------------------------------------------------------------- Escenario de volumen
def scenario_volume(prep: Prepared, last_real_t: int, mode: str, pct: dict[str, float]) -> np.ndarray:
    """Volumen por planta (11 x 25) con los meses futuros según el escenario elegido."""
    Vp = prep.V_plant.copy()
    for i, plant in enumerate(PLANT_ORDER):
        if mode == "ytd":
            m0 = last_real_t - 12
            num = prep.V_plant[i, 13:last_real_t + 1].sum()
            den = prep.V_plant[i, 1:m0 + 1].sum()
            ratio = num / den if den > 0 else 1.0
        for t in range(last_real_t + 1, 25):
            if mode == "py":
                Vp[i, t] = prep.V_plant[i, t - 12]
            elif mode == "ytd":
                Vp[i, t] = prep.V_plant[i, t - 12] * ratio
            Vp[i, t] *= 1.0 + pct.get(plant, 0.0) / 100.0
    return Vp


def volume_range_warnings(prep: Prepared, Vp: np.ndarray, last_real_t: int) -> pd.DataFrame:
    rows = []
    for i, plant in enumerate(PLANT_ORDER):
        hist = prep.V_plant[i, 1:last_real_t + 1]
        lo, hi = hist.min(), hist.max()
        for t in range(last_real_t + 1, 25):
            v = Vp[i, t]
            if v > hi or v < lo:
                rows.append({"Planta": plant, "Mes": t, "Volumen escenario": v, "Mínimo histórico": lo,
                             "Máximo histórico": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Predicción
def predict_future(prep: Prepared, models: dict, last_real_t: int, Vp: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Predicción de cada mes futuro. Devuelve tabla y matriz por árbol (USD 000's)."""
    V = Vp[prep.plant_idx]
    frames, trees = [], []
    for h, m in models.items():
        rows = rows_for_origin(prep, last_real_t, h, V=V)
        X = rows[m["cols"]].to_numpy(dtype=float)
        tm = tree_matrix(m["model"], X) * rows["scale"].to_numpy()[:, None]
        rows["pred"] = m["model"].predict(X) * rows["scale"].to_numpy()
        frames.append(rows[["plant", "series", "t", "h", "pred", "scale"]])
        trees.append(tm)
    pred = pd.concat(frames, ignore_index=True)
    T = np.vstack(trees)
    pred["p10"] = np.percentile(T, 10, axis=1)
    pred["p90"] = np.percentile(T, 90, axis=1)
    # Referencias: forecast de la hoja Actuals (8+4) y PY del mismo mes
    Y = prep.Y
    r = {(p, s): i for i, (p, s) in enumerate(zip(prep.keys["plant"], prep.keys["series"]))}
    idx = np.array([r[(p, s)] for p, s in zip(pred["plant"], pred["series"])])
    pred["ref_84"] = Y[idx, pred["t"].to_numpy()]
    pred["ref_py"] = Y[idx, pred["t"].to_numpy() - 12]
    return pred, T


def aggregate_net(pred: pd.DataFrame, T: np.ndarray) -> pd.DataFrame:
    """Spend neto por planta y mes (suma de las 11 series) con banda por árbol."""
    out = []
    for (plant, t), g in pred.groupby(["plant", "t"]):
        tt = T[g.index.to_numpy()].sum(axis=0)
        out.append({"plant": plant, "series": NET, "t": t, "h": g["h"].iloc[0],
                    "pred": g["pred"].sum(), "p10": np.percentile(tt, 10), "p90": np.percentile(tt, 90),
                    "ref_84": g["ref_84"].sum(), "ref_py": g["ref_py"].sum()})
    return pd.DataFrame(out)


def classify(pred: np.ndarray, ref: np.ndarray, pct: float, amt: float) -> np.ndarray:
    """(Income)/Expense: más alto = más costo. Riesgo si supera % y monto; Oportunidad al revés."""
    diff = np.asarray(pred, float) - np.asarray(ref, float)
    band = np.abs(np.asarray(ref, float)) * pct / 100.0
    cls = np.full(diff.shape, "En línea", dtype=object)
    cls[(diff > band) & (diff > amt)] = "Riesgo"
    cls[(diff < -band) & (diff < -amt)] = "Oportunidad"
    return cls


# ---------------------------------------------------------------- Validación
def _baseline_ytd(Y: np.ndarray, o: int, tau: int) -> np.ndarray:
    num = Y[:, 13:o + 1].sum(axis=1)
    den = Y[:, 1:o - 12 + 1].sum(axis=1)
    ratio = np.where(np.abs(den) >= SCALE_FLOOR, num / np.where(den == 0, 1, den), 1.0)
    ratio = np.clip(ratio, 0.25, 4.0)
    return Y[:, tau - 12] * ratio


def walk_forward(prep: Prepared, last_real_t: int, groups, n_trees, max_depth, max_h: int) -> pd.DataFrame:
    """Cortes con último mes real de ene al mes anterior al corte; predice hasta el último mes real."""
    Y = prep.Y
    rows = []
    for o in range(13, last_real_t):
        for h in range(1, min(max_h, last_real_t - o) + 1):
            rf, cols, _ = fit_horizon(prep, h, o, groups, n_trees, max_depth)
            ev = rows_for_origin(prep, o, h)          # volumen real del mes objetivo
            X = ev[cols].to_numpy(dtype=float)
            ev["pred"] = rf.predict(X) * ev["scale"].to_numpy()
            ev["b1"] = Y[:, o - N_LAGS + 1:o + 1].mean(axis=1)
            ev["b2"] = _baseline_ytd(Y, o, o + h)
            ev["py"] = Y[:, o + h - 12]
            ev["cut"] = o
            rows.append(ev[["plant", "series", "cut", "h", "t", "y", "pred", "b1", "b2", "py"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y).sum()
    return float(np.abs(y - p).sum() / den) if den > 0 else np.nan


def validation_summary(wf: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            "WAPE RF": wape(g["y"], g["pred"]),
            "WAPE Prom. 3m": wape(g["y"], g["b1"]),
            "WAPE PY×YTD": wape(g["y"], g["b2"]),
            "MAE RF": float(np.abs(g["y"] - g["pred"]).mean()),
            "MAE Prom. 3m": float(np.abs(g["y"] - g["b1"]).mean()),
            "MAE PY×YTD": float(np.abs(g["y"] - g["b2"]).mean()),
            "n": len(g),
        })
    s = wf.groupby(by).apply(agg, include_groups=False).reset_index() if by else agg(wf).to_frame().T
    s["RF gana"] = s["WAPE RF"] <= s[["WAPE Prom. 3m", "WAPE PY×YTD"]].min(axis=1)
    return s


def confusion(wf: pd.DataFrame, pct: float, amt: float) -> pd.DataFrame:
    real = classify(wf["y"], wf["py"], pct, amt)
    pred = classify(wf["pred"], wf["py"], pct, amt)
    cm = pd.crosstab(pd.Categorical(real, CLASSES), pd.Categorical(pred, CLASSES),
                     rownames=["Real"], colnames=["Predicción"], dropna=False)
    return cm


# ---------------------------------------------------------------- Importancia
def grouped_permutation_importance(prep: Prepared, last_real_t: int, groups, n_trees, max_depth,
                                   n_repeats: int = 5) -> pd.DataFrame:
    """Modelo h=1 entrenado hasta 3 meses antes del corte y evaluado en esos 3 meses."""
    o_fit = last_real_t - 3
    rf, cols, _ = fit_horizon(prep, 1, o_fit, groups, n_trees, max_depth)
    ev = pd.concat([rows_for_origin(prep, o, 1) for o in range(o_fit, last_real_t)], ignore_index=True)
    X = ev[cols].to_numpy(dtype=float)
    scale = ev["scale"].to_numpy()
    y = ev["y"].to_numpy()
    base = np.abs(rf.predict(X) * scale - y).mean()
    rng = np.random.default_rng(SEED)
    out = []
    for name, gcols in IMPORTANCE_GROUPS.items():
        idx = [cols.index(c) for c in gcols if c in cols]
        if not idx:
            continue
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = X[perm][:, idx]
            inc.append(np.abs(rf.predict(Xp) * scale - y).mean() - base)
        out.append({"Grupo": name, "Aumento MAE": float(np.mean(inc)), "Desv.": float(np.std(inc))})
    return pd.DataFrame(out).sort_values("Aumento MAE", ascending=False), base


# ==========================================================================================
# GRÁFICAS
# ==========================================================================================
FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13)


def _layout(fig: go.Figure, title: str, height: int = 460, ytitle: str = "USD 000's") -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, xanchor="left"), height=height, font=FONT,
                      margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    fig.update_yaxes(title_text=ytitle, tickformat=",.0f", zeroline=True)
    return fig


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def line_chart(hist: np.ndarray, pred: pd.DataFrame, last_real_t: int, title: str,
               ref_label: str, show_band: bool) -> go.Figure:
    """hist: valores t=0..24 (columna 0 sin uso). pred: filas t>corte con pred, p10, p90."""
    x_all = [t_label(t) for t in range(1, 25)]
    fig = go.Figure()
    real_t = list(range(1, last_real_t + 1))
    fut_t = list(range(last_real_t + 1, 25))
    pred = pred.sort_values("t")
    if show_band and len(pred):
        fig.add_trace(go.Scatter(x=[t_label(t) for t in pred["t"]], y=pred["p90"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[t_label(t) for t in pred["t"]], y=pred["p10"], mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor=_rgba(C_RF, 0.18),
                                 name="Dispersión entre árboles (P10 a P90)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[t_label(t) for t in range(13, 25)], y=hist[1:13], mode="lines",
                             line=dict(color=C_PY, width=1.5), name="PY (mismo mes)",
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=[t_label(t) for t in real_t], y=hist[real_t], mode="lines+markers",
                             line=dict(color=C_REAL, width=2), marker=dict(size=7), name="Real",
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=[t_label(t) for t in [last_real_t] + fut_t], y=hist[[last_real_t] + fut_t],
                             mode="lines+markers",
                             line=dict(color=C_84, width=2, dash="dash"), marker=dict(size=7),
                             name=ref_label, hovertemplate="%{y:,.0f}"))
    xs = [t_label(last_real_t)] + [t_label(t) for t in pred["t"]]
    ys = [hist[last_real_t]] + pred["pred"].tolist()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C_RF, width=2),
                             marker=dict(size=8), name="Random Forest", hovertemplate="%{y:,.0f}"))
    fig.update_xaxes(categoryorder="array", categoryarray=x_all)
    return _layout(fig, title)


def waterfall_chart(rows: pd.DataFrame, ref_col: str, ref_label: str, title: str) -> go.Figure:
    """rows: una fila por serie con pred y referencia acumuladas del periodo."""
    d = rows.assign(diff=rows["pred"] - rows[ref_col])
    d = d.reindex(d["diff"].abs().sort_values(ascending=False).index)
    ref_total = rows[ref_col].sum()
    x = [ref_label] + d["series"].tolist() + ["Random Forest"]
    y = [ref_total] + d["diff"].tolist() + [0]
    measure = ["absolute"] + ["relative"] * len(d) + ["total"]
    text = [f"{ref_total:,.0f}"] + [f"{v:+,.0f}" for v in d["diff"]] + [f"{rows['pred'].sum():,.0f}"]
    fig = go.Figure(go.Waterfall(
        x=x, y=y, measure=measure, text=text, textposition="outside",
        increasing=dict(marker=dict(color=C_RISK)), decreasing=dict(marker=dict(color=C_OPP)),
        totals=dict(marker=dict(color="#52514e")), connector=dict(line=dict(color="#c3c2b7", width=1)),
        hovertemplate="%{x}: %{text}<extra></extra>"))
    fig = _layout(fig, title, height=500)
    # Eje acotado al recorrido acumulado para que los pasos se lean (no inicia en cero)
    cum = np.cumsum([ref_total] + d["diff"].tolist())
    lo, hi = min(cum.min(), rows["pred"].sum()), max(cum.max(), rows["pred"].sum())
    pad = max((hi - lo) * 0.25, abs(hi) * 0.02, 1.0)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def class_heatmap(z: pd.DataFrame, diff: pd.DataFrame, cls: pd.DataFrame, title: str) -> go.Figure:
    """z: -1 Oportunidad, 0 En línea, 1 Riesgo. Texto con símbolo + diferencia (no solo color)."""
    sym = {"Riesgo": "▲", "Oportunidad": "▼", "En línea": "●"}
    text = [[f"{sym[c]} {v:+,.0f}" for c, v in zip(cr, dr)] for cr, dr in zip(cls.values, diff.values)]
    scale = [[0, C_OPP], [1 / 3, C_OPP], [1 / 3, C_OK], [2 / 3, C_OK], [2 / 3, C_RISK], [1, C_RISK]]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=z.columns.tolist(), y=z.index.tolist(), zmin=-1.5, zmax=1.5, colorscale=scale,
        showscale=False, text=text, texttemplate="%{text}", textfont=dict(size=11, color="#0b0b0b"),
        customdata=cls.values, xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>%{customdata}: %{text}<extra></extra>"))
    fig = _layout(fig, title, height=90 + 38 * len(z), ytitle="")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", tickformat=None)
    fig.update_xaxes(side="bottom", tickangle=-30)
    return fig


def validation_bars(s: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = [f"{int(h)} mes{'es' if h > 1 else ''}" for h in s["h"]]
    for col, name, color in [("WAPE RF", "Random Forest", C_RF), ("WAPE Prom. 3m", "Promedio 3 meses", C_B1),
                             ("WAPE PY×YTD", "PY × variación YTD", C_B2)]:
        fig.add_trace(go.Bar(x=x, y=s[col] * 100, name=name, marker=dict(color=color, cornerradius=4),
                             text=[f"{v:.1f}%" for v in s[col] * 100], textposition="outside",
                             hovertemplate="%{y:.1f}%"))
    fig = _layout(fig, "WAPE por meses hacia adelante (menor es mejor)", height=420, ytitle="WAPE %")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest")
    fig.update_yaxes(tickformat=".0f")
    return fig


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    blues = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#184f95"]]
    tot = cm.values.sum()
    text = [[f"{v:,} ({v / tot:.0%})" for v in row] for row in cm.values]
    fig = go.Figure(go.Heatmap(z=cm.values, x=[f"Pred: {c}" for c in cm.columns],
                               y=[f"Real: {c}" for c in cm.index], colorscale=blues, showscale=False,
                               text=text, texttemplate="%{text}", xgap=2, ygap=2,
                               textfont=dict(size=13, color="#0b0b0b"),
                               hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig = _layout(fig, "Clasificación contra PY en la validación", height=360, ytitle="")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", tickformat=None)
    return fig


def importance_bars(imp: pd.DataFrame) -> go.Figure:
    d = imp.sort_values("Aumento MAE")
    fig = go.Figure(go.Bar(x=d["Aumento MAE"], y=d["Grupo"], orientation="h",
                           marker=dict(color=C_REAL, cornerradius=4),
                           customdata=d["Desv."], text=[f"{v:,.1f}" for v in d["Aumento MAE"]],
                           textposition="outside",
                           hovertemplate="%{y}: %{x:,.1f} (desv. est. %{customdata:,.1f})<extra></extra>"))
    fig = _layout(fig, "Importancia por permutación: cuánto sube el MAE al desordenar cada grupo",
                  height=120 + 34 * len(d), ytitle="")
    fig.update_xaxes(title_text="Aumento del MAE (USD 000's)", tickformat=",.0f")
    fig.update_yaxes(tickformat=None)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ==========================================================================================
# APP
# ==========================================================================================
st.set_page_config(page_title="Random Forest por planta", layout="wide")


# ---------------------------------------------------------------- caché
@st.cache_resource(show_spinner="Leyendo el archivo…", max_entries=3)  # sin pickle: las clases viven en este archivo
def cached_dataset(file_bytes: bytes):
    ds = load_dataset(io.BytesIO(file_bytes))
    return ds, prepare(ds)


@st.cache_resource(show_spinner=False, max_entries=6)
def cached_models(_prep, key):
    _, last_real_t, groups, n_trees, depth = key
    return train_all(_prep, last_real_t, groups, n_trees, depth)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_validation(_prep, key):
    _, last_real_t, groups, n_trees, depth = key
    return walk_forward(_prep, last_real_t, groups, n_trees, depth, 24 - last_real_t)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_importance(_prep, key):
    _, last_real_t, groups, n_trees, depth = key
    return grouped_permutation_importance(_prep, last_real_t, groups, n_trees, depth)


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
cut_opts = list(range(min(MIN_CUT_MONTH, detected), detected + 1))
cut = st.sidebar.selectbox("Último mes real", cut_opts, index=len(cut_opts) - 1,
                           format_func=lambda m: f"{MONTHS_ES[m - 1]}-{str(CY_YEAR)[2:]}",
                           help=f"Detectado con las marcas YTD/YTG de Actuals: {MONTHS_ES[detected - 1]}. "
                                "No puede ir más allá, porque los meses siguientes son forecast.")
T0 = 12 + cut
future_t = list(range(T0 + 1, 25))
months_txt = f"{t_label(future_t[0])} a {t_label(future_t[-1])}"

st.sidebar.header("Volumen de los meses futuros")
vol_mode = st.sidebar.radio("Escenario", list(VOL_MODES), format_func=lambda k: VOL_MODES[k].replace("8+4", version))
pct = {}
with st.sidebar.expander("Ajuste de volumen por planta (%)"):
    for p in PLANT_ORDER:
        pct[p] = st.slider(p, -30, 30, 0, 1, key=f"vol_{p}")

st.sidebar.header("Clasificación")
ref_mode = st.sidebar.radio("Referencia", list(REF_MODES), format_func=lambda k: REF_MODES[k].replace("8+4", version))
thr_pct = st.sidebar.slider("Umbral %", 1.0, 20.0, DEFAULT_PCT, 0.5)
thr_amt = st.sidebar.slider("Monto mínimo (USD 000's)", 5.0, 250.0, DEFAULT_AMT, 5.0)

st.sidebar.header("Variables")
groups = tuple(g for g, label in GROUPS.items() if st.sidebar.checkbox(label, value=True, key=f"g_{g}"))

st.sidebar.header("Random Forest")
n_trees = st.sidebar.slider("Número de árboles", 100, 1000, DEFAULT_TREES, 50)
depth_opt = st.sidebar.selectbox("Profundidad máxima", ["Sin límite"] + list(range(3, 21)))
depth = None if depth_opt == "Sin límite" else int(depth_opt)
st.sidebar.caption(f"Fijos: {MIN_LEAF} filas mínimas por hoja, fracción de variables {MAX_FEATURES}, "
                   f"semilla {SEED}.")

if not groups:
    st.error("Prende al menos un grupo de variables.")
    st.stop()

key = (file_hash, T0, groups, n_trees, depth)
ref_col = "ref_84" if ref_mode == "84" else "ref_py"
ref_label = REF_MODES[ref_mode].replace("8+4", version)

# ---------------------------------------------------------------- entrenamiento y predicción
t0 = time.time()
with st.spinner(f"Entrenando {24 - T0} modelos (uno por mes hacia adelante)…"):
    models = cached_models(prep, key)
Vp = scenario_volume(prep, T0, vol_mode, pct)
pred, T = predict_future(prep, models, T0, Vp)
net = aggregate_net(pred, T)
with st.spinner("Validando con cortes sucesivos (la primera vez tarda cerca de un minuto)…"):
    wf = cached_validation(prep, key)
    imp, imp_base = cached_importance(prep, key)
elapsed = time.time() - t0

pred["clase"] = classify(pred["pred"], pred[ref_col], thr_pct, thr_amt)
net["clase"] = classify(net["pred"], net[ref_col], thr_pct, thr_amt)

# ---------------------------------------------------------------- avisos
st.caption(f"USD 000's · convención (Income)/Expense: más alto es más costo · último mes real "
           f"{t_label(T0)} · proyección {months_txt} · referencia de clases: {ref_label}")
for n in ds.notes:
    st.warning(n)
if cut < detected:
    st.info(f"El corte está antes de {MONTHS_ES[detected - 1]}: la referencia {version} de los meses entre el "
            "corte y el último mes real es el dato real.")
warn = volume_range_warnings(prep, Vp, T0)
if len(warn):
    with st.expander(f"Aviso: {len(warn)} combinaciones planta-mes con volumen fuera del rango histórico "
                     "(el Random Forest no extrapola; la predicción se aplana)"):
        w = warn.assign(Mes=warn["Mes"].map(t_label))
        st.dataframe(w.style.format({c: "{:,.0f}" for c in w.columns[2:]}), hide_index=True)

# ---------------------------------------------------------------- KPIs
tot_pred, tot_ref = net["pred"].sum(), net[ref_col].sum()
cum = pred.groupby(["plant", "series"])[["pred", ref_col]].sum().reset_index()
cum["clase"] = classify(cum["pred"], cum[ref_col], thr_pct, thr_amt)
s_all = validation_summary(wf, [])
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
    plant_sel = c1.selectbox("Planta", ["11 plantas"] + PLANT_ORDER, index=1)
    series_sel = c2.selectbox("Serie", [NET] + SERIES_ORDER)
    rows = np.ones(len(prep.keys), bool)
    if plant_sel != "11 plantas":
        rows &= (prep.keys["plant"] == plant_sel).to_numpy()
    if series_sel != NET:
        rows &= (prep.keys["series"] == series_sel).to_numpy()
    hist = prep.Y[rows].sum(axis=0)
    fuente = net if series_sel == NET else pred[pred["series"] == series_sel]
    if plant_sel != "11 plantas":
        fuente = fuente[fuente["plant"] == plant_sel]
    p = fuente.groupby("t")[["pred", "p10", "p90"]].sum().reset_index()
    band = plant_sel != "11 plantas" and series_sel != NET
    st.plotly_chart(line_chart(hist, p, T0, f"{plant_sel} · {series_sel}", f"Forecast {version} (hoja Actuals)",
                                  band), theme="streamlit")
    if not band:
        st.caption("La dispersión entre árboles se muestra solo para una planta y una serie: al sumar series, "
                   "la dispersión de cada árbol se acumula y deja de ser una lectura útil.")
    else:
        st.caption("La banda es la dispersión entre árboles (percentiles 10 y 90). No es un intervalo de "
                   "confianza calibrado.")

# ---------------------------------------------------------------- 2. Cascada
with tabs[1]:
    wplant = st.selectbox("Planta ", ["11 plantas"] + PLANT_ORDER, key="wf_plant")
    d = pred if wplant == "11 plantas" else pred[pred["plant"] == wplant]
    rows_w = d.groupby("series")[["pred", ref_col]].sum().reset_index()
    st.plotly_chart(waterfall_chart(rows_w, ref_col, ref_label,
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
    gn = g.groupby("plant")[["pred", ref_col]].sum().reset_index().assign(series=NET)
    ga = g.groupby("series")[["pred", ref_col]].sum().reset_index().assign(plant="11 plantas")
    gan = ga[["pred", ref_col]].sum().to_frame().T.assign(plant="11 plantas", series=NET)
    allg = pd.concat([g, gn, ga, gan], ignore_index=True)
    allg["diff"] = allg["pred"] - allg[ref_col]
    allg["clase"] = classify(allg["pred"], allg[ref_col], thr_pct, thr_amt)
    allg["z"] = allg["clase"].map({"Oportunidad": -1, "En línea": 0, "Riesgo": 1})
    rows_o, cols_o = PLANT_ORDER + ["11 plantas"], SERIES_ORDER + [NET]
    piv = lambda c: allg.pivot(index="plant", columns="series", values=c).reindex(index=rows_o, columns=cols_o)
    st.plotly_chart(class_heatmap(piv("z"), piv("diff"), piv("clase"),
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
    sh = validation_summary(wf, ["h"])
    st.plotly_chart(validation_bars(sh), theme="streamlit")
    fmt_v = {c: "{:.1%}" for c in ["WAPE RF", "WAPE Prom. 3m", "WAPE PY×YTD"]}
    fmt_v.update({c: "{:,.0f}" for c in ["MAE RF", "MAE Prom. 3m", "MAE PY×YTD", "n"]})
    c1, c2 = st.columns(2)
    ss = validation_summary(wf, ["series"]).rename(columns={"series": "Serie"})
    c1.markdown(f"Por serie: el RF gana en {int(ss['RF gana'].sum())} de {len(ss)}")
    c1.dataframe(ss.style.format(fmt_v), hide_index=True)
    sp = validation_summary(wf, ["plant"]).rename(columns={"plant": "Planta"})
    c2.markdown(f"Por planta: el RF gana en {int(sp['RF gana'].sum())} de {len(sp)}")
    c2.dataframe(sp.style.format(fmt_v), hide_index=True)
    sps = validation_summary(wf, ["plant", "series"])
    with st.expander(f"Planta × serie: el RF gana en {int(sps['RF gana'].sum())} de {len(sps)}"):
        st.dataframe(sps.rename(columns={"plant": "Planta", "series": "Serie"}).style.format(fmt_v),
                     hide_index=True)
    cm = confusion(wf, thr_pct, thr_amt)
    acc = np.trace(cm.values) / cm.values.sum()
    c3, c4 = st.columns([3, 2])
    c3.plotly_chart(confusion_heatmap(cm), theme="streamlit")
    c4.metric("Aciertos de clase contra PY", f"{acc:.1%}")
    c4.caption("La clasificación de la validación solo se puede medir contra PY: el archivo no trae el forecast "
               "que existía antes de cada corte.")

# ---------------------------------------------------------------- 5. Importancia
with tabs[4]:
    st.plotly_chart(importance_bars(imp), theme="streamlit")
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
                         "Valor": [up.name, t_label(T0), VOL_MODES[vol_mode], str({k: v for k, v in pct.items() if v}),
                                   ref_label, thr_pct, thr_amt, ", ".join(groups), n_trees, depth_opt, MIN_LEAF,
                                   MAX_FEATURES, SEED]})
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
