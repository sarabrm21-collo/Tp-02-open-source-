# -*- coding: utf-8 -*-
"""
APPLICATION 1.1 - TABLEAU DE BORD INTERACTIF AVEC PLOTLY
V05 : Importation Excel/CSV et analyse multivariee.

Fonctionnalites :
- Importation Excel (.xlsx, .xls) et CSV (.csv).
- Selection de la colonne Date et de la variable numerique a analyser.
- Filtres temporels et detection des valeurs atypiques (IQR / Z-score).
- KPI, quatre graphiques interactifs et tableau des anomalies.
- Fonctionnement local dans le navigateur.

Installation :
    conda install -c conda-forge numpy pandas plotly dash openpyxl
Pour les anciens fichiers .xls, installer egalement xlrd.
"""

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update


# ============================================================
# 1. PARAMETRES GENERAUX
# ============================================================
APP_TITLE = "TABLEAU DE BORD INTERACTIF - V05"
MOVING_AVERAGE_WINDOW = 7
REFERENCE_VALUE = 20.0
DEFAULT_UNIT = ""
DATE_FORMAT = "%d/%m/%Y"


# ============================================================
# 2. DONNEES DE DEMONSTRATION (UTILISEES AU DEMARRAGE)
# ============================================================
def create_demo_dataset():
    """Cree un jeu de demonstration clairement identifie comme synthetique."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    trend = np.linspace(0, 1.2, len(dates))
    seasonal = 0.8 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))
    temperature = 18 + trend + seasonal + rng.normal(0, 1.5, len(dates))
    temperature[[18, 53, 76]] += [5.5, -5.0, 6.0]

    # Variables synthetiques de qualite de l'eau, uniquement pour la demonstration.
    noise = lambda scale: rng.normal(0, scale, len(dates))
    do = 10.8 - 0.16 * temperature + noise(0.35)
    cod = 18 + 1.7 * temperature + noise(2.0)
    bod = 3.0 + 0.28 * cod + noise(0.8)
    tn = 1.2 + 0.035 * cod + noise(0.18)
    tp = 0.12 + 0.012 * bod + noise(0.035)
    ss = 14 + 2.0 * bod + noise(2.8)

    return pd.DataFrame({
        "Date": dates,
        "Temperature": temperature,
        "DO": do,
        "COD": cod,
        "BOD": bod,
        "TN": tn,
        "TP": tp,
        "SS": ss
    })


def dataframe_to_store(data):
    """Convertit le DataFrame en JSON compatible avec dcc.Store."""
    clean = data.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return clean.to_json(orient="split", date_format="iso")


def store_to_dataframe(json_data):
    if not json_data:
        return pd.DataFrame()
    data = pd.read_json(io.StringIO(json_data), orient="split")
    return data


# ============================================================
# 3. LECTURE ET PREPARATION DES FICHIERS
# ============================================================
def read_uploaded_file(contents, filename):
    """Lit un fichier transmis par dcc.Upload."""
    if not contents or not filename:
        raise ValueError("Aucun fichier n'a ete transmis.")

    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("Le contenu du fichier est illisible.") from exc

    extension = Path(filename).suffix.lower()
    if extension in (".xlsx", ".xls"):
        data = pd.read_excel(io.BytesIO(raw), sheet_name=0)
    elif extension == ".csv":
        try:
            data = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        except UnicodeDecodeError:
            data = pd.read_csv(io.BytesIO(raw), sep=None, engine="python",
                               encoding="latin-1")
    else:
        raise ValueError("Format non pris en charge. Utilisez Excel (.xlsx) ou CSV (.csv).")

    if data.empty:
        raise ValueError("Le fichier ne contient aucune observation.")

    data.columns = [str(c).strip() for c in data.columns]
    # Eviter les noms de colonnes vides ou dupliques.
    new_columns = []
    seen = {}
    for i, col in enumerate(data.columns):
        name = col if col and col.lower() != "nan" else f"Colonne_{i+1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        new_columns.append(name)
    data.columns = new_columns
    return data


def infer_date_column(data):
    """Propose la colonne date la plus plausible."""
    best_col, best_score = None, 0.0
    for col in data.columns:
        name_score = 1.0 if any(k in str(col).lower()
                                for k in ("date", "time", "jour", "temps")) else 0.0
        try:
            parsed = pd.to_datetime(data[col], errors="coerce", dayfirst=True)
            parse_score = float(parsed.notna().mean())
        except Exception:
            parse_score = 0.0
        score = parse_score + name_score * 0.15
        if parse_score >= 0.60 and score > best_score:
            best_col, best_score = col, score
    return best_col


def infer_numeric_column(data, date_col):
    numeric_cols = []
    for col in data.columns:
        if col == date_col:
            continue
        converted = pd.to_numeric(data[col].astype(str).str.replace(",", ".", regex=False),
                                  errors="coerce")
        if converted.notna().sum() >= 2:
            numeric_cols.append((col, int(converted.notna().sum())))
    return max(numeric_cols, key=lambda item: item[1])[0] if numeric_cols else None


def prepare_analysis_data(data, date_col, value_col):
    """Construit un tableau propre Date/Value pour les calculs."""
    if date_col not in data.columns or value_col not in data.columns:
        raise ValueError("Veuillez choisir une colonne Date et une variable numerique valides.")

    result = pd.DataFrame()
    result["Date"] = pd.to_datetime(data[date_col], errors="coerce", dayfirst=True)
    result["Value"] = pd.to_numeric(
        data[value_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    before = len(result)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["Date", "Value"]).sort_values("Date")
    result = result.reset_index(drop=True)
    removed = before - len(result)
    if result.empty:
        raise ValueError("Aucune ligne exploitable : verifiez les colonnes Date et numerique.")
    return result, removed


# ============================================================
# 4. DETECTION DES VALEURS ATYPIQUES ET KPI
# ============================================================
def detect_outliers(data, method="IQR"):
    result = data.copy()
    values = result["Value"].astype(float)

    if method == "IQR":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        description = "IQR : en dehors de [Q1 - 1.5 × IQR ; Q3 + 1.5 × IQR]."
    else:
        mean = values.mean()
        std = values.std(ddof=1)
        if not np.isfinite(std) or std == 0:
            z = pd.Series(0.0, index=values.index)
        else:
            z = (values - mean) / std
        mask = z.abs() > 3
        lower, upper = mean - 3 * (std if np.isfinite(std) else 0), mean + 3 * (std if np.isfinite(std) else 0)
        description = "Z-score : valeurs dont |z| > 3."

    result["Outlier"] = mask
    result["Lower_Bound"] = lower
    result["Upper_Bound"] = upper
    return result, description


def calculate_kpis(data):
    values = data["Value"].astype(float)
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0
    }


def kpi_card(label, value, unit="", accent="#2471A3"):
    return html.Div([
        html.Div(label, style={"fontSize": "13px", "color": "#526273",
                               "fontWeight": "bold", "marginBottom": "8px"}),
        html.Div(f"{value}{unit}", style={"fontSize": "26px", "fontWeight": "bold",
                                          "color": accent, "fontFamily": "Times New Roman"})
    ], style={"backgroundColor": "white", "border": "1px solid #D5DDE7",
              "borderLeft": f"5px solid {accent}", "borderRadius": "8px",
              "padding": "14px 16px", "minWidth": "140px", "flex": "1",
              "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"})


# ============================================================
# 5. GRAPHIQUES PLOTLY
# ============================================================
def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="#526273"))
    fig.update_layout(template="plotly_white", height=350)
    return fig


def build_figures(data, method, variable, unit):
    analysed, _ = detect_outliers(data, method)
    normal = analysed.loc[~analysed["Outlier"]]
    anomalies = analysed.loc[analysed["Outlier"]]
    unit_label = f" ({unit})" if unit else ""

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=normal["Date"], y=normal["Value"], mode="lines+markers",
        name="Observations normales", line=dict(color="#2471A3", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.3f}<extra></extra>"))
    if not anomalies.empty:
        fig_time.add_trace(go.Scatter(
            x=anomalies["Date"], y=anomalies["Value"], mode="markers",
            name="Valeurs atypiques",
            marker=dict(color="#C0392B", size=13, symbol="diamond",
                        line=dict(color="white", width=1)),
            hovertemplate="<b>VALEUR ATYPIQUE</b><br>%{x|%d/%m/%Y}<br>%{y:.3f}<extra></extra>"))
    fig_time.update_layout(title=f"Evolution temporelle de {variable}",
                           xaxis_title="Date", yaxis_title=f"{variable}{unit_label}",
                           template="plotly_white", hovermode="x unified",
                           height=390, legend=dict(orientation="h", y=1.12))

    fig_hist = go.Figure(go.Histogram(
        x=data["Value"], nbinsx=18, name="Effectif",
        marker_color="#55A6A6", marker_line_color="white", marker_line_width=1))
    mean_value = data["Value"].mean()
    fig_hist.add_vline(x=mean_value, line_dash="dash", line_color="#C0392B",
                       annotation_text=f"Moyenne : {mean_value:.3f}")
    fig_hist.update_layout(title=f"Distribution de {variable}",
                           xaxis_title=f"{variable}{unit_label}", yaxis_title="Effectif",
                           template="plotly_white", height=350)

    fig_box = go.Figure(go.Box(
        y=data["Value"], name=variable, boxpoints="outliers",
        marker_color="#7551A1", line_color="#563D7C",
        fillcolor="rgba(117,81,161,0.25)", boxmean=True))
    fig_box.update_layout(title=f"Dispersion de {variable}",
                          yaxis_title=f"{variable}{unit_label}",
                          template="plotly_white", height=350)

    smooth = data.sort_values("Date").copy()
    smooth["Moving_Average"] = smooth["Value"].rolling(
        window=MOVING_AVERAGE_WINDOW, min_periods=1).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=smooth["Date"], y=smooth["Value"],
                                mode="markers", name=variable,
                                marker=dict(color="#A7C7E7", size=7)))
    fig_ma.add_trace(go.Scatter(
        x=smooth["Date"], y=smooth["Moving_Average"], mode="lines",
        name=f"Moyenne mobile ({MOVING_AVERAGE_WINDOW} observations)",
        line=dict(color="#C0392B", width=3)))
    fig_ma.update_layout(title=f"Tendance de {variable} et moyenne mobile",
                         xaxis_title="Date", yaxis_title=f"{variable}{unit_label}",
                         template="plotly_white", height=350, hovermode="x unified")
    return fig_time, fig_hist, fig_box, fig_ma


# ============================================================
# 6. INTERFACE DASH
# ============================================================
demo = create_demo_dataset()
app = Dash(__name__)
app.title = "Dashboard Plotly - V05"

CARD_STYLE = {"backgroundColor": "white", "padding": "14px 16px",
              "border": "1px solid #D5DDE7", "borderRadius": "8px",
              "marginBottom": "12px"}
GRAPH_STYLE = {"backgroundColor": "white", "padding": "5px",
               "border": "1px solid #D5DDE7", "borderRadius": "8px"}

app.layout = html.Div([
    dcc.Store(id="data-store", data=dataframe_to_store(demo)),
    dcc.Store(id="source-store", data="Jeu de demonstration synthetique"),
    html.Div([
        html.H1("TABLEAU DE BORD INTERACTIF",
                style={"margin": "0", "fontFamily": "Times New Roman",
                       "fontSize": "30px", "fontWeight": "bold"}),
        html.H3("Analyse multivariee | Excel, CSV et correlations | V05",
                style={"margin": "5px 0 0 0", "fontWeight": "normal", "fontSize": "17px"})
    ], style={"backgroundColor": "#17365D", "color": "white",
              "padding": "22px 28px", "marginBottom": "14px"}),

    html.Div([
        html.H3("1. IMPORTATION DES DONNEES", style={"color": "#17365D", "marginTop": "0"}),
        dcc.Upload(id="upload-data", children=html.Div([
            "Glissez votre fichier Excel/CSV ici ou ",
            html.A("cliquez pour le selectionner", style={"fontWeight": "bold"})
        ]), multiple=False,
           style={"width": "100%", "height": "65px", "lineHeight": "65px",
                  "borderWidth": "2px", "borderStyle": "dashed",
                  "borderColor": "#2471A3", "borderRadius": "8px",
                  "textAlign": "center", "backgroundColor": "#F7FAFD",
                  "cursor": "pointer"}),
        html.Div(id="upload-status",
                 children="Jeu de demonstration charge. Importez votre fichier pour analyser vos propres mesures.",
                 style={"marginTop": "9px", "color": "#526273", "fontSize": "13px"})
    ], style=CARD_STYLE),

    html.Div([
        html.Div([
            html.Label("Colonne Date", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="date-column", options=[{"label": "Date", "value": "Date"}],
                         value="Date", clearable=False)
        ], style={"flex": "1", "minWidth": "220px"}),
        html.Div([
            html.Label("Variable a analyser", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="value-column", options=[{"label": "Temperature", "value": "Temperature"}],
                         value="Temperature", clearable=False)
        ], style={"flex": "1", "minWidth": "220px"}),
        html.Div([
            html.Label("Methode de detection", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="method-filter", options=[
                {"label": "IQR (1.5 × ecart interquartile)", "value": "IQR"},
                {"label": "Z-score (|z| > 3)", "value": "Z-score"}],
                value="IQR", clearable=False)
        ], style={"flex": "1", "minWidth": "220px"}),
        html.Div([
            html.Label("Affichage des anomalies", style={"fontWeight": "bold"}),
            dcc.RadioItems(id="outlier-filter", options=[
                {"label": "Toutes", "value": "all"},
                {"label": "Masquer les atypiques", "value": "hide"},
                {"label": "Atypiques seulement", "value": "only"}],
                value="all", labelStyle={"display": "block", "marginTop": "5px"})
        ], style={"flex": "1", "minWidth": "190px"})
    ], style={**CARD_STYLE, "display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Div([
        html.Div([
            html.Label("Periode d'analyse", style={"fontWeight": "bold"}),
            dcc.DatePickerRange(id="date-filter", display_format="DD/MM/YYYY",
                                minimum_nights=0, start_date=demo["Date"].min().date(),
                                end_date=demo["Date"].max().date())
        ])
    ], style=CARD_STYLE),

    html.H3("2. INDICATEURS CLES DE PERFORMANCE (KPI)",
            style={"color": "#17365D", "fontSize": "17px"}),
    html.Div(id="kpi-container", style={"display": "flex", "gap": "12px",
                                        "flexWrap": "wrap", "marginBottom": "15px"}),

    html.Div([
        dcc.Graph(id="time-chart", config={"displaylogo": False}),
        dcc.Graph(id="hist-chart", config={"displaylogo": False}),
        dcc.Graph(id="box-chart", config={"displaylogo": False}),
        dcc.Graph(id="ma-chart", config={"displaylogo": False})
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
              "gap": "12px"}),

    html.Div([
        html.H3("3. OBSERVATIONS IDENTIFIEES COMME ATYPIQUES",
                style={"color": "#17365D", "fontSize": "17px"}),
        html.Div(id="outlier-summary", style={"marginBottom": "8px", "color": "#526273"}),
        dash_table.DataTable(
            id="outlier-table",
            columns=[{"name": "Date", "id": "Date"},
                     {"name": "Valeur", "id": "Value"},
                     {"name": "Methode", "id": "Method"}],
            data=[], page_size=10, sort_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#17365D", "color": "white",
                          "fontWeight": "bold", "textAlign": "center"},
            style_cell={"padding": "9px", "textAlign": "center",
                        "fontFamily": "Arial", "fontSize": "13px"},
            style_data_conditional=[{"if": {"row_index": "odd"},
                                     "backgroundColor": "#F2F5F9"}])
    ], style={**CARD_STYLE, "marginTop": "15px"}),

    html.Div([
        html.H3("4. ANALYSE MULTIVARIEE ET RELATIONS ENTRE VARIABLES",
                style={"color": "#17365D", "fontSize": "19px"}),
        html.Div([
            html.Div([
                html.Label("Variables pour la matrice de correlation",
                           style={"fontWeight": "bold"}),
                dcc.Dropdown(id="multi-vars", options=[
                    {"label": c, "value": c}
                    for c in ["Temperature"]
                ], value=["Temperature", "DO", "COD", "BOD"], multi=True,
                   placeholder="Selectionnez plusieurs variables")
            ], style={"flex": "2", "minWidth": "280px"}),
            html.Div([
                html.Label("Coefficient de correlation",
                           style={"fontWeight": "bold"}),
                dcc.Dropdown(id="corr-method", options=[
                    {"label": "Pearson (relation lineaire)", "value": "pearson"},
                    {"label": "Spearman (relation monotone)", "value": "spearman"}
                ], value="pearson", clearable=False)
            ], style={"flex": "1", "minWidth": "220px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "12px"}),
        html.Div([
            html.Div(dcc.Graph(id="corr-chart", config={"displaylogo": False}),
                     style=GRAPH_STYLE),
            html.Div([
                html.Div([
                    html.Label("Variable X", style={"fontWeight": "bold"}),
                    dcc.Dropdown(id="scatter-x", options=[
                        {"label": c, "value": c}
                        for c in ["Temperature", "DO", "COD", "BOD", "TN", "TP", "SS"]
                    ], value="Temperature", clearable=False)
                ], style={"flex": "1", "minWidth": "180px"}),
                html.Div([
                    html.Label("Variable Y", style={"fontWeight": "bold"}),
                    dcc.Dropdown(id="scatter-y", options=[
                        {"label": c, "value": c}
                        for c in ["Temperature", "DO", "COD", "BOD", "TN", "TP", "SS"]
                    ], value="DO", clearable=False)
                ], style={"flex": "1", "minWidth": "180px"})
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                      "marginBottom": "8px"}),
            dcc.Graph(id="scatter-chart", config={"displaylogo": False})
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                  "gap": "12px"})
    ], style={**CARD_STYLE, "marginTop": "16px"}),

    html.Div("Application pedagogique | Importation Excel/CSV | Analyse univariee et multivariee | Pandas + Plotly + Dash",
             style={"textAlign": "center", "color": "#667788",
                    "fontSize": "12px", "padding": "12px"})
], style={"backgroundColor": "#F2F5F9", "minHeight": "100vh",
          "padding": "7px", "fontFamily": "Arial, sans-serif",
          "color": "#263238"})


# ============================================================
# 7. CALLBACK IMPORTATION
# ============================================================
@app.callback(
    Output("data-store", "data"),
    Output("source-store", "data"),
    Output("date-column", "options"),
    Output("date-column", "value"),
    Output("value-column", "options"),
    Output("value-column", "value"),
    Output("upload-status", "children"),
    Output("upload-status", "style"),
    Output("date-filter", "min_date_allowed"),
    Output("date-filter", "max_date_allowed"),
    Output("date-filter", "start_date"),
    Output("date-filter", "end_date"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def load_file(contents, filename):
    if not contents:
        return (no_update,) * 12
    try:
        raw = read_uploaded_file(contents, filename)
        date_col = infer_date_column(raw)
        if date_col is None:
            raise ValueError("Aucune colonne Date reconnue. Le fichier doit contenir une colonne de dates.")
        numeric_col = infer_numeric_column(raw, date_col)
        if numeric_col is None:
            raise ValueError("Aucune colonne numerique exploitable n'a ete detectee.")

        analysis, removed = prepare_analysis_data(raw, date_col, numeric_col)
        data_json = dataframe_to_store(raw)
        date_options = [{"label": str(c), "value": str(c)} for c in raw.columns]
        numeric_options = []
        for col in raw.columns:
            if col == date_col:
                continue
            converted = pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False),
                                      errors="coerce")
            if converted.notna().sum() >= 2:
                numeric_options.append({"label": str(col), "value": str(col)})

        min_d = analysis["Date"].min().date()
        max_d = analysis["Date"].max().date()
        status = (f"Fichier charge : {filename} | {len(analysis)} observations valides | "
                  f"{len(raw.columns)} colonnes | {removed} ligne(s) ignoree(s). "
                  "Vous pouvez modifier les colonnes Date et variable.")
        return (data_json, filename, date_options, date_col, numeric_options, numeric_col,
                status, {"marginTop": "9px", "color": "#238B57", "fontSize": "13px"},
                min_d, max_d, min_d, max_d)
    except Exception as exc:
        return (no_update, no_update, no_update, no_update, no_update, no_update,
                f"Erreur d'importation : {exc}",
                {"marginTop": "9px", "color": "#C0392B", "fontSize": "13px"},
                no_update, no_update, no_update, no_update)


# ============================================================
# 8. CALLBACK PRINCIPAL : KPI + GRAPHIQUES + ANOMALIES
# ============================================================
@app.callback(
    Output("kpi-container", "children"),
    Output("time-chart", "figure"),
    Output("hist-chart", "figure"),
    Output("box-chart", "figure"),
    Output("ma-chart", "figure"),
    Output("outlier-table", "data"),
    Output("outlier-summary", "children"),
    Input("data-store", "data"),
    Input("date-column", "value"),
    Input("value-column", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("method-filter", "value"),
    Input("outlier-filter", "value"),
    State("source-store", "data")
)
def update_dashboard(json_data, date_col, value_col, start_date, end_date,
                     method, display_mode, source_name):
    try:
        raw = store_to_dataframe(json_data)
        data, removed = prepare_analysis_data(raw, date_col, value_col)
        if start_date:
            data = data[data["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data["Date"] < pd.to_datetime(end_date) + pd.Timedelta(days=1)]
        if data.empty:
            raise ValueError("Aucune observation dans la periode selectionnee.")

        analysed, method_description = detect_outliers(data, method)
        anomalies = analysed[analysed["Outlier"]].copy()
        stats = calculate_kpis(data)
        unit = ""  # L'unite peut etre ajoutee dans le nom de variable ou adaptee dans le code.

        kpis = [
            kpi_card("Nombre d'observations", str(stats["count"]), accent="#2471A3"),
            kpi_card("Moyenne", f'{stats["mean"]:.3f}', unit, "#238B57"),
            kpi_card("Mediane", f'{stats["median"]:.3f}', unit, "#7551A1"),
            kpi_card("Minimum", f'{stats["minimum"]:.3f}', unit, "#2471A3"),
            kpi_card("Maximum", f'{stats["maximum"]:.3f}', unit, "#C0392B"),
            kpi_card("Ecart-type", f'{stats["std"]:.3f}', unit, "#D97706"),
            kpi_card("Valeurs atypiques", str(len(anomalies)), accent="#C0392B")
        ]

        if display_mode == "hide":
            chart_data = analysed.loc[~analysed["Outlier"], ["Date", "Value"]]
        elif display_mode == "only":
            chart_data = analysed.loc[analysed["Outlier"], ["Date", "Value"]]
        else:
            chart_data = data

        if chart_data.empty:
            figs = [empty_figure("Aucune observation pour ce filtre.") for _ in range(4)]
        else:
            figs = list(build_figures(chart_data, method, value_col, unit))
            # Dans le mode "Toutes", ajouter explicitement les points atypiques.
            if display_mode == "all" and not anomalies.empty:
                figs[0].add_trace(go.Scatter(
                    x=anomalies["Date"], y=anomalies["Value"], mode="markers",
                    name="Anomalies detectees",
                    marker=dict(color="#C0392B", size=14, symbol="x")))

        table = [{"Date": r["Date"].strftime(DATE_FORMAT),
                  "Value": round(float(r["Value"]), 4), "Method": method}
                 for _, r in anomalies.iterrows()]
        summary = (f"Source : {source_name or 'Fichier importe'} | "
                   f"{len(anomalies)} anomalie(s) detectee(s) sur {len(data)} observation(s). "
                   f"{method_description}")
        return kpis, *figs, table, summary

    except Exception as exc:
        empty = empty_figure(str(exc))
        kpis = [kpi_card("Statut", "A verifier", accent="#C0392B")]
        return kpis, empty, empty, empty, empty, [], f"Analyse impossible : {exc}"


# ============================================================
# 9. CALLBACK : OPTIONS DES VARIABLES MULTIVARIEES
# ============================================================
@app.callback(
    Output("multi-vars", "options"),
    Output("multi-vars", "value"),
    Output("scatter-x", "options"),
    Output("scatter-x", "value"),
    Output("scatter-y", "options"),
    Output("scatter-y", "value"),
    Input("data-store", "data"),
    State("date-column", "value"),
    State("value-column", "value")
)
def update_multivariate_options(json_data, date_col, value_col):
    try:
        raw = store_to_dataframe(json_data)
        numeric_cols = []
        for col in raw.columns:
            if col == date_col:
                continue
            values = pd.to_numeric(
                raw[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
            if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
                numeric_cols.append(col)

        opts = [{"label": str(c), "value": str(c)} for c in numeric_cols]
        selected = [value_col] if value_col in numeric_cols else numeric_cols[:min(4, len(numeric_cols))]
        x_value = numeric_cols[0] if numeric_cols else None
        y_value = numeric_cols[1] if len(numeric_cols) > 1 else x_value
        return opts, selected, opts, x_value, opts, y_value
    except Exception:
        return [], [], [], None, [], None


# ============================================================
# 10. CALLBACK : MATRICE DE CORRELATION ET SCATTER PLOT
# ============================================================
@app.callback(
    Output("corr-chart", "figure"),
    Output("scatter-chart", "figure"),
    Input("data-store", "data"),
    Input("date-column", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("multi-vars", "value"),
    Input("corr-method", "value"),
    Input("scatter-x", "value"),
    Input("scatter-y", "value")
)
def update_multivariate_charts(json_data, date_col, start_date, end_date,
                                selected_vars, corr_method, x_col, y_col):
    try:
        raw = store_to_dataframe(json_data)
        if date_col not in raw.columns:
            raise ValueError("Colonne Date non definie.")
        dates = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
        mask = dates.notna()
        if start_date:
            mask &= dates >= pd.to_datetime(start_date)
        if end_date:
            mask &= dates < pd.to_datetime(end_date) + pd.Timedelta(days=1)
        filtered = raw.loc[mask].copy()

        numeric = pd.DataFrame(index=filtered.index)
        for col in filtered.columns:
            if col == date_col:
                continue
            numeric[col] = pd.to_numeric(
                filtered[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
        numeric = numeric.replace([np.inf, -np.inf], np.nan)

        valid_vars = [c for c in (selected_vars or [])
                      if c in numeric.columns and numeric[c].notna().sum() >= 2
                      and numeric[c].nunique(dropna=True) > 1]
        if len(valid_vars) >= 2:
            corr = numeric[valid_vars].corr(method=corr_method, min_periods=2)
            corr_fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                colorbar=dict(title="Correlation"),
                hovertemplate="%{y} / %{x}<br>Coefficient : %{z:.3f}<extra></extra>"
            ))
            corr_fig.update_layout(
                title=f"Matrice de correlation - {corr_method.capitalize()}",
                template="plotly_white", height=470,
                xaxis_title="Variables", yaxis_title="Variables"
            )
        else:
            corr_fig = empty_figure(
                "Selectionnez au moins deux variables numeriques variables."
            )

        if (x_col in numeric.columns if x_col else False) and (
            y_col in numeric.columns if y_col else False
        ):
            pair = pd.DataFrame({
                "X": numeric[x_col],
                "Y": numeric[y_col]
            }).replace([np.inf, -np.inf], np.nan).dropna()
            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=pair["X"], y=pair["Y"], mode="markers",
                name="Observations",
                marker=dict(size=9, color="#2471A3", opacity=0.72),
                hovertemplate=f"{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<extra></extra>"
            ))
            if len(pair) >= 2 and pair["X"].nunique() > 1:
                slope, intercept = np.polyfit(pair["X"], pair["Y"], 1)
                predicted = slope * pair["X"] + intercept
                ss_res = float(((pair["Y"] - predicted) ** 2).sum())
                ss_tot = float(((pair["Y"] - pair["Y"].mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                x_line = np.linspace(pair["X"].min(), pair["X"].max(), 100)
                scatter_fig.add_trace(go.Scatter(
                    x=x_line, y=slope * x_line + intercept, mode="lines",
                    name="Regression lineaire",
                    line=dict(color="#C0392B", width=3)
                ))
                r = pair["X"].corr(pair["Y"])
                annotation = f"n = {len(pair)} | r = {r:.3f} | R² = {r2:.3f}"
                scatter_fig.add_annotation(
                    text=annotation, x=0.02, y=0.98, xref="paper", yref="paper",
                    showarrow=False, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#D5DDE7", borderwidth=1
                )
            scatter_fig.update_layout(
                title=f"Relation entre {x_col} et {y_col}",
                xaxis_title=x_col, yaxis_title=y_col,
                template="plotly_white", height=470
            )
        else:
            scatter_fig = empty_figure("Choisissez deux variables numeriques.")

        return corr_fig, scatter_fig
    except Exception as exc:
        return empty_figure(f"Analyse multivariee impossible : {exc}"), empty_figure(str(exc))


# ============================================================
# 11. LANCEMENT
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("DASHBOARD PLOTLY V05 - ANALYSE MULTIVARIEE")
    print("Adresse locale : http://127.0.0.1:8050")
    print("Importez un fichier, choisissez les colonnes et explorez les relations entre variables.")
    print("Arret du serveur : Ctrl + C dans la console.")
    print("=" * 72)
    app.run(debug=False, host="127.0.0.1", port=8050)
