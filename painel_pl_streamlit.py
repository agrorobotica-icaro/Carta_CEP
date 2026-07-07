from __future__ import annotations

import os
import re
from math import pi, sqrt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import shapiro, probplot


# ============================================================
# (Opcional) caminho fixo. Se deixar vazio, use upload no app.
EXCEL_PATH = r""  # ex.: r"C:\caminho\arquivo.xlsx"
EXCLUDE_SHEETS: List[str] = []
# ============================================================

MIN_COL_CANDIDATES = ["Mínimo", "Minimo", "MINIMO", "minimo", "mínimo", "mín"]
MEAN_COL_CANDIDATES = ["Média", "Media", "MEDIA", "media", "média", "méd"]
MAX_COL_CANDIDATES = ["Máximo", "Maximo", "MAXIMO", "maximo", "máximo", "máx"]
PL_COL_CANDIDATES = ["PL", "Pl", "pl"]
UNIDADE_COL_CANDIDATES = ["Unidade", "UNIDADE", "unidade"]
TIPO_COL_CANDIDATES = ["Tipo", "TIPO", "tipo"]

RE_SAIDA = re.compile(r"^\s*saida\s*[_\-\s]*([0-9]+)\s*$", re.IGNORECASE)
RE_VALOR = re.compile(r"^\s*valor\s*[_\-\s]*([0-9]+)\s*$", re.IGNORECASE)
RE_DATA = re.compile(r"^\s*data\s*[_\-\s]*([0-9]+)\s*$", re.IGNORECASE)


def pick_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def find_value_date_pairs(
    df: pd.DataFrame, value_regex: re.Pattern
) -> List[Tuple[str, str, int]]:
    value_map: Dict[int, str] = {}
    data_map: Dict[int, str] = {}

    for col in df.columns:
        s = str(col).strip()

        mv = value_regex.match(s)
        if mv:
            value_map[int(mv.group(1))] = col
            continue

        md = RE_DATA.match(s)
        if md:
            data_map[int(md.group(1))] = col

    pairs: List[Tuple[str, str, int]] = []
    for k in sorted(set(value_map) & set(data_map)):
        pairs.append((value_map[k], data_map[k], k))
    return pairs


def detect_history_kind(df: pd.DataFrame) -> Optional[str]:
    if find_value_date_pairs(df, RE_SAIDA):
        return "saida"
    if find_value_date_pairs(df, RE_VALOR):
        return "valor"
    return None


def to_long_history(df: pd.DataFrame, pl_col: str, kind: str) -> pd.DataFrame:
    value_regex = RE_SAIDA if kind == "saida" else RE_VALOR
    pairs = find_value_date_pairs(df, value_regex)

    if not pairs:
        return pd.DataFrame(columns=[pl_col, "idx", "data", "valor"])

    frames = []
    for v_col, d_col, idx in pairs:
        temp = df[[pl_col, v_col, d_col]].copy()
        temp.rename(columns={v_col: "valor", d_col: "data"}, inplace=True)
        temp["idx"] = idx
        temp["data"] = pd.to_datetime(
            temp["data"], errors="coerce", dayfirst=True)
        temp["valor"] = pd.to_numeric(temp["valor"], errors="coerce")
        frames.append(temp)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["data", "valor"])
    long_df = long_df.sort_values(
        [pl_col, "data", "idx"]).reset_index(drop=True)
    return long_df


def load_workbook(excel_obj, exclude_sheets: List[str]) -> Dict[str, dict]:
    all_sheets = pd.read_excel(excel_obj, sheet_name=None, engine="openpyxl")

    out: Dict[str, dict] = {}
    for sheet_name, df in all_sheets.items():
        if sheet_name in exclude_sheets:
            continue
        if df is None or df.empty:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        pl_col = pick_existing_col(df, PL_COL_CANDIDATES) or df.columns[0]
        min_col = pick_existing_col(df, MIN_COL_CANDIDATES)
        mean_col = pick_existing_col(df, MEAN_COL_CANDIDATES)
        max_col = pick_existing_col(df, MAX_COL_CANDIDATES)
        unidade_col = pick_existing_col(df, UNIDADE_COL_CANDIDATES)
        tipo_col = pick_existing_col(df, TIPO_COL_CANDIDATES)

        kind = detect_history_kind(df)
        hist = to_long_history(df, pl_col, kind) if kind else pd.DataFrame(
            columns=[pl_col, "idx", "data", "valor"]
        )

        out[sheet_name] = {
            "df": df,
            "pl_col": pl_col,
            "min_col": min_col,
            "mean_col": mean_col,
            "max_col": max_col,
            "unidade_col": unidade_col,
            "tipo_col": tipo_col,
            "hist": hist,
            "kind": kind,
        }

    return out


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    return (1.0 / (sigma * sqrt(2.0 * pi))) * pow(2.718281828459045, -0.5 * ((x - mu) / sigma) ** 2)


def make_normality_table(sheet: dict, selected_pl) -> pd.DataFrame:
    df = sheet["df"].copy()
    pl_col = sheet["pl_col"]

    dff = df[df[pl_col] == selected_pl].copy()

    excluded = {str(pl_col).strip().lower(), "data"}
    candidate_cols = []

    for col in dff.columns:
        if str(col).strip().lower() in excluded:
            continue
        serie = pd.to_numeric(dff[col], errors="coerce")
        if serie.notna().sum() > 0:
            candidate_cols.append(col)

    rows = []
    for col in candidate_cols:
        serie = pd.to_numeric(dff[col], errors="coerce").dropna()

        if len(serie) == 0:
            continue

        n = int(len(serie))
        media = float(serie.mean())
        dp = float(serie.std(ddof=1)) if n > 1 else float("nan")
        minimo = float(serie.min())
        q1 = float(serie.quantile(0.25))
        mediana = float(serie.median())
        q3 = float(serie.quantile(0.75))
        maximo = float(serie.max())

        p_valor = float("nan")
        interpretacao = "Amostra insuficiente"

        if 3 <= n <= 5000:
            try:
                _, p_valor = shapiro(serie)
                interpretacao = "Normal" if p_valor >= 0.05 else "Não normal"
            except Exception:
                p_valor = float("nan")
                interpretacao = "Erro no teste"
        elif n < 3:
            interpretacao = "n < 3"
        else:
            interpretacao = "n > 5000"

        rows.append(
            {
                "Parâmetro": col,
                "n": n,
                "Média": media,
                "Desvio-padrão": dp,
                "Mínimo": minimo,
                "Q1": q1,
                "Mediana": mediana,
                "Q3": q3,
                "Máximo": maximo,
                "p-valor (Shapiro-Wilk)": p_valor,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        num_cols = [
            "Média",
            "Desvio-padrão",
            "Mínimo",
            "Q1",
            "Mediana",
            "Q3",
            "Máximo",
            "p-valor (Shapiro-Wilk)",
        ]
        result[num_cols] = result[num_cols].round(4)

    return result


def make_distribution_figure(
    sheet_name: str,
    sheet: dict,
    selected_pl,
    selected_param: str,
) -> go.Figure:

    df = sheet["df"]
    pl_col = sheet["pl_col"]

    dff = df[df[pl_col] == selected_pl].copy()
    dff[selected_param] = pd.to_numeric(dff[selected_param], errors="coerce")
    dff = dff.dropna(subset=[selected_param])

    fig = go.Figure()

    if dff.empty:
        return fig

    valores = dff[selected_param].astype(float)

    mu = float(valores.mean())
    sigma = float(valores.std(ddof=1))

    xmin = valores.min()
    xmax = valores.max()

    if xmin == xmax:
        xmin -= 1
        xmax += 1

    # ============================
    # HISTOGRAMA (DADOS REAIS)
    # ============================

    fig.add_trace(
        go.Histogram(
            x=valores,
            nbinsx=25,
            histnorm="probability density",
            name="Distribuição observada",
            marker=dict(
                color="rgba(21,101,192,0.45)",
                line=dict(color="rgba(21,101,192,0.8)", width=1),
            ),
            showlegend=False,
        )
    )

    # ============================
    # CURVA NORMAL TEÓRICA
    # ============================

    xmin_curve = mu - 3 * sigma
    xmax_curve = mu + 3 * sigma

    npts = 400
    xs = [xmin_curve + (xmax_curve - xmin_curve) * i / (npts - 1)
          for i in range(npts)]
    ys = [normal_pdf(x, mu, sigma) for x in xs]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=3, color="#C62828"),
            name="Curva Normal",
            showlegend=False,
        )
    )

    # ============================
    # SOMBREAMENTO ±2σ
    # ============================

    shade_min = mu - 2 * sigma
    shade_max = mu + 2 * sigma

    xs_fill = [x for x in xs if shade_min <= x <= shade_max]
    ys_fill = [normal_pdf(x, mu, sigma) for x in xs_fill]

    if xs_fill:
        fig.add_trace(
            go.Scatter(
                x=xs_fill + xs_fill[::-1],
                y=ys_fill + [0] * len(ys_fill),
                fill="toself",
                fillcolor="rgba(31,111,95,0.22)",
                line=dict(color="rgba(31,111,95,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # ============================
    # LINHAS MÉDIA E ±2σ
    # ============================

    y_mu = normal_pdf(mu, mu, sigma)
    y_min = normal_pdf(shade_min, mu, sigma)
    y_max = normal_pdf(shade_max, mu, sigma)

    fig.add_trace(
        go.Scatter(
            x=[shade_min, shade_min],
            y=[0, y_min],
            mode="lines",
            line=dict(color="#2E7D32", width=2, dash="dot"),
            name="±2 Desvios-padrão",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[shade_max, shade_max],
            y=[0, y_max],
            mode="lines",
            line=dict(color="#2E7D32", width=2, dash="dot"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[mu, mu],
            y=[0, y_mu],
            mode="lines",
            line=dict(color="#C62828", width=2, dash="dot"),
            name="Média",
            showlegend=True,
        )
    )

    # ============================
    # ANOTAÇÃO
    # ============================

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.01,
        xanchor="right",
        yanchor="bottom",
        text=f"PL: {selected_pl}<br>Média: {mu:.3f}<br>DP: {sigma:.3f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.75)",
    )

    fig.update_layout(
        title=dict(
            text=f"Distribuição | {selected_param} | PL: {selected_pl}",
            font=dict(size=22, color="black"),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=dict(
            text=selected_param,
            font=dict(size=16, color="black")
        ),
        yaxis_title=dict(
            text="Densidade",
            font=dict(size=16, color="black")
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=80, t=80, b=60),
        font=dict(color="black"),
        legend=dict(
            x=0.96,
            y=0.97,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=12, color="black"),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickcolor="black",
        tickfont=dict(size=13, color="black"),
        title_font=dict(size=16, color="black"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickcolor="black",
        tickfont=dict(size=13, color="black"),
        title_font=dict(size=16, color="black"),
    )

    return fig


def make_qq_plot_figure(
    sheet_name: str,
    sheet: dict,
    selected_pl,
    selected_param: str,
) -> go.Figure:
    df = sheet["df"]
    pl_col = sheet["pl_col"]

    dff = df[df[pl_col] == selected_pl].copy()
    dff[selected_param] = pd.to_numeric(dff[selected_param], errors="coerce")
    dff = dff.dropna(subset=[selected_param])

    fig = go.Figure()

    if dff.empty or len(dff) < 3:
        fig.update_layout(
            title=dict(
                text=f"Q-Q Plot | {selected_param} | PL: {selected_pl}",
                font=dict(size=22, color="black"),
                x=0.5,
                xanchor="center",
            ),
            xaxis_title=dict(text="Quantis teóricos",
                             font=dict(size=16, color="black")),
            yaxis_title=dict(text="Quantis amostrais",
                             font=dict(size=16, color="black")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=40, t=80, b=60),
            font=dict(color="black"),
        )
        return fig

    valores = dff[selected_param].astype(float).dropna().values

    (osm, osr), (slope, intercept, r) = probplot(valores, dist="norm")

    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Observações",
            marker=dict(size=8, color="#1565C0"),
            showlegend=False,
        )
    )

    x_line = [float(min(osm)), float(max(osm))]
    y_line = [slope * x + intercept for x in x_line]

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Linha de referência",
            line=dict(color="#C62828", width=2, dash="dot"),
            showlegend=False,
        )
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.01,
        xanchor="right",
        yanchor="bottom",
        text=f"PL: {selected_pl}<br>n: {len(valores)}<br>R² aprox.: {r**2:.4f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        bgcolor="rgba(255,255,255,0.75)",
    )

    fig.update_layout(
        title=dict(
            text=f"Q-Q Plot | {selected_param} | PL: {selected_pl}",
            font=dict(size=22, color="black"),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=dict(text="Quantis teóricos",
                         font=dict(size=16, color="black")),
        yaxis_title=dict(text="Quantis amostrais",
                         font=dict(size=16, color="black")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=60),
        font=dict(color="black"),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickcolor="black",
        tickfont=dict(size=13, color="black"),
        title_font=dict(size=16, color="black"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickcolor="black",
        tickfont=dict(size=13, color="black"),
        title_font=dict(size=16, color="black"),
    )

    return fig


def make_figure(
    sheet_name: str,
    sheet: dict,
    selected_pl,
    use_rounding: bool = True,
    smoothing: float = 0.6,
) -> go.Figure:
    df = sheet["df"]
    pl_col = sheet["pl_col"]
    hist = sheet["hist"]
    kind = sheet.get("kind")
    unidade_col = sheet.get("unidade_col")
    tipo_col = sheet.get("tipo_col")

    fig = go.Figure()

    base_row = df[df[pl_col] == selected_pl]
    row0 = base_row.iloc[0] if not base_row.empty else None

    unidade_label = "Valor"
    if row0 is not None and unidade_col and unidade_col in df.columns:
        unidade_val = row0.get(unidade_col)
        if pd.notna(unidade_val) and str(unidade_val).strip():
            unidade_label = str(unidade_val).strip()

    tipo_text = None
    if row0 is not None and tipo_col and tipo_col in df.columns:
        tipo_val = row0.get(tipo_col)
        if pd.notna(tipo_val) and str(tipo_val).strip():
            tipo_text = str(tipo_val).strip()

    if hist.empty or not kind:
        fig.update_layout(
            title=dict(
                text=f"{sheet_name} — sem histórico (nenhum par Saida_n/Data_n ou Valor_n/Data_n)",
                font=dict(size=20, color="black"),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    dff = hist[hist[pl_col] == selected_pl].copy()
    dff = dff.dropna(subset=["data", "valor"]).sort_values("data")

    if kind == "valor":
        series_name = "Valor"
    else:
        series_name = "Saída"

    plot_mode = "lines+markers"

    if use_rounding:
        line_cfg = dict(shape="spline", smoothing=float(smoothing), width=2)
    else:
        line_cfg = dict(shape="linear", width=2)

    fig.add_trace(
        go.Scatter(
            x=dff["data"],
            y=dff["valor"],
            mode=plot_mode,
            name=series_name,
            line=line_cfg,
            marker=dict(size=7),
        )
    )

    if str(sheet_name).strip().lower() == "massa":
        fig.add_hline(
            y=20,
            line_dash="dot",
            line_color="red",
            annotation_text="Pouco (Não usar)",
            annotation_font_color="red",
            annotation_position="top left",
        )
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="red",
            annotation_text="Médio (Usar pouco)",
            annotation_font_color="red",
            annotation_position="top left",
        )
    else:
        if row0 is not None:
            def add_hline(colname: Optional[str], label: str, color: str):
                if colname and colname in df.columns:
                    val = pd.to_numeric(row0.get(colname), errors="coerce")
                    if pd.notna(val):
                        fig.add_hline(
                            y=float(val),
                            line_dash="dot",
                            line_color=color,
                            annotation_text=label,
                            annotation_font_color=color,
                            annotation_position="top left",
                        )

            add_hline(sheet["min_col"], "Mínimo", "red")
            add_hline(sheet["mean_col"], "Média", "black")
            add_hline(sheet["max_col"], "Máximo", "red")

    if tipo_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.995,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            text=f"Tipo: {tipo_text}",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="right",
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig.update_layout(
        title=dict(
            text=f"{sheet_name} — Histórico do PL: {selected_pl}",
            font=dict(size=22, color="black"),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=dict(text="Data", font=dict(size=16, color="black")),
        yaxis_title=dict(text=unidade_label,
                         font=dict(size=16, color="black")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=60),
        font=dict(color="black"),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickcolor="black",
        tickfont=dict(size=13, color="black"),
        title_font=dict(size=16, color="black"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickcolor="black",
        tickfont=dict(size=13, color="black"),
        title_font=dict(size=16, color="black"),
    )

    return fig


def main():
    st.set_page_config(page_title="Painel PL", layout="wide")

    st.markdown(
        """
<style>
.stApp { background: #F6FBF7; }
section[data-testid="stSidebar"] { background: #419D68; }
section[data-testid="stSidebar"] * { color: #E8F5E9 !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image("agrorobotica-1024x212_vx.png", use_container_width=True)
        st.markdown("## Painel PL")
        st.caption("Selecione a aba (parâmetro) e o PL.")
        st.divider()

        up = st.file_uploader("Escolha o .xlsx", type=["xlsx"])
        st.caption("Ou use EXCEL_PATH no código.")

        st.divider()
        st.markdown("### Curva")
        use_rounding = st.toggle("Arredondar arestas", value=True)
        smoothing = st.slider(
            "Suavização (só arredonda)",
            0.0,
            1.3,
            0.6,
            0.05,
            help="0 = reta (quinas). 0.4–0.7 = só tirar pontas. >1.0 pode ondular.",
        )

    excel_obj = None
    if up is not None:
        excel_obj = up
    elif EXCEL_PATH and os.path.exists(EXCEL_PATH):
        excel_obj = EXCEL_PATH

    if excel_obj is None:
        st.warning("Envie um .xlsx na sidebar ou preencha EXCEL_PATH no código.")
        st.stop()

    sheets = load_workbook(excel_obj, EXCLUDE_SHEETS)
    if not sheets:
        st.error("Nenhuma aba válida encontrada no Excel.")
        st.stop()

    sheet_names = list(sheets.keys())

    with st.sidebar:
        active = st.radio("Aba (parâmetro)", sheet_names, index=0)

    sheet = sheets[active]
    pl_col = sheet["pl_col"]

    pls = sheet["df"][pl_col].dropna().unique().tolist()
    pls = sorted(pls, key=lambda x: str(x))
    if not pls:
        st.error("Não há PLs nessa aba.")
        st.stop()

    if str(active).strip().lower() == "dist_normal_internos":
        excluded = {pl_col.lower(), "data"}
        numeric_params = []

        for col in sheet["df"].columns:
            if str(col).strip().lower() in excluded:
                continue
            serie = pd.to_numeric(sheet["df"][col], errors="coerce")
            if serie.notna().sum() > 0:
                numeric_params.append(col)

        if not numeric_params:
            st.error(
                "Nenhum parâmetro químico numérico encontrado na aba Dist_Normal_Internos.")
            st.stop()

        selected_pl = st.selectbox("PL", pls, index=0)
        selected_param = st.selectbox(
            "Parâmetro químico", numeric_params, index=0)

        fig = make_distribution_figure(
            active,
            sheet,
            selected_pl,
            selected_param,
        )
        st.plotly_chart(fig, use_container_width=True)

        qq_fig = make_qq_plot_figure(
            active,
            sheet,
            selected_pl,
            selected_param,
        )
        st.plotly_chart(qq_fig, use_container_width=True)

        st.markdown("### Estatísticas descritivas + teste de normalidade")
        normality_table = make_normality_table(sheet, selected_pl)

        if normality_table.empty:
            st.info("Não há dados suficientes para montar a tabela.")
        else:
            st.dataframe(normality_table, use_container_width=True)

        st.stop()

    selected_pl = st.selectbox("PL", pls, index=0)

    st.caption(
        f"Coluna de PL: **{pl_col}** | Linhas pontilhadas: "
        f"**{sheet['min_col'] or '-'} / {sheet['mean_col'] or '-'} / {sheet['max_col'] or '-'}**"
    )

    fig = make_figure(
        active,
        sheet,
        selected_pl,
        use_rounding=use_rounding,
        smoothing=smoothing,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
