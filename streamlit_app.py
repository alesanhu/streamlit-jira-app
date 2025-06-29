"""
Streamlit – Dashboard Jira + resumen AI
=======================================

Requisitos clave (en requirements.txt):
    jira pandas streamlit altair openai xlsxwriter
    # ↓ solo si quieres el fallback local de huggingface
    transformers torch sentencepiece
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
from jira import JIRA

# ────────── INTENTAR CARGAR LIBRERÍAS DE IA ──────────────────────────────
try:
    from openai import OpenAI          # openai ≥1.0
except ImportError:
    OpenAI = None                      # type: ignore[assignment]

try:
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None                    # type: ignore

# ────────── HELPERS ───────────────────────────────────────────────────────
def quote_list(vals: list[str]) -> str:
    """Devuelve 'A','B','C' escapando comillas simples."""
    escaped = [v.replace("'", r"\'") for v in vals]
    return ",".join(f"'{v}'" for v in escaped)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    """Conecta a Jira usando st.secrets."""
    server = st.secrets.get("JIRA_SERVER")
    user   = st.secrets.get("JIRA_USER")
    token  = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.error("Faltan credenciales Jira en *Secrets*.")
        return None
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.sidebar.error(f"Error de conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """Descarga issues (máx 2000) con comentarios incluidos."""
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error obteniendo tickets: {e}")
        return []


def summarise_openai(text: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres analista de soporte. Resume los puntos clave y alertas de los siguientes tickets:\n\n"
        "### Tickets\n" + text
    )
    rsp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return rsp.choices[0].message.content.strip()


def summarise_bart(text: str) -> str:
    if pipeline is None:
        return "⚠️ transformers/torch no instalados."

    summarizer = pipeline(
        "summarization", model="facebook/bart-large-cnn", device_map="auto"
    )
    chunks = textwrap.wrap(text, 3000)
    outs = [
        summarizer(chunk, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
        for chunk in chunks
    ]
    return "\n".join(outs)


# ────────── APP ───────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Jira Dashboard + AI", layout="wide")
    st.title("📊 Dashboard Jira + Resumen IA")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ────────── FILTROS LATERALES ───────────────────────────────────────
    st.sidebar.header("🔍 Filtros")

    # Proyectos
    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    # Fechas
    today = datetime.utcnow().date()
    start_date, end_date = st.sidebar.date_input(
        "Rango fechas creación", (today - pd.Timedelta(days=30), today)
    )

    # Construir JQL (solo proyecto + fechas)
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start_date}' AND created <= '{end_date}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se obtuvieron tickets para los filtros actuales.")
        st.stop()

    # ────────── DATAFRAME ────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]        = [i.key for i in issues]
    df["summary"]    = [i.fields.summary for i in issues]
    df["status"]     = df["fields.status.name"]
    df["priority"]   = df["fields.priority.name"].fillna("None")
    df["assignee"]   = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"]   = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["created"]    = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)
    df["area_destino"] = df.get("fields.customfield_10043.value", "Sin Área")

    # ────────── FILTROS DEPENDIENTES ─────────────────────────────────────
    statuses   = sorted(df["status"].unique())
    priorities = sorted(df["priority"].unique())
    assignees  = sorted(df["assignee"].unique())
    areas      = sorted(df["area_destino"].unique())

    sel_status = st.sidebar.multiselect("Estados", statuses, statuses)
    sel_pri    = st.sidebar.multiselect("Prioridades", priorities, priorities)
    sel_ass    = st.sidebar.multiselect("Responsable", assignees, assignees)
    sel_area   = st.sidebar.multiselect("Área destino", areas, areas)

    df = df[
        df["status"].isin(sel_status)
        & df["priority"].isin(sel_pri)
        & df["assignee"].isin(sel_ass)
        & df["area_destino"].isin(sel_area)
    ]

    # ────────── RESUMEN IA ───────────────────────────────────────────────
    st.subheader("📝 Resumen automático")
    joined_text = []
    for issue in issues:
        if issue.key not in df["key"].values:
            continue  # filtrado
        desc = issue.fields.description or ""
        comments = [c.body for c in issue.fields.comment.comments[:3]]
        joined_text.append(desc + "\n".join(comments))
    corpus = "\n\n---\n\n".join(joined_text)[:16000]

    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        summary = summarise_openai(corpus)
    else:
        summary = summarise_bart(corpus)
    st.text_area("Resumen / alertas", summary, height=220)

    # ────────── KPIs ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets", len(df))
    abiertos = df["fields.resolutiondate"].isna().sum()
    col2.metric("Abiertos", abiertos)
    col3.metric("Resueltos", len(df) - abiertos)

    # ────────── GRÁFICOS  ────────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    st.altair_chart(
        alt.hconcat(
            alt.Chart(df).mark_bar().encode(x="status:N", y="count():Q").properties(width=350),
            alt.Chart(df).mark_bar().encode(x="priority:N", y="count():Q").properties(width=350),
            spacing=60,
        ),
        use_container_width=True,
    )

    # ────────── EXPORTAR EXCEL ───────────────────────────────────────────
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wrt:
        df.to_excel(wrt, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buf.getvalue(), file_name="tickets.xlsx")


if __name__ == "__main__":
    main()
