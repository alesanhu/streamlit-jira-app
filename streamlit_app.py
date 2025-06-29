"""
Streamlit – Jira Dashboard con resumen automático (sin Torch)
-------------------------------------------------------------
• Lee issues de Jira.
• KPIs, gráficos y exportación Excel.
• Resumen/alertas:
    – Usa OpenAI si hay OPENAI_API_KEY.
    – Si no, usa TextRank de sumy (100 % Python).
"""

from __future__ import annotations
import os, textwrap
from datetime import datetime
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
from jira import JIRA
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

# ───────────────────────── UTILIDADES ──────────────────────────
def quote_list(vals: list[str]) -> str:
    """Escapa valores para JQL ('PR1','PR2')."""
    return ", ".join("'" + v.replace("'", "\\'") + "'" for v in vals)

@st.cache_resource(show_spinner=False)
def jira_client() -> JIRA | None:
    srv, usr, tok = (st.secrets.get(k) for k in ("JIRA_SERVER", "JIRA_USER", "JIRA_TOKEN"))
    if not all((srv, usr, tok)):
        st.sidebar.error("⚠️ Agrega JIRA_SERVER, JIRA_USER y JIRA_TOKEN en *Secrets*.")
        return None
    try:
        return JIRA(server=srv, basic_auth=(usr, tok))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error Jira: {e}")
        return []

def sumy_textrank(text: str, sentences: int = 5) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("spanish"))
    summarizer = TextRankSummarizer()
    return " ".join(str(s) for s in summarizer(parser.document, sentences))

def summary_from_tickets(issues) -> str:
    # Unir descripción + primeros 3 comentarios de cada issue (máx ~8 k carac.)
    parts = []
    for issue in issues:
        body = issue.fields.description or ""
        comments = [c.body for c in issue.fields.comment.comments[:3]]
        parts.append(body + "\n".join(comments))
    corpus = "\n\n---\n\n".join(parts)[:8000] or "Sin texto."

    if "OPENAI_API_KEY" in st.secrets:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = ("Resume puntos clave y alertas de estos tickets de Jira:\n\n" + corpus)
        rsp = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.4,
        )
        return rsp.choices[0].message.content.strip()
    # Fallback a TextRank
    return sumy_textrank(corpus)

# ───────────────────────── INTERFAZ ──────────────────────────
def main():
    st.set_page_config(page_title="Jira Dashboard + Resumen", layout="wide")
    st.title("📊 Jira Dashboard")

    jira = jira_client()
    if jira is None:
        st.stop()

    st.sidebar.header("Filtros")

    # Proyectos
    try:
        projects = sorted(p.key for p in jira.projects() if not p.raw.get("archived"))
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    # Fechas
    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input(
        "Rango creación",
        value=(today - pd.Timedelta(days=30), today),
    )

    # JQL (solo proyecto + fechas; filtros finos se aplican local)
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)
    if not issues:
        st.warning("No se obtuvieron tickets.")
        st.stop()

    # DataFrame base
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]       = [i.key for i in issues]
    df["status"]    = df["fields.status.name"]
    df["priority"]  = df["fields.priority.name"].fillna("None")
    df["assignee"]  = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["area_destino"] = df.get("fields.customfield_10043.value", "Sin Área")
    df["created"]   = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # Opciones dinámicas
    statuses   = sorted(df["status"].dropna().unique())
    priorities = sorted(df["priority"].dropna().unique())
    assignees  = sorted(df["assignee"].dropna().unique())
    areas      = sorted(df["area_destino"].dropna().unique())

    sel_status = st.sidebar.multiselect("Estados", statuses, statuses)
    sel_pri    = st.sidebar.multiselect("Prioridades", priorities, priorities)
    sel_ass    = st.sidebar.multiselect("Responsable", assignees, assignees)
    sel_area   = st.sidebar.multiselect("Área destino", areas, areas)

    # Filtro local
    df = df[
        df["status"].isin(sel_status)
        & df["priority"].isin(sel_pri)
        & df["assignee"].isin(sel_ass)
        & df["area_destino"].isin(sel_area)
    ]

    # KPIs
    col1, col2 = st.columns(2)
    col1.metric("Tickets", len(df))
    col2.metric("Responsables únicos", df["assignee"].nunique())

    # Gráficos
    st.subheader("Distribución")
    s_counts = df["status"].value_counts().reset_index()
    s_counts.columns = ["Estado", "Cantidad"]
    p_counts = df["priority"].value_counts().reset_index()
    p_counts.columns = ["Prioridad", "Cantidad"]
    st.altair_chart(
        alt.Chart(s_counts).mark_bar().encode(x="Estado", y="Cantidad") |
        alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad"),
        use_container_width=True
    )

    # Botón para generar resumen
    if st.button("Generar resumen / alertas"):
        with st.spinner("Resumiendo…"):
            resumen = summary_from_tickets(issues)
        st.text_area("Resumen", resumen, height=260)

    # Exportar
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Tickets")
    st.download_button("Exportar a Excel", buf.getvalue(), file_name="tickets.xlsx")

# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
