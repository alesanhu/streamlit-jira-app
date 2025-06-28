"""
Streamlit – Jira Dashboard con resumen automático
------------------------------------------------
• Lee issues de Jira usando la librería `jira`.
• Muestra KPIs, gráficos y tablas.
• Genera un resumen/alertas de las descripciones + comentarios.
  - Usa OpenAI GPT si `OPENAI_API_KEY` está definido.
  - Si no hay clave, cae a un modelo local (BART) vía 🤗 Transformers.

✱ Dependencias clave (añádelas a `requirements.txt`):
  jira, pandas, streamlit, altair, openai, xlsxwriter, transformers, torch, sentencepiece
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

# ▒▒▒ SUMMARIZACIÓN  ▒▒▒ ---------------------------------------------------
try:
    from openai import OpenAI  # SDK ≥1.0
except ImportError:  # openai no instalado
    OpenAI = None  # type: ignore

try:
    # Se carga on‑demand si se necesita el modelo local
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None  # type: ignore


# ░▒░ Funciones auxiliares ░▒░ ------------------------------------------------

def quote_list(vals: list[str]) -> str:
    """Escapa valores para JQL («'ISIL','PUCP'»)."""
    escaped = [v.replace("'", "\\'") for v in vals]
    return ",".join(f"'{v}'" for v in escaped)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    server = st.secrets.get("JIRA_SERVER")
    user = st.secrets.get("JIRA_USER")
    token = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.error("Faltan credenciales Jira en Secrets.")
        return None
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(jira: JIRA, jql: str):
    """Descarga hasta 2000 issues que cumplan el JQL."""
    try:
        return jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []


def summarise_with_openai(text: str) -> str:
    """Resumen vía OpenAI GPT."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. Resume los puntos más importantes y cualquier alerta/impedimento "
        "de los siguientes tickets de Jira.\n\n### Tickets\n" + text
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def summarise_with_bart(text: str) -> str:
    """Resumen offline usando BART‑large‑cnn (Transformers)."""
    if pipeline is None:
        return "❌ transformers/torch no instalados: no se pudo usar el modelo local."
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    chunks = textwrap.wrap(text, 3000)
    outs = [summarizer(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"] for c in chunks]
    return "\n".join(outs)


# ░▒░ Interfaz Streamlit ░▒░ --------------------------------------------------

def main():
    st.set_page_config(page_title="Jira Dashboard + Resumen", layout="wide")
    st.title("📊 Gestión de Tickets en Jira + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ─── Filtros laterales ────────────────────────────────────────────────
    st.sidebar.header("🔍 Filtros")

    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    try:
        statuses = [s.name for s in jira.statuses()]
    except Exception:
        statuses = []
    sel_status = st.sidebar.multiselect("Estados", statuses, statuses)

    try:
        priorities = [p.name for p in jira.priorities()]
    except Exception as e:
        priorities = []
        st.sidebar.warning("No se pudieron cargar prioridades (¿credenciales/API Token?).")
    sel_priority = st.sidebar.multiselect("Prioridades", priorities, priorities)

    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input("Rango fechas creación", (today - pd.Timedelta(days=30), today))

    # ─── Construir JQL ────────────────────────────────────────────────────
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets de Jira…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No hay tickets para los filtros elegidos.")
        st.stop()

    # ─── DataFrame ───────────────────────────────────────────────────────
    raw = [i.raw for i in issues]
    df = pd.json_normalize(raw)
    df["key"] = [i.key for i in issues]
    df["summary"] = [i.fields.summary for i in issues]
    df["assignee"] = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"] = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"] = df["fields.status.name"]
    df["priority"] = df["fields.priority.name"].fillna("None")
    df["created"] = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # ─── Resumen (descripciones + primeros comentarios) ──────────────────
    st.subheader("📝 Resumen AI")
    combined_texts = []
    for issue in issues:
        body = [issue.fields.description or ""]
        comments = [c.body for c in issue.fields.comment.comments[:3]]  # primeros 3 comentarios
        combined_texts.append("\n".join(body + comments))
    corpus = "\n\n---\n\n".join(combined_texts)[:16000]

    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        summary = summarise_with_openai(corpus)
    else:
        summary = summarise_with_bart(corpus)
    st.text_area("Resumen generado", summary, height=200)

    # ─── KPIs ────────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    resolved_df = df.dropna(subset=["fields.resolutiondate"]).copy()
    resolved_df["resolve_days"] = (
        pd.to_datetime(resolved_df["fields.resolutiondate"], utc=True).dt.tz_localize(None) - resolved_df["created"]
    ).dt.days

    def kpi(label: str, value):
        st.metric(label, value)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(df))
    col2.metric("Abiertos", df["fields.resolutiondate"].isna().sum())
    col3.metric("Media días abiertos", round(df["age_days"].mean(), 1))
    col4.metric("Media días resolución", round(resolved_df["resolve_days"].mean(), 1) if not resolved_df.empty else "‑")

    # ─── Gráficos  ────────────────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    status_counts = df["status"].value_counts().reset_index().rename(columns={"index": "Status", "status": "Cantidad"})
    pri_counts = df["priority"].value_counts().reset_index().rename(columns={"index": "Priority", "priority": "Cantidad"})

    bar1 = alt.Chart(status_counts).mark_bar().encode(x="Status", y="Cantidad")
    bar2 = alt.Chart(pri_counts).mark_bar().encode(x="Priority", y="Cantidad")
    st.altair_chart(bar1 | bar2, use_container_width=True)

    # ─── Exportar a Excel ────────────────────────────────────────────────
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buffer.getvalue(), file_name="tickets.xlsx")


if __name__ == "__main__":
    main()

