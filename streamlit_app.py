"""
Streamlit – Jira Dashboard + Resumen AI (GPT o BART)
Requisitos extra: jira, pandas, streamlit, altair, openai,
                 transformers, torch, sentencepiece, xlsxwriter
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

# ─── Intentamos cargar clientes de IA ─────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
try:
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None  # type: ignore


# ─── Utilidades ──────────────────────────────────────────────────────────
def quote_list(vals: list[str]) -> str:
    """'ISIL','PUCP'…  (sin escapar complejo)."""
    return ",".join(f"'{v}'" for v in vals)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    srv, usr, tok = (
        st.secrets.get("JIRA_SERVER"),
        st.secrets.get("JIRA_USER"),
        st.secrets.get("JIRA_TOKEN"),
    )
    if not (srv and usr and tok):
        st.sidebar.error("Faltan JIRA_SERVER/JIRA_USER/JIRA_TOKEN en Secrets.")
        return None
    try:
        return JIRA(server=srv, basic_auth=(usr, tok))
    except Exception as e:
        st.sidebar.error(f"Conexión Jira falló: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error al consultar Jira: {e}")
        return []


def summarise(text: str) -> str:
    """GPT → BART → mensaje."""
    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = (
            "Eres un analista. Resume los puntos clave y alertas "
            "de los tickets de Jira siguientes:\n\n" + text[:15000]
        )
        r = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        return r.choices[0].message.content.strip()

    if pipeline is not None:
        model = pipeline(
            "summarization", model="facebook/bart-large-cnn", device_map="auto"
        )
        chunks = textwrap.wrap(text, 3000)
        outs = model(chunks, max_length=180, min_length=40, do_sample=False)
        return "\n".join(o["summary_text"] for o in outs)

    return "⚠️ No hay OPENAI_API_KEY ni librería transformers instalada."


# ─── APP ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Jira Dashboard + AI Summary", layout="wide")
    st.title("📊 Jira Dashboard    🤖 Resumen AI bajo demanda")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ── Filtros de barra lateral ────────────────────────────────────────
    st.sidebar.header("Filtros")
    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    today = datetime.utcnow().date()
    date_val = st.sidebar.date_input("Rango creación", (today - pd.Timedelta(30), today))
    start, end = date_val if isinstance(date_val, tuple) else (date_val, date_val)

    # JQL solo fecha + proyecto (status/prioridad después en local)
    parts = []
    if sel_proj:
        parts.append(f"project in ({quote_list(sel_proj)})")
    parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)
    if not issues:
        st.warning("Sin tickets para estos filtros.")
        st.stop()

    # DataFrame base
    df = pd.json_normalize([i.raw for i in issues])
    df["key"] = [i.key for i in issues]
    df["summary"] = [i.fields.summary for i in issues]
    df["assignee"] = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["priority"] = df["fields.priority.name"].fillna("None")
    df["status"] = df["fields.status.name"]
    df["created"] = pd.to_datetime(df["fields.created"]).dt.tz_localize(None)
    df["area_destino"] = df.get("fields.customfield_10043.value", pd.NA).astype(str)

    # Opciones dinámicas
    statuses = sorted(df["status"].dropna().unique())
    priorities = sorted(df["priority"].dropna().unique())
    assignees = sorted(df["assignee"].dropna().unique())
    areas = sorted(df["area_destino"].dropna().unique())

    sel_status = st.sidebar.multiselect("Estados", statuses, statuses)
    sel_pri = st.sidebar.multiselect("Prioridades", priorities, priorities)
    sel_ass = st.sidebar.multiselect("Responsable", assignees, assignees)
    sel_area = st.sidebar.multiselect("Área Destino", areas, areas)

    df = df[
        df["status"].isin(sel_status)
        & df["priority"].isin(sel_pri)
        & df["assignee"].isin(sel_ass)
        & df["area_destino"].isin(sel_area)
    ]

    # KPI rápidos
    now = pd.Timestamp.utcnow()
    df["age_days"] = (now - df["created"]).dt.days
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets", len(df))
    col2.metric("Prom. días abiertos", round(df["age_days"].mean(), 1))
    col3.metric("Máx. días abiertos", int(df["age_days"].max()))

    # Gráficos sencillos
    st.subheader("Distribución por Estado y Prioridad")
    s_counts = df["status"].value_counts().reset_index().rename(columns={"index": "Estado", "status": "Cantidad"})
    p_counts = df["priority"].value_counts().reset_index().rename(columns={"index": "Prioridad", "priority": "Cantidad"})
    st.altair_chart(
        (alt.Chart(s_counts).mark_bar().encode(x="Estado", y="Cantidad"))
        | (alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad")),
        use_container_width=True,
    )

    # Botón para generar resumen
    st.subheader("Resumen / Alertas")
    if st.button("🪄 Generar resumen de descripciones + comentarios"):
        texts = []
        for i in issues:
            body = i.fields.description or ""
            comments = "\n".join(c.body for c in i.fields.comment.comments[:3])
            texts.append(f"{i.key} – {i.fields.summary}\n{body}\n{comments}")
        combined = "\n\n---\n\n".join(texts)
        with st.spinner("Generando resumen…"):
            summary = summarise(combined)
        st.text_area("Resultado", summary, height=260)
    else:
        st.info("Pulsa el botón para generar el resumen IA.")

    # Exportar Excel
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as wrt:
        df.to_excel(wrt, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buffer.getvalue(), file_name="tickets.xlsx")


if __name__ == "__main__":
    main()
