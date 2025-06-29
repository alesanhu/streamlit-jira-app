"""
Streamlit – Jira Dashboard con resumen AI
=========================================
• Lee issues de Jira.
• KPIs, gráficos y exportación a Excel.
• Resumen de descripciones y primeros comentarios (OpenAI o BART-CNN).
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from io import BytesIO
from torch import accelerate 

import altair as alt
import pandas as pd
import streamlit as st
from jira import JIRA

# ─────────── Configuración de try/except opcionales ──────────────────────
try:
    from openai import OpenAI                        # SDK ≥1.0
except ImportError:
    OpenAI = None                                    # type: ignore

try:
    from transformers import pipeline                # modo offline
except ImportError:
    pipeline = None                                  # type: ignore



# ═════ Helpers ════════════════════════════════════════════════════════════
def quote_list(vals: list[str]) -> str:
    """Devuelve "'A','B','C'" escapando comillas simples internas."""
    return ",".join("'" + v.replace("'", "\\'") + "'" for v in vals)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    """Obtiene cliente Jira usando credenciales de `st.secrets`."""
    srv, usr, tok = (
        st.secrets.get("JIRA_SERVER"),
        st.secrets.get("JIRA_USER"),
        st.secrets.get("JIRA_TOKEN"),
    )
    if not (srv and usr and tok):
        st.sidebar.error("❌ Faltan *JIRA_SERVER / JIRA_USER / JIRA_TOKEN* en *Secrets*")
        return None
    try:
        return JIRA(server=srv, basic_auth=(usr, tok))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """Devuelve hasta 2000 issues + comentarios."""
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []


def summarise_openai(text: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. Resume los puntos clave y alertas de los siguientes "
        "tickets de Jira.\n\n### Tickets\n" + text[:16000]
    )
    r = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return r.choices[0].message.content.strip()


def summarise_local(text: str) -> str:
    if pipeline is None:
        return "⚠️ transformers/torch no instalados: sin resumen local."
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    chunks = textwrap.wrap(text, 3000)
    outs = [summarizer(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"] for c in chunks]
    return "\n".join(outs)


# ═════ App principal ══════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="Jira Dashboard + AI", layout="wide")
    st.title("📊 Gestión de Tickets en Jira + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ── Filtros laterales ────────────────────────────────────────────────
    st.sidebar.header("🔍 Filtros")

    # Proyectos
    try:
        projects = sorted(p.key for p in jira.projects() if not p.raw.get("archived"))
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    # Fechas
    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input(
        "Rango fechas creación", (today - pd.Timedelta(days=30), today)
    )

    # ── JQL solo proyecto + fechas ───────────────────────────────────────
    parts = []
    if sel_proj:
        parts.append(f"project in ({quote_list(sel_proj)})")
    parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se obtuvieron tickets para los filtros actuales.")
        st.stop()

    # ── DataFrame base ───────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]        = [i.key for i in issues]
    df["summary"]    = [i.fields.summary for i in issues]
    df["created"]    = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)
    df["assignee"]   = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"]   = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"]     = df["fields.status.name"].fillna("Desconocido")
    df["priority"]   = df["fields.priority.name"].fillna("None")
    df["area_destino"] = df.get("fields.customfield_10043.value", pd.NA).fillna("Sin Área")

    # ── Opciones dinámicas y filtros adicionales ────────────────────────
    statuses   = sorted(str(x) for x in df["status"].dropna().unique())
    priorities = sorted(str(x) for x in df["priority"].dropna().unique())
    assignees  = sorted(str(x) for x in df["assignee"].dropna().unique())
    areas      = sorted(str(x) for x in df["area_destino"].dropna().unique())

    sel_status = st.sidebar.multiselect("Estados", statuses, statuses)
    sel_pri    = st.sidebar.multiselect("Prioridades", priorities, priorities)
    sel_ass    = st.sidebar.multiselect("Responsable", assignees, assignees)
    sel_area   = st.sidebar.multiselect("Área Destino", areas, areas)

    df = df[
        df["status"      ].isin(sel_status) &
        df["priority"    ].isin(sel_pri)    &
        df["assignee"    ].isin(sel_ass)    &
        df["area_destino"].isin(sel_area)
    ]

    # ── Resumen AI ───────────────────────────────────────────────────────
    st.subheader("📝 Resumen y alertas")
    corpus = []
    for it in issues[:50]:  # limitar corpus
        body = it.fields.description or ""
        first_comments = " ".join(c.body for c in it.fields.comment.comments[:2])
        corpus.append(f"{it.key}: {body}\n{first_comments}")
    corpus_text = "\n\n---\n\n".join(corpus)

    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        summary = summarise_openai(corpus_text)
    else:
        summary = summarise_local(corpus_text)
    st.text_area("Resumen generado", summary, height=220)

    # ── KPIs ─────────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    open_count = df["area_destino"].notna().sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets", len(df))
    col2.metric("Abiertos", df["status"].isin(["Todo","Backlog","To Do","Sin asignar","Open","Abierta"]).sum())
    col3.metric("Media días abiertos", round(df["age_days"].mean(), 1))

    # ── Gráficos simples ────────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    s_counts = df["status"].value_counts().reset_index(names=["Estado", "Cantidad"])
    p_counts = df["priority"].value_counts().reset_index(names=["Prioridad", "Cantidad"])
    chart_s  = alt.Chart(s_counts).mark_bar().encode(x="Estado", y="Cantidad")
    chart_p  = alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad")
    st.altair_chart(chart_s | chart_p, use_container_width=True)

    # ── Exportar a Excel ────────────────────────────────────────────────
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar Excel", buffer.getvalue(), file_name="tickets.xlsx")


if __name__ == "__main__":
    main()
