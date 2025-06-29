"""
Streamlit – Jira Dashboard con resumen automático
-------------------------------------------------
• Lee issues de Jira (librería `jira`).
• KPIs, gráficos, filtros dinámicos y exportación Excel.
• Resumen/alertas (OpenAI GPT si hay clave; BART local si no).

Requisitos (requirements.txt):
  pandas streamlit altair jira openai xlsxwriter
  # Opcional para modelo local ↓
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

# ─── Resumen IA ───────────────────────────────────────────────────────────
try:
    from openai import OpenAI           # SDK ≥1.0
except ImportError:
    OpenAI = None                       # type: ignore

try:
    from transformers import pipeline   # type: ignore
except ImportError:
    pipeline = None                     # type: ignore


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def quote_list(vals: list[str]) -> str:
    """'ISIL','PUCP' … (escapa comillas simples)."""
    return ",".join("'" + v.replace("'", r"\'") + "'" for v in vals)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    server = st.secrets.get("JIRA_SERVER")
    user   = st.secrets.get("JIRA_USER")
    token  = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.error("❌ Faltan credenciales Jira en *Secrets*.")
        return None
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """Hasta 2 000 issues con comentarios incluidos."""
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []


def summarise_openai(text: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. Resume puntos clave y alertas de estos tickets Jira:\n\n" + text
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
        return "⚠️ No hay transformers/torch instalados."
    summ = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    chunks = textwrap.wrap(text, 3000)
    return "\n".join(
        summ(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
        for c in chunks
    )


# ══════════════════════════════════════════════════════════════════════════
#  Main app
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="Jira Dashboard + AI Summary", layout="wide")
    st.title("📊 Dashboard de Tickets Jira + Resumen IA")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ─── Filtros laterales ───────────────────────────────────────────────
    st.sidebar.header("🔍 Filtros")

    # proyectos
    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    # fechas
    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input("Rango creación", (today - pd.Timedelta(days=30), today))

    # — JQL solamente proyectos + fechas
    jql_parts: list[str] = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    # conseguir issues
    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se obtuvieron tickets para los filtros actuales.")
        st.stop()

    # ─── DataFrame base ─────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]       = [i.key for i in issues]
    df["summary"]   = [i.fields.summary for i in issues]
    df["assignee"]  = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"]  = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"]    = df["fields.status.name"]
    df["priority"]  = df["fields.priority.name"].fillna("None")
    df["created"]   = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)
    df["area_destino"] = (
        df.get("fields.customfield_10043.value")
          .fillna("Sin Área")
          .astype(str)
    )

    # ─── Filtros dinámicos ──────────────────────────────────────────────
    statuses   = sorted(df["status"].dropna().astype(str).unique())
    priorities = sorted(df["priority"].dropna().astype(str).unique())
    assignees  = sorted(df["assignee"].dropna().astype(str).unique())
    areas      = sorted(df["area_destino"].dropna().astype(str).unique())

    sel_status  = st.sidebar.multiselect("Estados",   statuses,   statuses)
    sel_pri     = st.sidebar.multiselect("Prioridad", priorities, priorities)
    sel_ass     = st.sidebar.multiselect("Responsable", assignees, assignees)
    sel_area    = st.sidebar.multiselect("Área Destino", areas, areas)

    df = df[
        df["status"].isin(sel_status) &
        df["priority"].isin(sel_pri) &
        df["assignee"].isin(sel_ass) &
        df["area_destino"].isin(sel_area)
    ]
    if df.empty:
        st.info("No hay datos tras aplicar los filtros.")
        st.stop()

    # ─── Resumen IA ─────────────────────────────────────────────────────
    st.subheader("📝 Resumen IA")
    corpus = []
    for iss in issues:
        if iss.key not in df["key"].values:  # respetar filtros
            continue
        desc = iss.fields.description or ""
        coms = [c.body for c in iss.fields.comment.comments[:3]]
        corpus.append("\n".join([f"[{iss.key}] {desc}"] + coms))
    text_block = "\n\n---\n\n".join(corpus)[:16000]

    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        summary = summarise_openai(text_block)
    else:
        summary = summarise_bart(text_block)
    st.text_area("Resumen generado", summary, height=220)

    # ─── KPIs simples ───────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    resolved = df.dropna(subset=["fields.resolutiondate"]).copy()
    if not resolved.empty:
        resolved["resolve_days"] = (
            pd.to_datetime(resolved["fields.resolutiondate"], utc=True).dt.tz_localize(None)
            - resolved["created"]
        ).dt.days

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickets", len(df))
    col2.metric("Abiertos", df["fields.resolutiondate"].isna().sum())
    col3.metric("Media días abiertos", round(df["age_days"].mean(), 1))
    col4.metric(
        "Media días resolución",
        round(resolved["resolve_days"].mean(), 1) if not resolved.empty else "-",
    )

    # ─── Gráficos breves ────────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    s_counts = (
        df["status"].value_counts()
          .reset_index()
          .rename(columns={"index": "Estado", "status": "Cantidad"})
    )
    p_counts = (
        df["priority"].value_counts()
          .reset_index()
          .rename(columns={"index": "Prioridad", "priority": "Cantidad"})
    )

    ch_s = alt.Chart(s_counts).mark_bar().encode(x="Estado", y="Cantidad")
    ch_p = alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad")
    st.altair_chart(ch_s | ch_p, use_container_width=True)

    # ─── Exportar Excel ─────────────────────────────────────────────────
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buf.getvalue(), file_name="tickets.xlsx")


# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()
