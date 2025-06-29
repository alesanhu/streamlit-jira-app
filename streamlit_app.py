"""
Streamlit – Jira Dashboard con resumen automático
-------------------------------------------------
• Lee issues de Jira mediante ``jira`` y los muestra en tablas, KPIs y gráficos.
• Genera un resumen (descripción + primeros comentarios) con:
      – OpenAI GPT si ``OPENAI_API_KEY`` está definido.
      – Facebook BART-large-cnn offline (transformers) como fallback.
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

# ─────── Inteligencia artificial ────────
try:
    from openai import OpenAI          # SDK ≥1.0
except ImportError:
    OpenAI = None                      # type: ignore

try:
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None                    # type: ignore

# ═════════ Helper funciones ═════════════

def quote_list(vals: list[str]) -> str:
    """Escapa cada valor con comillas simples para JQL."""
    # →  'ISIL','PUCP','UDEP'
    return ",".join("'" + v.replace("'", "\\'") + "'" for v in vals)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    srv, usr, tok = (
        st.secrets.get("JIRA_SERVER"),
        st.secrets.get("JIRA_USER"),
        st.secrets.get("JIRA_TOKEN"),
    )
    if not (srv and usr and tok):
        st.sidebar.error("⚠️ Falta JIRA_SERVER / JIRA_USER / JIRA_TOKEN en Secrets.")
        return None
    try:
        return JIRA(server=srv, basic_auth=(usr, tok))
    except Exception as e:
        st.sidebar.error(f"Error de conexión a Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(jira: JIRA, jql: str):
    """Devuelve Issues + comentarios (máx 2000)."""
    try:
        return jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error al traer tickets: {e}")
        return []


def summarise_openai(text: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. A partir de las descripciones y comentarios de Jira, "
        "resume puntos claves, riesgos o bloqueos:\n\n### Tickets:\n" + text
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def summarise_bart(text: str) -> str:
    if pipeline is None:
        return "❌ transformers ausente. Agrega `transformers torch sentencepiece` a requirements."
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    chunks = textwrap.wrap(text, 3000)
    outs = [summarizer(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"] for c in chunks]
    return "\n".join(outs)

# ═════════ UI principal ═════════════════

def main() -> None:
    st.set_page_config("Jira Dashboard + Resumen", layout="wide")
    st.title("📊 Jira Dashboard + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ─── Filtros base (solo proyecto y rango fechas) ────────────────────
    st.sidebar.header("🎛 Filtros")

    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []

    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input(
        "Rango creación", (today - pd.Timedelta(days=30), today)
    )

    # ─── Construir JQL y obtener issues ─────────────────────────────────
    jql_parts: list[str] = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se obtuvieron tickets para los filtros actuales.")
        st.stop()

    # ─── DataFrame base ────────────────────────────────────────────────
    raw = [i.raw for i in issues]
    df = pd.json_normalize(raw)
    df["key"]         = [i.key for i in issues]
    df["summary"]     = [i.fields.summary for i in issues]
    df["assignee"]    = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"]    = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"]      = df["fields.status.name"]
    df["priority"]    = df["fields.priority.name"].fillna("None")
    df["created"]     = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)
    df["area_destino"] = df.get("fields.customfield_10043.value", "Sin Área")

    # ─── Filtros dinámicos (responsable / estado / prioridad / área) ───
    dyn_status   = sorted(df["status"].unique())
    dyn_pri      = sorted(df["priority"].unique())
    dyn_assignee = sorted(df["assignee"].unique())
    dyn_area     = sorted(df["area_destino"].unique())

    sel_status = st.sidebar.multiselect("Estados",   dyn_status,   dyn_status)
    sel_pri    = st.sidebar.multiselect("Prioridad", dyn_pri,      dyn_pri)
    sel_ass    = st.sidebar.multiselect("Responsable", dyn_assignee, dyn_assignee)
    sel_area   = st.sidebar.multiselect("Área destino", dyn_area, dyn_area)

    df = df[
        df["status"].isin(sel_status) &
        df["priority"].isin(sel_pri)  &
        df["assignee"].isin(sel_ass)  &
        df["area_destino"].isin(sel_area)
    ]

    # ─── Resumen AI ─────────────────────────────────────────────────────
    st.subheader("📝 Resumen AI")
    texts = []
    for iss in issues:
        if iss.key not in df["key"].values:
            continue  # filtrado afuera
        descr = iss.fields.description or ""
        comms = [c.body for c in iss.fields.comment.comments[:3]]
        texts.append(descr + "\n" + "\n".join(comms))
    full_text = "\n\n---\n\n".join(texts)[:16000] or "Sin descripciones."

    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        summary = summarise_openai(full_text)
    else:
        summary = summarise_bart(full_text)
    st.text_area("Resumen generado", summary, height=220)

    # ─── KPIs rápidos ──────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    open_tickets = df[df["fields.resolutiondate"].isna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets totales", len(df))
    col2.metric("Abiertos", len(open_tickets))
    col3.metric("Edad media (días)", round(df["age_days"].mean(), 1))

    # ─── Gráficos estado / prioridad ───────────────────────────────────
    st.subheader("📊 Distribución")
    status_counts = df["status"].value_counts().reset_index().rename(columns={"index": "Estado", "status": "Cantidad"})
    pri_counts    = df["priority"].value_counts().reset_index().rename(columns={"index": "Prioridad", "priority": "Cantidad"})

    bar_status = alt.Chart(status_counts).mark_bar().encode(x="Estado:N", y="Cantidad:Q", tooltip=["Estado", "Cantidad"])
    bar_priority = alt.Chart(pri_counts).mark_bar().encode(x="Prioridad:N", y="Cantidad:Q", tooltip=["Prioridad", "Cantidad"])
    st.altair_chart(bar_status | bar_priority, use_container_width=True)

    # ─── Exportar a Excel ───────────────────────────────────────────────
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        df.to_excel(wr, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buf.getvalue(), file_name="tickets.xlsx")

# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()

