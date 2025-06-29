"""
Streamlit – Jira Dashboard + resumen AI
---------------------------------------
• Lee issues de Jira (librería `jira`).
• KPIs, gráficos, exportación Excel.
• Resumen / alertas de descripciones y comentarios:
    ◦ Si existe `OPENAI_API_KEY` → GPT-3.5-turbo.
    ◦ Si no, recurre a BART-large-cnn local (transformers).
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

# ─── AI helpers ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI               # >= 1.0
except ImportError:
    OpenAI = None                           # type: ignore

try:
    from transformers import pipeline       # sólo se importa si hace falta
except ImportError:
    pipeline = None                         # type: ignore


# ─── funciones auxiliares ────────────────────────────────────────────────────
def quote_list(vals: list[str]) -> str:
    """Escapa / cita para JQL → 'ISIL','PUCP' …"""
    return ",".join(f"'{v.replace(\"'\", \"\\\\'\")}'" for v in vals)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    """Crea y cachea el cliente Jira usando las credenciales de st.secrets."""
    srv, usr, tok = (
        st.secrets.get("JIRA_SERVER"),
        st.secrets.get("JIRA_USER"),
        st.secrets.get("JIRA_TOKEN"),
    )
    if not (srv and usr and tok):
        st.sidebar.error("⚠️ Faltan JIRA_SERVER/JIRA_USER/JIRA_TOKEN en *Secrets*.")
        return None
    try:
        return JIRA(server=srv, basic_auth=(usr, tok))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """Descarga hasta 2000 issues + comentarios (expand=comments)."""
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error al pedir issues: {e}")
        return []


def summarise_openai(text: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. Resume brevemente los puntos clave y cualquier alerta "
        "importante de los siguientes tickets de Jira.\n\n---\n\n" + text
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
        return "❌ Sin transformers/torch: no se puede resumir localmente."
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    chunks = textwrap.wrap(text, 3000)
    outs = [summarizer(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"] for c in chunks]
    return "\n".join(outs)


# ─── Streamlit app ───────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Jira Dashboard + Resumen AI", layout="wide")
    st.title("📊 Dashboard de Tickets Jira + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ╭─ Filtros laterales ────────────────────────────────────────────────
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
    except Exception:
        priorities = []
        st.sidebar.warning("No se pudieron cargar prioridades (¿token?).")
    sel_priority = st.sidebar.multiselect("Prioridades", priorities, priorities)

    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input(
        "Rango fechas creación", (today - pd.Timedelta(days=30), today)
    )

    # ╭─ Construir JQL ────────────────────────────────────────────────────
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("📥 Descargando tickets…"):
        issues = fetch_issues(jira, jql)
    if not issues:
        st.warning("No hay tickets para los filtros.")
        st.stop()

    # ╭─ DataFrame base ──────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"] = [i.key for i in issues]
    df["summary"] = [i.fields.summary for i in issues]
    df["assignee"] = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"] = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"] = df["fields.status.name"]
    df["priority"] = df["fields.priority.name"].fillna("None")
    df["created"] = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # ╭─ Resumen AI ───────────────────────────────────────────────────────
    st.subheader("📝 Resumen automático")
    texts = []
    for it in issues:
        desc = it.fields.description or ""
        comments = "\n".join(c.body for c in it.fields.comment.comments[:5])
        texts.append(desc + "\n" + comments)
    corpus = "\n\n---\n\n".join(texts)[:16000]  # límite prudente

    if "OPENAI_API_KEY" in st.secrets and OpenAI is not None:
        summary = summarise_openai(corpus)
    else:
        summary = summarise_bart(corpus)
    st.text_area("Resumen / Alertas", summary, height=220)

    # ╭─ KPIs rápidos ────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_d"] = (now - df["created"]).dt.days
    resolved = df.dropna(subset=["fields.resolutiondate"]).copy()
    if not resolved.empty:
        resolved["resolve_d"] = (
            pd.to_datetime(resolved["fields.resolutiondate"], utc=True).dt.tz_localize(None) - resolved["created"]
        ).dt.days

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets", len(df))
    k2.metric("Abiertos", df["fields.resolutiondate"].isna().sum())
    k3.metric("⌀ días abiertos", round(df["age_d"].mean(), 1))
    k4.metric("⌀ días resolución", round(resolved["resolve_d"].mean(), 1) if not resolved.empty else "–")

    # ╭─ Gráficos barra ──────────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    st.altair_chart(
        (alt.Chart(df).mark_bar().encode(x="status:N", y="count()")).properties(width=450) |
        (alt.Chart(df).mark_bar().encode(x="priority:N", y="count()")).properties(width=450),
        use_container_width=True,
    )

    # ╭─ Exportar a Excel ────────────────────────────────────────────────
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar Excel", buf.getvalue(), file_name="tickets.xlsx")


if __name__ == "__main__":
    main()
