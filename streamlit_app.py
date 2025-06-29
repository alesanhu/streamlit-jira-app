"""
Streamlit – Jira Dashboard con resumen automático
------------------------------------------------
• Lee issues de Jira usando la librería `jira`.
• Muestra KPIs, gráficos y tablas.
• Genera un resumen/alertas de las descripciones + comentarios.
  - Usa OpenAI GPT si `OPENAI_API_KEY` está definido.
  - Si no hay clave, cae a un modelo local (BART) vía 🤗 Transformers.

✱ Requisitos (añádelos a requirements.txt):
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

# ╭─  INTENTAR cargar dependencias de IA ───────────────────────────────────╮
try:
    from openai import OpenAI            # SDK ≥1.0
except ImportError:
    OpenAI = None                         # type: ignore

try:
    from transformers import pipeline     # modelo local
except ImportError:
    pipeline = None                       # type: ignore
# ╰──────────────────────────────────────────────────────────────────────────╯


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def quote_list(vals: list[str]) -> str:
    """Escapa valores para JQL →  'ISIL','PUCP' …"""
    escaped = [v.replace("'", "\\'") for v in vals]
    return ",".join(f"'{v}'" for v in escaped)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    """Devuelve un cliente Jira autenticado o None."""
    server = st.secrets.get("JIRA_SERVER")
    user   = st.secrets.get("JIRA_USER")
    token  = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.error("❌ Faltan credenciales de Jira en *Secrets*.")
        return None
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """Descarga hasta 2000 issues que cumplan el JQL.  
    *Nota*: el parámetro lleva guion bajo para que Streamlit no intente hashearlo.*
    """
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error al traer tickets: {e}")
        return []


def summarise_with_openai(text: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. Resume los puntos clave y riesgos de los siguientes "
        "tickets de Jira.\n\n### Tickets\n" + text
    )
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return res.choices[0].message.content.strip()


def summarise_with_bart(text: str) -> str:
    if pipeline is None:
        return "⚠️ No se pudo usar modelo local: instala *transformers* + *torch*."
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    chunks = textwrap.wrap(text, 3000)
    return "\n".join(
        summarizer(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
        for c in chunks
    )


# ────────────────────────────────────────────────────────────────────────────
#  UI principal
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config("Jira Dashboard + IA", layout="wide")
    st.title("📊 Jira Dashboard + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ── Filtros laterales ────────────────────────────────────────────────
    st.sidebar.header("🔍 Filtros")

    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyecto(s)", projects, projects)

    try:
        statuses = [s.name for s in jira.statuses()]
    except Exception:
        statuses = []
    sel_status = st.sidebar.multiselect("Estado(s)", statuses, statuses)

    try:
        priorities = [p.name for p in jira.priorities()]
    except Exception:
        priorities = []
        st.sidebar.warning("⚠️ No se pudieron cargar prioridades. Revisa el API-token.")
    sel_pri = st.sidebar.multiselect("Prioridad(es)", priorities, priorities)

    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input("Rango de creación", (today - pd.Timedelta(days=30), today))

    # ── JQL y fetch ─────────────────────────────────────────────────────
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("⌛ Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se encontraron tickets para esos filtros.")
        st.stop()

    # ── DataFrame base ─────────────────────────────────────────────────
    raw = [i.raw for i in issues]
    df = pd.json_normalize(raw)
    df["key"]      = [i.key for i in issues]
    df["summary"]  = [i.fields.summary for i in issues]
    df["assignee"] = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"] = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"]   = df["fields.status.name"]
    df["priority"] = df["fields.priority.name"].fillna("None")
    df["created"]  = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # ── Resumen AI ──────────────────────────────────────────────────────
    st.subheader("📝 Resumen automático")
    texts = []
    for iss in issues:
        desc = iss.fields.description or ""
        comments = [c.body for c in iss.fields.comment.comments[:3]]
        texts.append("\n".join([desc] + comments))
    corpus = "\n\n---\n\n".join(texts)[:16000]

    if "OPENAI_API_KEY" in st.secrets and OpenAI:
        summary = summarise_with_openai(corpus)
    else:
        summary = summarise_with_bart(corpus)

    st.text_area("Resumen/alertas", summary, height=220)

    # ── KPIs ────────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    resolved = df.dropna(subset=["fields.resolutiondate"]).copy()
    resolved["resolve_days"] = (
        pd.to_datetime(resolved["fields.resolutiondate"], utc=True).dt.tz_localize(None) - resolved["created"]
    ).dt.days

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("Abiertos", df["fields.resolutiondate"].isna().sum())
    c3.metric("⌀ días abiertos", round(df["age_days"].mean(), 1))
    c4.metric("⌀ días resolución", round(resolved["resolve_days"].mean(), 1) if not resolved.empty else "-")

    # ── Gráficos rápido con Altair ──────────────────────────────────────
    st.subheader("Distribución por Estado / Prioridad")
    status_cnt = df["status"].value_counts().reset_index().rename(columns={"index": "Estado", "status": "Tickets"})
    pri_cnt    = df["priority"].value_counts().reset_index().rename(columns={"index": "Prioridad", "priority": "Tickets"})
    bar1 = alt.Chart(status_cnt).mark_bar().encode(x="Estado", y="Tickets")
    bar2 = alt.Chart(pri_cnt).mark_bar().encode(x="Prioridad", y="Tickets")
    st.altair_chart(bar1 | bar2, use_container_width=True)

    # ── Exportar a Excel ────────────────────────────────────────────────
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Tickets", index=False)
    st.download_button("⬇️ Exportar Excel", buff.getvalue(), file_name="tickets_jira.xlsx")


if __name__ == "__main__":
    main()
