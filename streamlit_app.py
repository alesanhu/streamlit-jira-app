"""
Streamlit - Jira Dashboard + Resumen AI
---------------------------------------
• KPIs, filtros, gráficos y export a Excel.
• Botón “Generar resumen”:
    – Con OPENAI_API_KEY → GPT-3.5-Turbo.
    – Sin clave → resumen local muy ligero (NLTK).
"""

from __future__ import annotations
import os, textwrap, re
from io import BytesIO
from datetime import datetime, UTC

import altair as alt
import pandas as pd
import streamlit as st
from jira import JIRA

# ════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ------------------------------------------------------------------------

def quote_list(vals: list[str]) -> str:
    """Escapa comillas simples para JQL."""
    return ", ".join("'{}'".format(v.replace("'", r"\'")) for v in vals)

@st.cache_resource(show_spinner=False)
def make_jira() -> JIRA | None:
    s = st.secrets
    if not all(k in s for k in ("JIRA_SERVER", "JIRA_USER", "JIRA_TOKEN")):
        st.sidebar.error("⚠️  Falta configurar `JIRA_*` en Secrets.")
        return None
    try:
        return JIRA(server=s["JIRA_SERVER"],
                    basic_auth=(s["JIRA_USER"], s["JIRA_TOKEN"]))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """Busca hasta 2000 issues. El Jira client se pasa con _ prefijo para que Streamlit no lo intente hashear."""
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []

def local_summary(text: str, max_sentences: int = 5) -> str:
    """Resumen sencillo (frecuencia de palabras) usando solo NLTK."""
    import nltk, heapq
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("spanish"))
    words = word_tokenize(re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúñÑ]", " ", text.lower()))
    freq = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1
    sents = sent_tokenize(text)
    scored = []
    for sent in sents:
        score = sum(freq.get(w.lower(), 0) for w in word_tokenize(sent))
        scored.append((score, sent))
    top = heapq.nlargest(max_sentences, scored)
    return "\n".join(s for _, s in sorted(top, key=lambda x: -x[0]))

def gpt_summary(text: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres analista de soporte. Resume puntos clave y alertas "
        "de los siguientes tickets de Jira:\n\n"
        "### Tickets\n" + text[:16000]
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()

# ════════════════════════════════════════════════════════════════════════
# APP
# ------------------------------------------------------------------------

def main() -> None:
    st.set_page_config("Jira Dashboard", layout="wide")
    st.title("📊 Gestión de Tickets en Jira")

    jira = make_jira()
    if jira is None:
        st.stop()

    # ── Filtros laterales ───────────────────────────────────────────────
    st.sidebar.header("🔍 Filtros")

    try:
        projects = sorted(p.key for p in jira.projects() if not p.raw.get("archived", False))
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyectos", projects, projects)

    today = datetime.now(UTC).date()
    start, end = st.sidebar.date_input(
        "Rango creación", (today - pd.Timedelta(days=30), today),
        help="Filtra por fecha de creación del ticket."
    )

    # JQL básico (solo proyecto + fecha)
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se obtuvieron tickets para los filtros actuales.")
        st.stop()

    # ── DataFrame base ──────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]       = [i.key for i in issues]
    df["summary"]   = [i.fields.summary for i in issues]
    df["assignee"]  = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["priority"]  = df["fields.priority.name"].fillna("None")
    df["status"]    = df["fields.status.name"]
    df["created"]   = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)
    df["area_destino"] = df.get("fields.customfield_10043.value", "Sin Área").fillna("Sin Área")

    # ── Filtros dependientes ────────────────────────────────────────────
    statuses   = sorted(map(str, df["status"].unique()))
    priorities = sorted(map(str, df["priority"].unique()))
    assignees  = sorted(map(str, df["assignee"].unique()))
    areas      = sorted(map(str, df["area_destino"].unique()))

    sel_status = st.sidebar.multiselect("Estados",   statuses,   statuses)
    sel_pri    = st.sidebar.multiselect("Prioridad", priorities, priorities)
    sel_ass    = st.sidebar.multiselect("Responsable", assignees,  assignees)
    sel_area   = st.sidebar.multiselect("Área destino", areas, areas)

    # Aplicar filtro local
    df = df[
        df["status"].isin(sel_status)
        & df["priority"].isin(sel_pri)
        & df["assignee"].isin(sel_ass)
        & df["area_destino"].isin(sel_area)
    ]

    # ── KPIs ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total tickets", len(df))
    col2.metric("Abiertos", df["fields.resolutiondate"].isna().sum())
    col3.metric("Promedio días abiertos", round(((datetime.now() - df["created"]).dt.days).mean(), 1))

    # ── Gráficos simples ───────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    s_counts = df["status"].value_counts().reset_index()
    s_counts.columns = ["Estado", "Cantidad"]
    p_counts = df["priority"].value_counts().reset_index()
    p_counts.columns = ["Prioridad", "Cantidad"]
    chart_s = alt.Chart(s_counts).mark_bar().encode(x="Estado", y="Cantidad")
    chart_p = alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad")
    st.altair_chart(chart_s | chart_p, use_container_width=True)

    # ── Export a Excel ─────────────────────────────────────────────────
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buff.getvalue(), file_name="tickets.xlsx")

    # ── Resumen AI bajo demanda ────────────────────────────────────────
    if st.button("✨ Generar resumen / alertas"):
        with st.spinner("Generando resumen…"):
            blocks = []
            for it in issues:
                desc = it.fields.description or ""
                comments = "\n".join(c.body for c in it.fields.comment.comments[:3])
                blocks.append(f"*{it.key}* — {it.fields.summary}\n{desc}\n{comments}")
            corpus = "\n\n---\n\n".join(blocks)

            if "OPENAI_API_KEY" in st.secrets:
                summary = gpt_summary(corpus)
            else:
                summary = local_summary(corpus)

        st.text_area("Resumen generado", summary, height=250)

# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
