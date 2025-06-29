"""
Streamlit – Jira Dashboard + Resumen (sin dependencias pesadas)
Requiere en requirements.txt:
    streamlit
    pandas
    altair
    jira
    openai           # opcional: solo si usas tu clave
    nltk             # ligero, para el fallback de resumen
    xlsxwriter
"""
from __future__ import annotations
import os, html, textwrap
from datetime import datetime
from io import BytesIO

import pandas as pd
import altair as alt
import streamlit as st
from jira import JIRA
import nltk

# ───────────── helpers ────────────────────────────────────────────
@st.cache_resource
def nltk_setup():
    # solo la primera vez descarga stopwords
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

@st.cache_resource
def jira_client():
    s = st.secrets
    if not all(k in s for k in ("JIRA_SERVER", "JIRA_USER", "JIRA_TOKEN")):
        st.sidebar.error("🔑 Configura credenciales Jira en secrets.")
        return None
    try:
        return JIRA(server=s["JIRA_SERVER"],
                    basic_auth=(s["JIRA_USER"], s["JIRA_TOKEN"]))
    except Exception as e:
        st.sidebar.error(f"Error conectando a Jira: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error Jira {e}")
        return []

def quote(vals:list[str]) -> str:
    return ", ".join(f"'{v.replace(\"'\", \"\\'\")}'" for v in vals)

def quick_summary(text:str, max_sent:int = 6)->str:
    """Resumido con NLTK – elige las frases más largas como proxy de relevancia."""
    nltk_setup()
    sents = nltk.sent_tokenize(text)
    sents = sorted(sents, key=len, reverse=True)[:max_sent]
    return " ".join(sents)

def gpt_summary(text:str)->str:
    from openai import OpenAI
    cli = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    msg = ("Resume los puntos clave y alertas de estos tickets:\n\n"+text)[:15000]
    r = cli.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role":"user","content":msg}],
            max_tokens=400, temperature=0.4)
    return r.choices[0].message.content.strip()

# ───────────── UI ─────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Jira Dashboard", layout="wide")
    st.title("📊 Jira Dashboard")

    jira = jira_client()
    if jira is None:
        st.stop()

    # ----- filtros laterales -----
    st.sidebar.header("Filtros Jira")
    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects=[]
    sel_proj = st.sidebar.multiselect("Proyecto(s)", projects, projects)

    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input(
        "Rango de creación", (today-pd.Timedelta(days=30), today))
    if isinstance(start, datetime):  # si solo devolvió un valor
        start, end = start.date(), today

    jql = []
    if sel_proj: jql.append(f"project in ({quote(sel_proj)})")
    jql.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(jql) + " order by created desc"

    issues = fetch_issues(jira, jql)
    if not issues:
        st.warning("No hay tickets para los filtros elegidos.")
        st.stop()

    # ----- DataFrame base -----
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]        = [i.key for i in issues]
    df["status"]     = df["fields.status.name"]
    df["priority"]   = df["fields.priority.name"].fillna("None")
    df["assignee"]   = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["area_dest"]  = df.get("fields.customfield_10043.value", "Sin Área")
    df["created"]    = pd.to_datetime(df["fields.created"]).dt.date

    # filtros dinámicos
    sel_status = st.sidebar.multiselect("Estado", sorted(df["status"].unique()),
                                        sorted(df["status"].unique()))
    sel_pri    = st.sidebar.multiselect("Prioridad", sorted(df["priority"].unique()),
                                        sorted(df["priority"].unique()))
    sel_ass    = st.sidebar.multiselect("Responsable", sorted(df["assignee"].unique()),
                                        sorted(df["assignee"].unique()))
    sel_area   = st.sidebar.multiselect("Área destino", sorted(df["area_dest"].unique()),
                                        sorted(df["area_dest"].unique()))

    df = df[df["status"].isin(sel_status)
            & df["priority"].isin(sel_pri)
            & df["assignee"].isin(sel_ass)
            & df["area_dest"].isin(sel_area)]

    st.subheader(f"Tickets filtrados: {len(df)}")

    # ----- KPIs -----
    col1,col2 = st.columns(2)
    col1.metric("Total", len(df))
    abiertos = df["fields.resolutiondate"].isna().sum()
    col2.metric("Abiertos", abiertos)

    # ----- Gráficos -----
    st.subheader("Distribuciones")
    s_counts = df["status"].value_counts().reset_index().rename(
        columns={"index":"Estado","status":"Cantidad"})
    p_counts = df["priority"].value_counts().reset_index().rename(
        columns={"index":"Prioridad","priority":"Cantidad"})
    chart_s = alt.Chart(s_counts).mark_bar().encode(x="Estado", y="Cantidad")
    chart_p = alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad")
    st.altair_chart(chart_s | chart_p, use_container_width=True)

    # ----- Resumen AI bajo demanda -----
    with st.expander("📝 Generar resumen / alertas", expanded=False):
        if st.button("Generar resumen ahora"):
            corpus = []
            for it in issues[:60]:  # máx 60 para no explotar
                txt = (it.fields.description or "") + "\n".join(
                      c.body for c in it.fields.comment.comments[:2])
                corpus.append(html.unescape(txt))
            long_text = "\n\n---\n\n".join(corpus)

            if "OPENAI_API_KEY" in st.secrets and os.getenv("OPENAI_API_KEY"):
                st.info("Usando OpenAI…")
                summary = gpt_summary(long_text)
            else:
                st.info("Usando resumen rápido local…")
                summary = quick_summary(long_text)

            st.text_area("Resumen", summary, height=250)

    # ----- Exportar -----
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="tickets")
    st.download_button("⬇️ Exportar a Excel", buf.getvalue(),
                       file_name="tickets.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
