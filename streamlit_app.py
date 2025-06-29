"""
Streamlit – Jira Dashboard con resumen AI (sólo OpenAI)
-------------------------------------------------------
Requisitos (requirements.txt):
    streamlit
    jira
    pandas
    altair
    openai           # <- sólo si usarás la parte GPT
    xlsxwriter
"""

from __future__ import annotations
import textwrap, os
from datetime import datetime
from io import BytesIO

import pandas as pd
import altair as alt
import streamlit as st
from jira import JIRA

# ╭─ Helpers ───────────────────────────────────────────────────────────────╮
def quote_list(vals: list[str]) -> str:
    return ",".join(f"'{v.replace(\"'\", \"\\\\'\")}'" for v in vals)

@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    srv, usr, tok = (st.secrets.get(k) for k in ("JIRA_SERVER","JIRA_USER","JIRA_TOKEN"))
    if not all((srv, usr, tok)):
        st.sidebar.error("⚠️ Añade JIRA_SERVER / JIRA_USER / JIRA_TOKEN en *Secrets*")
        return None
    try:
        return JIRA(server=srv, basic_auth=(usr, tok))
    except Exception as e:
        st.sidebar.error(f"Error al conectar Jira: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_issues(jira:JIRA, jql:str):
    try:
        return jira.search_issues(jql, maxResults=2000, expand="comment")
    except Exception as e:
        st.error(f"Error leyendo tickets: {e}")
        return []

# ── Resumen GPT (solo si hay clave) ───────────────────────────────────────
def gpt_summary(text:str)->str:
    if "OPENAI_API_KEY" not in st.secrets:
        return "🔒 Sin OPENAI_API_KEY → no se generó resumen."
    from openai import OpenAI           # se importa sólo si existe
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = ("Eres un analista. Extrae insights y alertas de estos tickets:\n\n"
              "### Texto\n"+text)
    r = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400, temperature=.4)
    return r.choices[0].message.content.strip()

# ╭─ App ───────────────────────────────────────────────────────────────────╮
def main():
    st.set_page_config("Jira Dashboard", layout="wide")
    st.title("📊 Jira Dashboard + Resumen AI")

    jira = create_jira_client()
    if jira is None: st.stop()

    # ── Sidebar filtros ──────────────────────────────────────────────────
    st.sidebar.header("Filtros")

    try: projects = [p.key for p in jira.projects() if not p.raw.get("archived")]
    except: projects=[]
    projs = st.sidebar.multiselect("Proyecto(s)", projects, projects)

    try: statuses = [s.name for s in jira.statuses()]
    except: statuses=[]
    stats = st.sidebar.multiselect("Estado(s)", statuses, statuses)

    try: priorities = [p.name for p in jira.priorities()]
    except: priorities=[]
    pris  = st.sidebar.multiselect("Prioridad(es)", priorities, priorities)

    today = datetime.utcnow().date()
    start,end = st.sidebar.date_input("Creación entre", (today-pd.Timedelta(days=30), today))

    # ── JQL + fetch ──────────────────────────────────────────────────────
    jql = f"created >= '{start}' AND created <= '{end}'"
    if projs: jql = f"project in ({quote_list(projs)}) AND "+jql
    with st.spinner("Consultando Jira …"):
        issues = fetch_issues(jira, jql)
    if not issues: st.warning("No hay tickets."); st.stop()

    # ── DataFrame base ───────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"] = [i.key for i in issues]
    df["status"] = df["fields.status.name"];  df = df[df["status"].isin(stats)]
    df["priority"] = df["fields.priority.name"].fillna("None");  df = df[df["priority"].isin(pris)]
    df["created"] = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # ── Resumen (descripción + 1er comentario) ───────────────────────────
    texts=[]
    for it in issues:
        body = it.fields.description or ""
        first_comment = it.fields.comment.comments[0].body if it.fields.comment.comments else ""
        texts.append(body+"\n"+first_comment)
    summary = gpt_summary("\n\n---\n\n".join(textwrap.shorten(t,4000) for t in texts)[:16000])
    st.subheader("📝 Resumen / alertas")
    st.text_area("GPT", summary, height=200)

    # ── KPIs ─────────────────────────────────────────────────────────────
    now = pd.Timestamp.now(); df["age"]= (now-df["created"]).dt.days
    st.metric("Total", len(df)), st.metric("Abiertos", df["fields.resolutiondate"].isna().sum())

    # ── Charts ───────────────────────────────────────────────────────────
    st.subheader("Distribución")
    st.altair_chart(
        (alt.Chart(df.groupby("status").size().reset_index(name="count"))
            .mark_bar().encode(x="status",y="count"))
        |(alt.Chart(df.groupby("priority").size().reset_index(name="count"))
            .mark_bar().encode(x="priority",y="count")),
        use_container_width=True)

    # ── Export ───────────────────────────────────────────────────────────
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine="xlsxwriter") as w: df.to_excel(w,index=False)
    st.download_button("⬇️ Excel", buf.getvalue(),"tickets.xlsx")

if __name__=="__main__":
    main()
