"""
📊 Streamlit ▸ Jira dashboard + resumen AI
-----------------------------------------
Requisitos mínimos   : pandas, streamlit, altair, jira, xlsxwriter
Resumen con OpenAI   : + openai
Resumen offline (op.) : + torch, transformers, sentencepiece
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

# ════════════ utilidades ═════════════════════════════════════════════════════
def quote_list(vals: list[str]) -> str:
    """Devuelve 'A','B','C' escapando comillas internas."""
    escaped = [v.replace("'", "\\'") for v in vals]
    return ", ".join(f"'{e}'" for e in escaped)


@st.cache_resource(show_spinner=False)
def jira_client() -> JIRA | None:
    srv, usr, tok = (st.secrets.get(k) for k in ("JIRA_SERVER", "JIRA_USER", "JIRA_TOKEN"))
    if not (srv and usr and tok):
        st.sidebar.error("👉 Añade JIRA_SERVER, JIRA_USER y JIRA_TOKEN en *Secrets*")
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
        st.error(f"Error obteniendo tickets: {e}")
        return []


# ════════════ resumen AI ═════════════════════════════════════════════════════
def summarise(text: str) -> str:
    # 1- OpenAI si hay clave
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "Eres un analista. Resume las ideas clave y riesgos de los siguientes tickets:\n\n" + text
            )
            r = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ OpenAI dio error: {e}"

    # 2- BART-CNN local (requiere torch/transformers/sentencepiece)
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
        chunks = textwrap.wrap(text, 2500)           # trozos ≲ 2 500 chars
        out = [
            summarizer(c, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
            for c in chunks
        ]
        return "\n\n".join(out)
    except Exception as e:
        return (
            "⚠️ No hay OPENAI_API_KEY y el modelo local no pudo cargarse "
            f"(falta torch/transformers o sentencepiece): {e}"
        )


# ════════════ interfaz Streamlit ═════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="Jira dashboard + resumen", layout="wide")
    st.title("📊 Jira dashboard + resumen AI")

    jira = jira_client()
    if jira is None:
        st.stop()

    # ── Filtros laterales ──────────────────────────────────────────────────
    st.sidebar.header("Filtros")

    # proyectos
    try:
        projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        projects = []
    sel_proj = st.sidebar.multiselect("Proyecto", projects, projects)

    # fechas
    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input("Rango creación", (today - pd.Timedelta(days=30), today))

    # JQL (solo proyecto+fecha: resto lo filtramos en memoria)
    clauses: list[str] = []
    if sel_proj:
        clauses.append(f"project in ({quote_list(sel_proj)})")
    clauses.append(f"created >= '{start}' AND created <= '{end}'")
    jql = " AND ".join(clauses) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No hay tickets para esos filtros.")
        st.stop()

    # ── DataFrame ──────────────────────────────────────────────────────────
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]       = [i.key for i in issues]
    df["assignee"]  = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["status"]    = df["fields.status.name"]
    df["priority"]  = df["fields.priority.name"].fillna("None")
    df["area_destino"] = df.get("fields.customfield_10043.value", "Sin Área")
    df["created"]   = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # filtros dinámicos
    sel_status = st.sidebar.multiselect("Estado",   sorted(df["status"].unique()),   sorted(df["status"].unique()))
    sel_pri    = st.sidebar.multiselect("Prioridad",sorted(df["priority"].unique()), sorted(df["priority"].unique()))
    sel_ass    = st.sidebar.multiselect("Responsable",sorted(df["assignee"].unique()), sorted(df["assignee"].unique()))
    sel_area   = st.sidebar.multiselect("Área destino",sorted(df["area_destino"].unique()), sorted(df["area_destino"].unique()))

    df = df[
        df["status"].isin(sel_status)
        & df["priority"].isin(sel_pri)
        & df["assignee"].isin(sel_ass)
        & df["area_destino"].isin(sel_area)
    ]

    # ── KPIs ───────────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days

    st.subheader("KPI")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tickets", len(df))
    c2.metric("Asignados", (df["assignee"]!="Sin asignar").sum())
    c3.metric("Prom. días abiertos", round(df["age_days"].mean(),1))
    c4.metric("Último ticket", df["created"].max().date())

    # ── Gráficos simples ──────────────────────────────────────────────────
    st.subheader("Distribución por Estado y Prioridad")
    s_counts = df["status"].value_counts().reset_index().rename(columns={"index":"Estado", "status":"Cantidad"})
    p_counts = df["priority"].value_counts().reset_index().rename(columns={"index":"Prioridad", "priority":"Cantidad"})
    chart_s  = alt.Chart(s_counts).mark_bar().encode(x="Estado",    y="Cantidad")
    chart_p  = alt.Chart(p_counts).mark_bar().encode(x="Prioridad", y="Cantidad")
    st.altair_chart(chart_s | chart_p, use_container_width=True)

    # ── Botón para generar resumen ────────────────────────────────────────
    if st.button("Generar resumen AI", type="primary"):
        with st.spinner("Resumiendo…"):
            texts = []
            for issue in issues:
                desc = issue.fields.description or ""
                comments = "\n".join(c.body for c in issue.fields.comment.comments[:3])  # 1ros 3 comentarios
                texts.append(desc + "\n" + comments)
            corpus = "\n\n---\n\n".join(texts)[:16000]
            st.text_area("Resumen generado", summarise(corpus), height=250)

    # ── Exportar Excel ────────────────────────────────────────────────────
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Tickets", index=False)
    st.download_button("⬇️ Exportar Excel", buffer.getvalue(), file_name="tickets.xlsx")

# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
