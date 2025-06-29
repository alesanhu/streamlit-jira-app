"""
Streamlit – Jira Dashboard con resumen automático
-------------------------------------------------
• Lee issues de Jira (librería `jira`).
• KPI, gráficos y tablas.
• Resumen/alertas de `fields.description` + 3 primeros comentarios.
  · Usa OpenAI GPT si `OPENAI_API_KEY` está en st.secrets.
  · Si no hay clave, muestra un aviso (sin transformers locales).
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from io import BytesIO
from typing import List

import altair as alt
import pandas as pd
import streamlit as st
from jira import JIRA

# ======== INTENTAR usar OpenAI si hay clave ===============================
try:
    from openai import OpenAI  # SDK ≥1.0
except ImportError:  # openai no instalado
    OpenAI = None  # type: ignore


# ══════════════ Helpers ───────────────────────────────────────────────────


def quote_list(vals: List[str]) -> str:
    """
    Devuelve `'A','B'…` escapando comillas simples internas.
    """
    escaped = [v.replace("'", "\\'") for v in vals]
    return ",".join(f"'{e}'" for e in escaped)


@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    """
    Conecta a Jira usando las credenciales guardadas en st.secrets.
    """
    server = st.secrets.get("JIRA_SERVER")
    user = st.secrets.get("JIRA_USER")
    token = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.error("Faltan credenciales Jira en Secrets.")
        return None
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch_issues(_jira: JIRA, jql: str):
    """
    Descarga hasta 2000 issues que cumplan el JQL.
    (El parámetro va con guion bajo para que Streamlit no intente hashearlo)
    """
    try:
        return _jira.search_issues(jql, maxResults=2000, expand="comments")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []


def summarise_with_openai(text: str) -> str:
    """Resumen vía OpenAI GPT."""
    if OpenAI is None:
        return "❌ openai-python no está instalado."
    if "OPENAI_API_KEY" not in st.secrets:
        return "🔑 No hay OPENAI_API_KEY en Secrets: sin resumen GPT."

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        "Eres un analista. Resume los puntos clave y alertas presentes en estos tickets de Jira:\n\n"
        "### Tickets\n" + text
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ══════════════ App ───────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Jira Dashboard + Resumen AI", layout="wide")
    st.title("📊 Gestión de Tickets en Jira + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ----------------- Filtros laterales ----------------------------------
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
        st.sidebar.warning("No se pudieron cargar Prioridades (¿token Jira correcto?).")
    sel_priority = st.sidebar.multiselect("Prioridades", priorities, priorities)

    today = datetime.utcnow().date()
    start, end = st.sidebar.date_input(
        "Rango fechas creación",
        (today - pd.Timedelta(days=30), today),
    )

    # ----------------- Construir JQL --------------------------------------
    # ── JQL y carga (solo proyecto + fechas) ──────────────────────────────
jql_parts = []
if sel_proj:
    jql_parts.append(f"project in ({quote_list(sel_proj)})")

# Rango de fechas
jql_parts.append(f"created >= '{start}' AND created <= '{end}'")

# 👇 NO añadimos ni status ni priority aquí
jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

with st.spinner("Cargando tickets de Jira…"):
    issues = fetch_issues(jira, jql)


if not issues:
        st.warning("No hay tickets para los filtros elegidos.")
        st.stop()

    # ----------------- DataFrame base -------------------------------------
    raw = [i.raw for i in issues]
    df = pd.json_normalize(raw)
    df["key"] = [i.key for i in issues]
    df["summary"] = [i.fields.summary for i in issues]
    df["assignee"] = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"] = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["Estado"] = df["fields.status.name"]
    df["Prioridad"] = df["fields.priority.name"].fillna("None")
    df["created"] = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)

    # ----------------- Resumen AI -----------------------------------------
    st.subheader("📝 Resumen / Alertas")
    texts = []
    for issue in issues:
        body = issue.fields.description or ""
        first_comments = "\n".join(c.body for c in issue.fields.comment.comments[:3])
        texts.append(f"*{issue.key}* – {issue.fields.summary}\n{body}\n{first_comments}")
    corpus = "\n\n---\n\n".join(texts)[:16000]  # límite de prompt

    summary = summarise_with_openai(corpus)
    st.text_area("Resumen generado", summary, height=220)

    # ----------------- KPIs ----------------------------------------------
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    resolved_df = df.dropna(subset=["fields.resolutiondate"]).copy()
    resolved_df["resolve_days"] = (
        pd.to_datetime(resolved_df["fields.resolutiondate"], utc=True)
        .dt.tz_localize(None)
        - resolved_df["created"]
    ).dt.days

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total", len(df))
    k2.metric("Abiertos", df["fields.resolutiondate"].isna().sum())
    k3.metric("Media días abiertos", round(df["age_days"].mean(), 1))
    k4.metric(
        "Media días resolución",
        round(resolved_df["resolve_days"].mean(), 1) if not resolved_df.empty else "-",
    )

    # ----------------- Gráficos -------------------------------------------
    st.subheader("Distribución por Estado y Prioridad")

    status_counts = df["Estado"].value_counts().reset_index().rename(
        columns={"index": "Estado", "Estado": "Cantidad"}
    )
    pri_counts = df["Prioridad"].value_counts().reset_index().rename(
        columns={"index": "Prioridad", "Prioridad": "Cantidad"}
    )

    bar1 = alt.Chart(status_counts).mark_bar().encode(x="Estado:N", y="Cantidad:Q")
    bar2 = alt.Chart(pri_counts).mark_bar().encode(x="Prioridad:N", y="Cantidad:Q")
    st.altair_chart(bar1 | bar2, use_container_width=True)

    # ----------------- Tabla responsable ----------------------------------
    st.subheader("📋 Tickets por Responsable")
    agg = df.groupby("assignee").size().reset_index(name="tickets").sort_values("tickets", ascending=False)
    st.dataframe(agg, use_container_width=True)

    # ----------------- Exportar -------------------------------------------
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buffer.getvalue(), "tickets.xlsx")


if __name__ == "__main__":
    main()
