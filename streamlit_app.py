"""
Streamlit – Jira dashboard con resumen automático
-------------------------------------------------
• Lee issues de Jira con `jira` (expande comentarios).
• Filtros dinámicos: proyecto, fecha, estado, prioridad,
  responsable, área-destino (customfield_10043).
• Muestra KPIs, gráficos Altair y permite exportar a Excel.
• Resume descripciones + primeros comentarios:
  - Si hay OPENAI_API_KEY en `st.secrets` ➟ lo hace con GPT-3.5-turbo.
  - Si no, muestra aviso de que no hay clave.
Requisitos principales (requirements.txt):
    jira
    pandas
    streamlit
    altair
    openai
    xlsxwriter
"""
from __future__ import annotations

import textwrap
from datetime import datetime
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
from jira import JIRA

# ─────────────────────── Helpers ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def create_jira_client() -> JIRA | None:
    """Devuelve un cliente JIRA usando credenciales guardadas en st.secrets."""
    server = st.secrets.get("JIRA_SERVER")
    user   = st.secrets.get("JIRA_USER")
    token  = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.error("⚠️ Faltan JIRA_SERVER, JIRA_USER o JIRA_TOKEN en Secrets.")
        return None
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.sidebar.error(f"Error conexión Jira: {e}")
        return None


def quote_list(vals: list[str]) -> str:
    """Convierte ['ISIL','PUCP'] ➟  'ISIL','PUCP' escapando comillas simples."""
    return ",".join(f"'{v.replace(\"'\", \"\\\\'\")}'" for v in vals)


@st.cache_data(show_spinner=False)
def fetch_issues(jira: JIRA, jql: str):
    """Descarga hasta 2000 issues y expande comentarios."""
    try:
        return jira.search_issues(jql, maxResults=2000, expand="comment")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []


def summarize_tickets(text: str) -> str:
    """Genera un resumen con OpenAI si hay clave; si no, devuelve aviso."""
    if "OPENAI_API_KEY" not in st.secrets:
        return "🔑 No hay OPENAI_API_KEY en Secrets; no se generó resumen."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = (
            "Eres analista de soporte. Resume los puntos clave y alertas de los "
            "siguientes tickets de Jira:\n\n### Tickets\n" + text
        )
        rsp = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error OpenAI: {e}"


# ─────────────────────── App principal ──────────────────────────────────────
def main():
    st.set_page_config("Jira Dashboard + AI Summary", layout="wide")
    st.title("📊 Gestión de Tickets Jira + Resumen AI")

    jira = create_jira_client()
    if jira is None:
        st.stop()

    # ---------------------- Filtros laterales ------------------------------
    st.sidebar.header("🔍 Filtros")

    try:
        all_projects = [p.key for p in jira.projects() if not p.raw.get("archived", False)]
    except Exception:
        all_projects = []
    sel_proj = st.sidebar.multiselect("Proyecto(s)", all_projects, all_projects)

    today = datetime.utcnow().date()
    start_date, end_date = st.sidebar.date_input(
        "Rango creación", (today - pd.Timedelta(days=30), today)
    )

    # -- JQL solo con proyecto + fecha (los demás filtros se aplican local) --
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start_date}' AND created <= '{end_date}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets de Jira…"):
        issues = fetch_issues(jira, jql)

    if not issues:
        st.warning("No se obtuvieron tickets con los filtros actuales.")
        st.stop()

    # -------------------- DataFrame base -----------------------------------
    df = pd.json_normalize([i.raw for i in issues])
    df["key"]        = [i.key for i in issues]
    df["summary"]    = [i.fields.summary for i in issues]
    df["assignee"]   = df["fields.assignee.displayName"].fillna("Sin asignar")
    df["reporter"]   = df["fields.reporter.displayName"].fillna("Sin asignar")
    df["status"]     = df["fields.status.name"]
    df["priority"]   = df["fields.priority.name"].fillna("Sin prioridad")
    df["created"]    = pd.to_datetime(df["fields.created"], utc=True).dt.tz_localize(None)
    df["area_destino"] = df.get("fields.customfield_10043.value", "Sin Área")

    # -------------------- Opciones dinámicas -------------------------------
    statuses   = sorted(df["status"].dropna().unique())
    priorities = sorted(df["priority"].dropna().unique())
    assignees  = sorted(df["assignee"].dropna().unique())
    areas      = sorted(df["area_destino"].dropna().unique())

    sel_status = st.sidebar.multiselect("Estados", statuses, statuses)
    sel_pri    = st.sidebar.multiselect("Prioridades", priorities, priorities)
    sel_ass    = st.sidebar.multiselect("Responsable", assignees, assignees)
    sel_area   = st.sidebar.multiselect("Área Destino", areas, areas)

    # -------------------- Filtrado local -----------------------------------
    df = df[
        df["status"      ].isin(sel_status) &
        df["priority"    ].isin(sel_pri)    &
        df["assignee"    ].isin(sel_ass)    &
        df["area_destino"].isin(sel_area)
    ]

    # -------------------- Resumen AI ---------------------------------------
    st.subheader("📝 Resumen / alertas")
    ticket_texts = []
    for issue in issues:
        if issue.key not in df["key"].values:
            continue  # ticket filtrado fuera
        body = issue.fields.description or ""
        comments = [c.body for c in issue.fields.comment.comments[:3]]
        ticket_texts.append(f"*{issue.key}* – {issue.fields.summary}\n" + body + "\n".join(comments))
    corpus = "\n\n-----\n\n".join(ticket_texts)[:16000]  # límite de tokens

    summary = summarize_tickets(corpus)
    st.text_area("Resumen generado", summary, height=220)

    # -------------------- KPIs --------------------------------------------
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created"]).dt.days
    open_tickets = df[df["fields.resolutiondate"].isna()]
    closed       = df[~df["fields.resolutiondate"].isna()]
    closed["resolve_days"] = (
        pd.to_datetime(closed["fields.resolutiondate"], utc=True).dt.tz_localize(None) - closed["created"]
    ).dt.days

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total", len(df))
    k2.metric("Abiertos", len(open_tickets))
    k3.metric("Promedio días abiertos", round(open_tickets["age_days"].mean(), 1) if not open_tickets.empty else "-")
    k4.metric("Promedio días resolución", round(closed["resolve_days"].mean(), 1) if not closed.empty else "-")

    # -------------------- Gráficos ----------------------------------------
    st.subheader("Distribución por Estado y Prioridad")
    state_counts = df["status"].value_counts().reset_index(names=["Estado", "Cantidad"])
    pri_counts   = df["priority"].value_counts().reset_index(names=["Prioridad", "Cantidad"])
    chart_state = alt.Chart(state_counts).mark_bar(size=40).encode(x="Estado:N", y="Cantidad:Q")
    chart_pri   = alt.Chart(pri_counts  ).mark_bar(size=40).encode(x="Prioridad:N", y="Cantidad:Q")
    st.altair_chart(chart_state | chart_pri, use_container_width=True)

    # -------------------- Exportar a Excel --------------------------------
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    st.download_button("⬇️ Exportar a Excel", buffer.getvalue(), file_name="tickets.xlsx", mime="application/vnd.ms-excel")


if __name__ == "__main__":
    main()
