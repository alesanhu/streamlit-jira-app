import os
from io import BytesIO
from datetime import datetime

import pandas as pd
import altair as alt
import streamlit as st
from jira import JIRA
from openai import OpenAI  # Cliente OpenAI actualizado

# --- Helpers --------------------------------------------------------------
server = st.secrets.get("JIRA_SERVER")
user   = st.secrets.get("JIRA_USER")
token  = st.secrets.get("JIRA_TOKEN")
st.sidebar.write("DEBUG: servidor=", server, "usuario=", user)

@st.cache_resource
def create_jira_client(server: str, user: str, token: str):
    """Crea un cliente JIRA"""
    try:
        return JIRA(server=server, basic_auth=(user, token))
    except Exception as e:
        st.error(f"Error conexión Jira: {e}")
        return None

@st.cache_data
def fetch_tickets(_jira, jql: str):
    """Recupera issues desde JIRA usando JQL"""
    try:
        return _jira.search_issues(jql, maxResults=2000)
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")
        return []


def quote_list(vals: list) -> str:
    """Escapa y cita valores para usar en JQL"""
    escaped = [v.replace("'", "\\'") for v in vals]
    return ",".join(f"'{v}'" for v in escaped)


def show_metrics(df: pd.DataFrame, resolved_df: pd.DataFrame):
    """Muestra los indicadores clave de rendimiento"""
    now = pd.Timestamp.now()
    total = len(df)
    resolved_count = len(resolved_df)
    avg_res = resolved_df['resolve_days'].mean().round(1) if resolved_count else 0
    avg_open_age = (
        now - df.loc[df['resolved'].isna(), 'created']
    ).dt.days.mean().round(1) if df['resolved'].isna().any() else 0
    min_res = int(resolved_df['resolve_days'].min()) if resolved_count else 0
    max_res = int(resolved_df['resolve_days'].max()) if resolved_count else 0
    sla7 = (resolved_df['resolve_days'] <= 7).mean() * 100 if resolved_count else 0

    cols = st.columns(5)
    cols[0].metric("Total tickets", total)
    cols[1].metric("Resueltos", resolved_count)
    cols[2].metric("Promedio días resolución", avg_res)
    cols[3].metric("Promedio días abiertos", avg_open_age)
    cols[4].metric("% SLA ≤7d", f"{sla7:.1f}%")

    c2 = st.columns(2)
    c2[0].metric("Mín. días resolución", min_res)
    c2[1].metric("Máx. días resolución", max_res)


def summarize_tickets_text(df: pd.DataFrame) -> str:
    """
    Genera un resumen y puntos de alerta usando OpenAI GPT basados en descripción y comentarios.
    Usa las credenciales definidas en st.secrets.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "No se encontró OPENAI_API_KEY. Configure su clave en Secrets."

    api = OpenAI(api_key=api_key)
    texts = []
    for desc in df.get('fields.description', []):
        if pd.notna(desc):
            texts.append(desc)
    combined = ' \n---\n '.join(texts[:10]) or 'No hay descripciones disponibles.'
    prompt = (
        "Eres un asistente que resume tickets de Jira. "
        "El siguiente texto son las descripciones de tickets filtrados; "
        "Proporciona un resumen breve y puntos de alerta clave:\n" + combined
    )
    try:
        resp = api.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.5
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar resumen: {e}"

# --- App ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Gestión de Tickets en Jira con Resumen GPT", layout="wide")
    st.title("📊 Gestión de Tickets en Jira con Resumen GPT")

    # -- Conexión Jira ------------------------------------------------------
    st.sidebar.header("🔐 Conexión Jira")
    server = st.secrets.get("JIRA_SERVER")
    user   = st.secrets.get("JIRA_USER")
    token  = st.secrets.get("JIRA_TOKEN")
    if not (server and user and token):
        st.sidebar.warning("Configure JIRA_SERVER, JIRA_USER y JIRA_TOKEN en Secrets.")
        return

    jira = create_jira_client(server, user, token)
    if not jira:
        return

    # -- Filtros ------------------------------------------------------------
    st.sidebar.header("🔍 Filtros")
    projects = []
try:
    projects = [p.key for p in jira.projects() if not p.raw.get('archived', False)]
except Exception as e:
    st.sidebar.error(f"No se pudieron cargar proyectos: {e}")




    sel_proj = st.sidebar.multiselect("Proyectos", options=projects, default=projects, key="proj_filter")

   statuses = []
try:
    statuses = [s.name.strip() for s in jira.statuses()]
except Exception as e:
    st.sidebar.error(f"No se pudieron cargar estados: {e}")
    sel_status = st.sidebar.multiselect("Estados", options=statuses, default=statuses, key="status_filter")

    priorities = []
try:
    priorities = [p.name.strip() for p in jira.priorities()]
except Exception as e:
    st.sidebar.error(f"No se pudieron cargar prioridades: {e}")

    sel_pri = st.sidebar.multiselect("Prioridades", options=priorities, default=priorities, key="pri_filter")

    today = datetime.utcnow().date()
    default_start = today - pd.Timedelta(days=30)
    start_date, end_date = st.sidebar.date_input(
        "Rango fechas creación", value=(default_start, today), key="date_filter"
    )

    # -- JQL y carga --------------------------------------------------------
    jql_parts = []
    if sel_proj:
        jql_parts.append(f"project in ({quote_list(sel_proj)})")
    jql_parts.append(f"created >= '{start_date}' AND created <= '{end_date}'")
    jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

    with st.spinner("Cargando tickets..."):
        issues = fetch_tickets(jira, jql)
    if not issues:
        st.warning("No hay tickets para los filtros seleccionados.")
        return

    # -- DataFrame ----------------------------------------------------------
    df = pd.json_normalize([i.raw for i in issues])
    df['assignee']     = df['fields.assignee.displayName'].fillna('Sin asignar')
    df['reporter']     = df['fields.reporter.displayName'].fillna('Sin asignar')
    df['status']       = df['fields.status.name']
    df['priority']     = df['fields.priority.name'].fillna('None')
    df['created']      = pd.to_datetime(df['fields.created'], utc=True).dt.tz_localize(None)
    df['resolved']     = pd.to_datetime(df['fields.resolutiondate'], utc=True).dt.tz_localize(None)
    df['area_destino'] = df.get('fields.customfield_10043.value', 'Sin Área')

    # -- Filtros locales ----------------------------------------------------
    df = df[df['status'].isin(sel_status)]
    df = df[df['priority'].isin(sel_pri)]

    sel_assignee = st.sidebar.multiselect("Responsable", df['assignee'].unique(), df['assignee'].unique(), key="assignee_sel")
    df = df[df['assignee'].isin(sel_assignee)]

    sel_reporter = st.sidebar.multiselect("Reportero", df['reporter'].unique(), df['reporter'].unique(), key="reporter_sel")
    df = df[df['reporter'].isin(sel_reporter)]

    sel_area = st.sidebar.multiselect("Área Destino", df['area_destino'].unique(), df['area_destino'].unique(), key="area_sel")
    df = df[df['area_destino'].isin(sel_area)]

    # -- Métricas de tiempo --------------------------------------------------
    now = pd.Timestamp.now()
    df['age_days']       = (now - df['created']).dt.days
    resolved_df          = df.dropna(subset=['resolved']).copy()
    resolved_df['resolve_days'] = (resolved_df['resolved'] - resolved_df['created']).dt.days

    # -- GPT Summary ---------------------------------------------------------
    st.subheader("📝 Resumen de Tickets Generado por GPT")
    summary = summarize_tickets_text(df)
    st.text_area("Resumen y Alertas", value=summary, height=200)

    # -- KPI tabla ----------------------------------------------------------
    show_metrics(df, resolved_df)

    # -- Gráficos ----------------------------------------------------------
    st.subheader("📊 Tickets por Estado y Prioridad")
    sc = df['status'].value_counts().reset_index(); sc.columns=['status','count']
    pc = df['priority'].value_counts().reset_index(); pc.columns=['priority','count']

    chart_status = alt.Chart(sc).mark_bar(size=40).encode(
        x=alt.X('status:N', title='Estado', sort='-y'),
        y=alt.Y('count:Q', title='Cantidad'),
        tooltip=['status','count']
    ).properties(width=600, height=400)

    chart_pri = alt.Chart(pc).mark_bar(size=40).encode(
        x=alt.X('priority:N', title='Prioridad', sort='-y'),
        y=alt.Y('count:Q', title='Cantidad'),
        tooltip=['priority','count']
    ).properties(width=600, height=400)

    combined = alt.hconcat(chart_status, chart_pri, spacing=60).resolve_scale(y='independent')
    st.altair_chart(combined, use_container_width=True)

    st.subheader("📈 Tendencia diaria (por día)")
    trend = df.copy(); trend['fecha'] = trend['created'].dt.floor('D')
    tc = trend.groupby('fecha').size().reset_index(name='count')
    chart_trend = alt.Chart(tc).mark_line(point=True).encode(
        x=alt.X('fecha:T', title='Fecha', axis=alt.Axis(format='%Y-%m-%d', tickCount='day')),
        y=alt.Y('count:Q', title='Tickets creados'),
        tooltip=[alt.Tooltip('fecha:T', title='Fecha'),'count']
    ).properties(width=1200, height=400)
    st.altair_chart(chart_trend, use_container_width=True)

    # -- Tablas -------------------------------------------------------------
    st.subheader("📋 Tickets por Responsable")
    ta = df.groupby('assignee').size().reset_index(name='tickets')
    st.dataframe(ta.sort_values('tickets', ascending=False), use_container_width=True)

    st.subheader("🏷️ Top 3 Tickets con Mayor Tiempo por Estado")
    tops = []
    for stt in df['status'].unique():
        tmp = df[df['status']==stt].copy()
        tmp['url'] = tmp['key'].apply(lambda k: f"{server}/browse/{k}")
        tops.append(tmp.nlargest(3, 'age_days')[['status','key','url','age_days']])
    st.dataframe(pd.concat(tops), use_container_width=True)

    # -- Export -------------------------------------------------------------
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Tickets", index=False)
    st.download_button("⬇️ Exportar a Excel", buffer.getvalue(), file_name="tickets_jira.xlsx")

if __name__ == '__main__':
    main()
