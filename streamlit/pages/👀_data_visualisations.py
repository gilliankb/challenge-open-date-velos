import streamlit as st
import pandas as pd
from pathlib import Path
from utils import load_df, get_filters, load_df_accidents, JOUR_MAPPING, GRAVITE_MAPPING
import plotly.express as px


ACCIDENT_VELO_DF = Path(__file__).parent.parent.parent / "data" / "df_velos_metropoles.csv"

def show_accidents_per_year(df: pd.DataFrame):
    st.markdown("#### 🥳 Accidents par année")
    grouped_per_year = df.groupby(df.date.dt.year).agg({"commune": "count"}).reset_index()
    grouped_per_year.columns = ["date", "nb_accidents"]
    fig = px.line(grouped_per_year, x="date", y="nb_accidents", title='Accidents par année')
    st.plotly_chart(fig, use_container_width=True)


def show_accidents_per_week(df_velos: pd.DataFrame):
    st.markdown("#### 📆 Accidents par semaine")
    df = df_velos.copy()
    if st.session_state.get("data_region", None):
        df = df[df.departement == '75']

    df["nb_accidents"] = 1
    df['Date'] = pd.to_datetime(df['date']) - pd.to_timedelta(7, unit='d')
    df = (df.groupby([pd.Grouper(key='Date', freq='W-MON')])["nb_accidents"]
        .sum()
        .reset_index()
        .sort_values('Date')
        )
    fig = px.line(df, x="Date", y="nb_accidents", title='Accidents par semaine sur la période')
    st.plotly_chart(fig, use_container_width=True)
    

def show_accident_by_weekday(df_velos: pd.DataFrame):
    df = df_velos.copy()
    df["Jour"] = df.date.dt.weekday
    df["gravite accident"] = df["gravite accident"].apply(lambda x: GRAVITE_MAPPING.get(x, None))
    df["count"] = 1
    df = df.groupby(["Jour", "gravite accident"]).agg({"count": "count"}).sort_index().reset_index()
    df.Jour = df.Jour.apply(lambda x: JOUR_MAPPING[x])
    fig = px.bar(df, x="Jour", y="count", color="gravite accident", title="Accidents par jour de la semaine")
    st.plotly_chart(fig, use_container_width=True)



def main():
    st.title("👀 Visualisation des données")
    df_accidents = load_df()
    get_filters(["year", "data_region"])
    
    df = load_df_accidents(
        df_accidents,
        year=st.session_state.get("year", None),
        region=st.session_state.get("data_region", None)
    )
    if not st.session_state["year"]:
        show_accidents_per_year(df)
    show_accidents_per_week(df)

    show_accident_by_weekday(df)


if __name__ == '__main__':
    main()