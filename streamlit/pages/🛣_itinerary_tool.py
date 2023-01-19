import streamlit as st
import pandas as pd
from pathlib import Path
import pydeck as pdk
import numpy as np
from utils import load_df, get_itineraire_map
from streamlit_folium import st_folium, folium_static


def get_start_end():
    st.write("Veuillez entrer des adresses valides à Paris")
    with st.form("itinerary"):
        start = st.text_input("Départ")
        end = st.text_input("Arrivée")
        if st.form_submit_button("Valider"):
            return start, end
    return None, None

def display_accidents_map(start, end, df):
    st.markdown("#### Itinéraire et accidents qui ont eu lieu à moins de 50 m")
    st.text('💡 Cliquez sur un pin pour plus de détails !')
    true_length, n_accidents, itinerary_map = get_itineraire_map(start, end, df)

    accidents_per_km_per_year = (n_accidents / true_length) * 1000 / (2021 - 2010)

    folium_static(itinerary_map, width=1200)

    cols = st.columns(3)
    cols[0].metric("Longueur du trajet (km)", value=round(true_length/1000, 2))
    cols[1].metric("Total accidents", value=n_accidents)
    cols[2].metric("Accidents par km par an", value=round(accidents_per_km_per_year, 2))


def main():
    st.title("🛣 Estimate la dangerosité de ton itinéraire !")
    df_accidents = load_df()
    df_accidents_paris = df_accidents[df_accidents.departement == "75"]
    start, end = get_start_end()
    if start:
        display_accidents_map(start, end, df_accidents_paris)

if __name__ == '__main__':
    main()