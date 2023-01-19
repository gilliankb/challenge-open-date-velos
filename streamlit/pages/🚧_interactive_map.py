import streamlit as st
import pandas as pd
from pathlib import Path
# import pydeck as pdk
from utils import  get_filters, load_df, load_df_accidents
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static


ACCIDENT_VELO_DF = Path(__file__).parent.parent.parent / "data" / "df_velos_metropoles.csv"
PARIS_LAT_LONG = [48.864716, 2.349014]
ZOOM_FRANCE = 4
ZOOM_PARIS = 11

def intify(x):
    try:
        return int(x)
    except Exception as e:
        return 0


def load_paris_map(df: pd.DataFrame):
    region = st.session_state.get("region", None)

    if region == "Paris":
        st.markdown("### Map - 🗼 Paris")
        zoom = ZOOM_PARIS

    else:
        st.markdown("### Map - 🇫🇷 France")
        zoom = ZOOM_FRANCE

    st.text("Visualisation Basique")
    df_to_display = df[df.lat > 0]
    return st.map(df_to_display[["lat", "lon"]], zoom=zoom)


# def display_hexagon_pydeck_map(df: pd.DataFrame):
#     chart_data = df[["lat", "lon"]]
#     initial_lat, initial_lon = chart_data.iloc[-1].lat, chart_data.iloc[-1].lon
#     st.pydeck_chart(pdk.Deck(
#         map_style=None,
#         initial_view_state=pdk.ViewState(
#             latitude=initial_lat,
#             longitude=-initial_lon,
#             zoom=11,
#             pitch=20,
#         ),
#         layers=[
#             pdk.Layer(
#             'HexagonLayer',
#             data=chart_data,
#             get_position='[lon, lat]',
#             radius=220,
#             elevation_scale=10,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#             ),
#             pdk.Layer(
#                 'ScatterplotLayer',
#                 data=chart_data,
#                 get_position='[lon, lat]',
#                 get_color='[200, 30, 0, 160, 20, 50]',
#                 get_radius=200,
#             ),
#         ],
#     ))

def display_heatmap(df):
    st.text("Heatmap des accidents")
    locations = df[["lat", "lon"]]
    weights = [1 for _ in locations]
    map_obj = folium.Map(location=[48.864716, 2.349014], zoom_start=10)
    HeatMap(locations, weights).add_to(map_obj)
    return folium_static(map_obj, width=1200)

def main():
    st.title("🚧 Visualisation des accidents de vélo par an")
    st.write("Voici une carte interactive, n'hésitez pas à jouer avec les filtres à votre gauche !")

    df_accidents = load_df()

    get_filters(["year", "region"])
    df = load_df_accidents(
        df_accidents, 
        year= st.session_state.get("year"),
        region=st.session_state.get("region")
    )

    load_paris_map(df)

    # display_hexagon_pydeck_map(df)

    display_heatmap(df)

    # display_heatmap(df)

if __name__ == '__main__':
    main()