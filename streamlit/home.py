import streamlit as st

st.set_page_config(page_title="Hexabike", page_icon="🚴‍♀️", layout="wide")

LOGO_LATITIUDES = "data/Latitudes-logo-carre.png"
LOGO_DATAGOUV = "data/data-gouv.png"
WIDTH = 300

st.title("# Welcome to Hexabike! 👋")
st.write("# ")
st.markdown("#### Bienvenue dans ce projet en collaboration avec Latitudes")
cols = st.columns(3)
cols[-1].write("# ")
cols[-1].image(LOGO_DATAGOUV, width=WIDTH//2)
cols[0].image(LOGO_LATITIUDES, width=WIDTH)
st.info("(*) Outil fondé sur des données publiques : data.gouv.fr")
st.sidebar.success("Sélectionne une fonctionnalité ici !")