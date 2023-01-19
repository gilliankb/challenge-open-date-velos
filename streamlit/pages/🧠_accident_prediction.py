import streamlit as st
from pathlib import Path
import json
from utils import JOUR_MAPPING, prediction

FEATURE_ACCIDENT_PATH = Path(__file__).parent.parent.parent / "data" / "model_feature_choices.json"
features_choices = json.load(open(FEATURE_ACCIDENT_PATH, "r"))

def get_features():
    expander = st.expander("Formulaire")
    with st.form("my_form"):
        jour = expander.select_slider("Jour", options=features_choices["jour"], format_func=lambda x: JOUR_MAPPING[x])
        luminosite = expander.selectbox("Luminosité", options=features_choices["luminosite"])
        conditions_atmospheriques = expander.selectbox("Conditions atmosphériques", options=features_choices["conditions_atmosperiques"])
        arrondissement = expander.selectbox("Arrondissement", options=list(range(1, 21)))

        type_route = expander.selectbox("Type de route", options=features_choices["type_route"])
        nb_voies = expander.slider("Nombre de voies", min_value=1, max_value=8)
        piste_cyclable = expander.radio("Sur piste cyclable", ["True", "False"])
    
        sexe = expander.radio("Sexe", ["F", "M"])
        age = expander.slider("Age", min_value=14, max_value=89)
        categorie_usager = expander.selectbox("Usager secondaire", options=features_choices["categorie_usager"])

        model_type = expander.radio("Type de modèle", ["Decision Tree", "Random Forest"])
        submitted = st.form_submit_button("Prédiction")
        if submitted:
            st.success("Formulaire validé")
            return [
                jour,
                luminosite,
                conditions_atmospheriques,
                type_route,
                nb_voies,
                categorie_usager,
                sexe,
                age,
                piste_cyclable,
                arrondissement,
            ], model_type
        return None, None

def predict_result(features: list, model_type: str = "Decision Tree"):
    result = prediction(features, model_type)
    st.write(f"""
    La prédiction de gravité d'un accident dans les circonstances précisées est de :\n
    """
    )
    st.metric(label="Gravité estimée", value=round(result, 2))
    st.info(
        """
        0 - Indemne\n
        1 - Blessé léger \n
        2 - Blessé hospitalisé \n
        3 - Tué
        """
    )

def main():
    st.title("🧠 Prédiction de la gravité d'un accident")
    st.subheader("D'après la localisation et le profil")
    features, model_type = get_features()
    if features:
        predict_result(features, model_type)

if __name__ == '__main__':
    main()