import streamlit as st
import pandas as pd
from pathlib import Path
import requests
import urllib
import math
import folium
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn.metrics as metrics
import json

ACCIDENT_VELO_DF = Path(__file__).parent.parent / "data" / "df_velos_metropoles.csv"

ITINERAIRE_API_URL = "https://wxs.ign.fr/calcul/geoportail/itineraire/rest/1.0.0/route?"

PROFILE = "pedestrian" # or "car"
RESOURCE = "bdtopo-osrm" # more precise, less options than "bdtopo-pgr"
OPTI = "shortest" # or "fastest"

PARIS_LAT_LONG = [48.864716, 2.349014]
ZOOM_START = 13

JOUR_MAPPING = {
    0: "Lundi",
    1: "Mardi",
    2: "Mercredi",
    3: "Jeudi",
    4: "Vendredi",
    5: "Samedi",
    6: "Dimanche"
}

GRAVITE_MAPPING = {
    '1 - Blessé léger': 1,
    '2 - Blessé hospitalisé': 2,
    'Blessé léger': 1,
    '0 - Indemne': 0,
    'Blessé hospitalisé ': 2,
    '3 - Tué': 4,
    'Indemne': 0,
    'Tué': 4,
    'Non renseigné': -1
}

def load_df():
    # Load dataframe
    df_accidents = pd.read_csv(ACCIDENT_VELO_DF)
    df_accidents.date = pd.to_datetime(df_accidents.date)
    df_accidents = df_accidents[df_accidents.departement.apply(lambda x: intify(x)) < 101]
    df_accidents = df_accidents[df_accidents.date.dt.year > 2010]
    return df_accidents

def intify(x):
    try:
        return int(x)
    except Exception as e:
        return 0


def get_filters(filters: list, view="dataviz"):
    if "year" in filters:
        if st.sidebar.checkbox("Filter by year"):
            st.session_state["year"] = st.sidebar.select_slider(
                "Choisis une année", 
                options=list(range(2011, 2022))
            )
        else:
            st.session_state["year"] = None
    if "region" in filters:
        st.session_state["region"] = st.sidebar.selectbox("Région", ["Paris", "Toute la métropole"])
    
    if "data_region" in filters:
        st.session_state["data_region"] = st.sidebar.selectbox("Région", ["Paris", "Toute la métropole"])


def load_df_accidents(df, year: int = None, region: str = None):
    new_df = df.copy()
    if year:
        new_df = new_df[new_df.date.dt.year == year]

    if region == "Paris":
        new_df = new_df[new_df.departement.apply(lambda x: intify(x)) == 75]

    return new_df


def get_location(address: str):
    if not address:
        return None
    url = f'https://nominatim.openstreetmap.org/search/{urllib.parse.quote(address)}?format=json'
    response = requests.get(url).json()
    return response[0]["lat"], response[0]["lon"]

def stringify_loc(location):
    return f"{location[0]},{location[1]}"

def get_path_from_addresses(start, end, profile: str = PROFILE, resource: str = RESOURCE, opti: str = OPTI):
    if isinstance(start, str) & isinstance(end, str):
        start = get_location(start)
        end = get_location(end)
    
    start = start[1], start[0]
    end = end[1], end[0]
    
    query = {
        "start": stringify_loc(start),
        "end": stringify_loc(end),
        "resource": resource, 
        "profile": profile,
        "optimization": opti
    }
    response_dict = requests.get(ITINERAIRE_API_URL, params=query).json()
    path = [(y, x) for x, y in response_dict["geometry"]["coordinates"]]
    true_length = response_dict["distance"]
    return path, true_length


def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)

def pnt2line(pnt, start, end):
    """
    Calcule la distance entre un points et un segment
    -> renvoie la distance min et le point le plus proche du segment
    """
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def distance_in_meters_between_2_points(x, y):
    # points : lat, lon
    R = 6371e3 # metres
    phi1 = x[0] * math.pi/180 # φ, λ in radians
    phi2 = y[0] * math.pi/180
    D_phi = (y[0]-x[0]) * math.pi/180
    D_lambda = (y[1]-x[1]) * math.pi/180

    a = math.sin(D_phi/2) * math.sin(D_phi/2) + math.cos(phi1) * math.cos(phi2) * math.sin(D_lambda/2) * math.sin(D_lambda/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def calculate_distance_from_path(point: tuple, polyline: list):
    dist_by_segment = [
        pnt2line(point, polyline[i], polyline[i+1]) 
        for i in range(len(polyline)-1) 
        if polyline[i] != polyline[i+1]
    ]
    _, nearest_point = min(dist_by_segment, key=lambda x: x[0], default=(0, 0))
    return nearest_point, distance_in_meters_between_2_points(point, nearest_point)


def show_accident_and_line(accident_point: tuple, polyline: list, start: tuple, end: tuple, nearest: tuple):
    """
    Returns a map containing
    - a polyline corresponding to an itinerary
    - the shortest line enabling to calculate distance between a given point and the line (by the nearest point from the line)
    """
    map_paris = folium.Map(location=PARIS_LAT_LONG, zoom_start=ZOOM_START)
    folium.PolyLine(polyline, color="red", weight=2.5, opacity=1).add_to(map_paris)
    folium.Marker(start, popup="start").add_to(map_paris)
    folium.Marker(end, popup="end").add_to(map_paris)
    folium.Marker(accident_point, popup="accident")
    folium.PolyLine([accident_point, nearest], color="blue", weight=2.5, opacity=1).add_to(map_paris)
    return map_paris


def get_close_accidents(polyline: list, df_accidents: pd.DataFrame):
    df_results = df_accidents[["lat", "lon", "gravite accident", "date"]]
    for idx, row in df_results.iterrows():
        accident_loc = row.lat, row.lon
        nearest, dist = calculate_distance_from_path(accident_loc, polyline)
        df_results.loc[idx, "distance_in_m"] = distance_in_meters_between_2_points(accident_loc, nearest)
        
    return df_results


def show_near_accidents(polyline: list, start: tuple, end: tuple, df_distances: pd.DataFrame):
    map_paris = folium.Map(location=PARIS_LAT_LONG, zoom_start=ZOOM_START)
    folium.PolyLine(polyline, color="red", weight=2.5, opacity=1).add_to(map_paris)
    folium.Marker(start, popup="start").add_to(map_paris)
    folium.Marker(end, popup="end").add_to(map_paris)
    for _, row in df_distances.iterrows():
        accident_point = row.lat, row.lon
        folium.Marker(accident_point, popup=f"Accident à une distance de {round(row.distance_in_m, 2)} mètres le {row.date}.\n {row['gravite accident']}").add_to(map_paris)
    return map_paris


def get_itineraire_map(start: str, end: str, df_velos: pd.DataFrame):
    """Returns a map with itinerary and close past accidents given a start and an end"""
    start_map = get_location(start)
    end_map = get_location(end)
    line, true_length = get_path_from_addresses(start_map, end_map)

    distances = get_close_accidents(line, df_velos)
    near_accidents = distances[distances["distance_in_m"] < 50]
    return true_length, len(near_accidents), show_near_accidents(line, start_map, end_map, near_accidents)



### Partie ML


template = json.load(open("data/template.json", "r"))["template"]

model_df = pd.read_csv("data/data_model.csv", index_col=0)
x_train, x_test, y_train, y_test = train_test_split(
    model_df.drop(columns="gravite_accident"), 
                  model_df["gravite_accident"], 
                           test_size=0.2
                           )


def Decision_Tree(max_depth, min_samples_leaf, min_samples_split):
    """returns RMSE and R2 for decision tree"""
    regressor = tree.DecisionTreeRegressor(
    max_depth = max_depth,
    min_samples_leaf = min_samples_leaf,
    min_samples_split = min_samples_split)
    regressor = regressor.fit(x_train, y_train)
    return regressor

def Random_Forest( max_depth, max_features, n_estimators):
    """returns RMSE and R2 for random forest"""
    regressor = RandomForestRegressor(bootstrap = True,
                max_depth = max_depth,
                max_features = max_features,
                n_estimators = n_estimators)
    regressor = regressor.fit(x_train, y_train)
    return regressor


param_grid = {
        'max_depth': [44, 45, 46, 47, 48, 49],
        'n_estimators': [850, 875, 900, 925]
    }

def check(categ: str, value):
    one_hot_encoder=[]
    for k in range(len(template)):
        if template[k][:len(categ)]==categ:
            if template[k][len(categ):]==value:
                one_hot_encoder.append(1)
            else:
                one_hot_encoder.append(0)
    return one_hot_encoder


def one_hot_encode(entry):
    L = [entry[0], entry[4], entry[7]]
    L+=check("luminosite_",entry[1])
    L+=check("type_route_",entry[3])
    L+=check("conditions_atmosperiques_",entry[2])
    L+=check("categorie_usager_",entry[5])
    L+=check("sexe_",entry[6])
    L+=check("arrondissement_",entry[9])
    L+=check("piste_cyclable_",entry[8])

    return L

def prediction(features:list, regressor_type: str = "Decision Tree"):

    if regressor_type == "Decision Tree":
        regressor = Decision_Tree(max_depth = 3, min_samples_leaf = 4, min_samples_split = 2)
    else:
        regressor = Random_Forest(max_depth = 48, max_features = 'sqrt', n_estimators = 900)

    x_val = pd.DataFrame(columns=template)
    x_val.loc[1] = one_hot_encode(features)

    scaler = StandardScaler()
    to_scale = ['jour', 'nb_voies', 'age']
    for col in to_scale:
       x_val[col] = scaler.fit_transform(np.array(x_val[col]).reshape(-1,1))

    y_pred = regressor.predict(x_val)[0]
    print(
        f"La gravité prédite de l'accident, allant de 0 (Blessé léger) à 3 (Tué) est de {y_pred}"
    )
    return y_pred