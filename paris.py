"""This file is meant to explore and visualize the key takeaways from the paris bike count database found
at the following link : 
https://www.data.gouv.fr/fr/datasets/comptage-velo-donnees-compteurs/"""

# Import section
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import folium
from folium.plugins import HeatMap

# Read data

df = pd.read_csv(
    "../data/comptage-velo-donnees-compteurs.csv",
    sep=";",
    usecols=[
        "id_compteur",
        "nom_compteur",
        "id",
        "name",
        "sum_counts",
        "date",
        "installation_date",
        "coordinates",
        "counter",
        "mois_annee_comptage",
    ],
    nrows =100
)

df["date"] = pd.to_datetime(df["date"])

df[["Latitude", "Longitude"]] = df.coordinates.str.split(",", expand=True).astype(
    "float"
)


def stats(df):

    print("The dataframe has the following columns", list(df.columns))

    # We wish to compute the average hourly bike count over our dataframe
    # date, sum counts.
    temp = df.groupby(["date"])["sum_counts"].sum()
    print("The average bike count per hour is {}.".format(temp.mean()))

    # We wish to plot the monthly bike count over time.
    temp = df.copy()
    temp = temp.groupby(pd.Grouper(key="date", freq="M"))["sum_counts"].sum()
    fig = temp.plot.line(x="date", y="sum_counts")
    fig.save("plot.png")

    # We wish to map the locations with the highest bike count on average.
    heatmap_df = df.groupby(["Latitude", "Longitude"])["sum_counts"].sum()
    locations = heatmap_df.index
    weights = heatmap_df
    map_obj = folium.Map(location=[48.864716, 2.349014], zoom_start=10)
    HeatMap(locations, weights).add_to(map_obj)
    map_obj.save("test.html")


stats(df)
