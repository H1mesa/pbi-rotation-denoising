"""
auto_mpg.py : chargement et nettoyage du jeu de données réel "Auto MPG" (étape A).

Source : UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/9/auto+mpg
398 voitures (1970-1982). On s'en sert pour prédire la consommation `mpg`
(miles par gallon : plus c'est grand, moins la voiture consomme) à partir de `horsepower`.

Le fichier brut a des défauts typiques de données réelles :
- colonnes séparées par un nombre variable d'espaces, nom de voiture entre guillemets après une tabulation ;
- 6 puissances manquantes écrites "?" ;
- pas d'en-tête.
Le nettoyage est fait ici une fois pour toutes, et chaque étape est comptée dans un rapport.
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin", "car_name"]

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "auto-mpg.data"


def load_raw(path=DEFAULT_PATH):
    """Lit le fichier brut tel quel. Les "?" deviennent des NaN (valeurs manquantes)."""
    return pd.read_csv(path, sep=r"\s+", names=COLUMNS, na_values="?", quotechar='"')


def clean(df):
    """Nettoie le tableau et renvoie (df_propre, rapport).

    Étapes, dans l'ordre :
    1. supprimer les lignes où mpg ou horsepower manque (on ne peut pas les inventer) ;
    2. supprimer les doublons exacts ;
    3. garder seulement les valeurs physiquement plausibles (mpg > 0, horsepower > 0).
    Le rapport compte les lignes retirées à chaque étape : à mettre dans le compte rendu.
    """
    rapport = {"lignes_brutes": len(df)}

    df = df.dropna(subset=["mpg", "horsepower"])
    rapport["apres_valeurs_manquantes"] = len(df)

    df = df.drop_duplicates()
    rapport["apres_doublons"] = len(df)

    df = df[(df["mpg"] > 0) & (df["horsepower"] > 0)]
    rapport["apres_plausibilite"] = len(df)

    return df.reset_index(drop=True), rapport


def load_xy(path=DEFAULT_PATH):
    """Raccourci : renvoie x = horsepower (N,), y = mpg (N,) et le rapport de nettoyage."""
    df, rapport = clean(load_raw(path))
    return df["horsepower"].to_numpy(float), df["mpg"].to_numpy(float), rapport
