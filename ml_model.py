# ml_model.py  —  VERSION 2.0
# Entraîne et utilise un modèle Random Forest pour prédire
# la cardinalité Merise à partir de caractéristiques numériques
# et de scores sémantiques associés au nom de relation.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split,
                                      cross_val_score,
                                      StratifiedKFold)
from sklearn.metrics import (classification_report,
                               accuracy_score,
                               confusion_matrix)
import joblib

MAPPING  = {0: '1:1', 1: '1:N', 2: 'N:M'}
FEATURES = ['nb_attr_e1', 'nb_attr_e2', 'ratio_attrs', 'max_attrs',
            'total_attrs', 'longueur_nom', 'score_1_1', 'score_1_N',
            'score_N_M']

# Vocabulaire partagé avec dataset_generator.py.
# Les mêmes scores sémantiques sont nécessaires pour produire
# des features de prédiction cohérentes.
VOCABULAIRE = {
    "dirige":              (0.90, 0.15, 0.05),
    "est_marie_a":         (0.95, 0.05, 0.05),
    "correspond":          (0.85, 0.20, 0.10),
    "est_assigne_a":       (0.80, 0.25, 0.10),
    "possede_unique":      (0.85, 0.20, 0.05),
    "represente":          (0.75, 0.30, 0.15),
    "est_dirige_par":      (0.82, 0.22, 0.08),
    "a_pour_chef":         (0.88, 0.18, 0.06),
    "commande":            (0.05, 0.90, 0.20),
    "passe":               (0.10, 0.85, 0.20),
    "effectue":            (0.10, 0.85, 0.15),
    "reserve":             (0.10, 0.80, 0.25),
    "emploie":             (0.15, 0.85, 0.15),
    "supervise":           (0.30, 0.75, 0.15),
    "envoie":              (0.05, 0.80, 0.25),
    "fabrique":            (0.10, 0.75, 0.30),
    "livre":               (0.10, 0.78, 0.28),
    "appartient":          (0.10, 0.82, 0.25),
    "participe":           (0.05, 0.20, 0.90),
    "inscrit":             (0.10, 0.25, 0.85),
    "emprunte":            (0.05, 0.20, 0.88),
    "redige":              (0.05, 0.25, 0.85),
    "enseigne":            (0.08, 0.25, 0.86),
    "consulte":            (0.10, 0.25, 0.86),
    "collabore":           (0.10, 0.15, 0.90),
    "echange":             (0.10, 0.20, 0.85),
    "negocie":             (0.10, 0.20, 0.80),
    "partage":             (0.15, 0.30, 0.75),
    "co_gere":             (0.25, 0.20, 0.75),
    "co_produit":          (0.05, 0.15, 0.90),
    "gere":                (0.60, 0.55, 0.20),
    "possede":             (0.40, 0.65, 0.25),
    "utilise":             (0.15, 0.55, 0.60),
    "contient":            (0.10, 0.60, 0.55),
    "associe":             (0.30, 0.45, 0.55),
    "concerne":            (0.20, 0.50, 0.55),
    "traite":              (0.25, 0.55, 0.50),
    "relie":               (0.20, 0.45, 0.60),
    "implique":            (0.20, 0.45, 0.60),
    "inclut":              (0.10, 0.60, 0.50),
    "achete":              (0.05, 0.45, 0.70),
    "produit":             (0.15, 0.65, 0.45),
    "couvre":              (0.15, 0.55, 0.55),
}


def extraire_features(nb_attr_e1, nb_attr_e2, nom_association):
    """Construit les caractéristiques d'entrée utilisées par le modèle."""
    nom    = nom_association.lower().strip()
    scores = VOCABULAIRE.get(nom, (0.25, 0.35, 0.45))
    return {
        'nb_attr_e1':    nb_attr_e1,
        'nb_attr_e2':    nb_attr_e2,
        'ratio_attrs':   round(nb_attr_e1 / max(nb_attr_e2, 1), 3),
        'max_attrs':     max(nb_attr_e1, nb_attr_e2),
        'total_attrs':   nb_attr_e1 + nb_attr_e2,
        'longueur_nom':  len(nom),
        'score_1_1':     scores[0],
        'score_1_N':     scores[1],
        'score_N_M':     scores[2],
    }


def entrainer_modele(chemin_csv='dataset_mcd.csv'):

    try:
        df = pd.read_csv(chemin_csv, encoding='utf-8', low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError):
        try:
            df = pd.read_csv(chemin_csv, encoding='latin-1', low_memory=False)
        except Exception:
            # Fallback: utiliser l'interpréteur Python du parser et ignorer les lignes problématiques
            df = pd.read_csv(chemin_csv, engine='python', encoding='utf-8', on_bad_lines='skip')
    print(f"Dataset charge : {len(df)} exemples\n - ml_model.py:89")

    X = df[FEATURES]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train)}  |  Test : {len(X_test)}\n - ml_model.py:97")

    modele = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    modele.fit(X_train, y_train)
    print("Entrainement termine.\n - ml_model.py:108")

    # Évaluation du modèle sur les données de test et validation croisée
    y_pred = modele.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(modele, X, y, cv=cv, scoring='accuracy')

    print(f"Accuracy Test       : {acc * 100:.2f}% - ml_model.py:117")
    print(f"CrossVal 5Fold    : {scores.mean()*100:.2f}% - ml_model.py:118"
          f"  (+/- {scores.std()*100:.2f}%)\n")

    print("Rapport de classification : - ml_model.py:121")
    print(classification_report(y_test, y_pred,
                                 target_names=['1:1', '1:N', 'N:M']))

    print("Matrice de confusion : - ml_model.py:125")
    cm = confusion_matrix(y_test, y_pred)
    print(f"1:1   1:N   N:M - ml_model.py:127")
    for i, row in enumerate(cm):
        print(f"Reel {MAPPING[i]} : {row} - ml_model.py:129")

    # Affiche l'importance relative de chaque variable utilisée par le modèle
    print("\nImportance des features : - ml_model.py:132")
    imp_sorted = sorted(zip(FEATURES, modele.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    for feat, imp in imp_sorted:
        barre = '|' * int(imp * 50)
        print(f"{feat:18s}  {imp:.3f}  {barre} - ml_model.py:137")

    joblib.dump(modele, 'modele_rf.pkl')
    print("\nModele sauvegarde : modele_rf.pkl - ml_model.py:140")
    return modele


def charger_modele(chemin='modele_rf.pkl'):
    return joblib.load(chemin)


def predire_cardinalite(modele, nb_attr_e1, nb_attr_e2, nom_association):
    # Prépare les features avant d’interroger le modèle entraîné
    features = pd.DataFrame([
        extraire_features(nb_attr_e1, nb_attr_e2, nom_association)
    ])[FEATURES]
    prediction   = modele.predict(features)[0]
    probabilites = modele.predict_proba(features)[0]
    confiance    = max(probabilites) * 100
    label        = MAPPING[prediction]
    print(f"Association '{nom_association}' > {label} - ml_model.py:157"
          f"  (confiance : {confiance:.1f}%)")
    return prediction, label, confiance


if __name__ == "__main__":
    from dataset_generator import generer_dataset
    generer_dataset(3000)
    modele = entrainer_modele()

    print("\n Demonstrations - ml_model.py:167")
    predire_cardinalite(modele, 4, 3, "commande")   # fort 1:N
    predire_cardinalite(modele, 5, 6, "participe")  # fort N:M
    predire_cardinalite(modele, 3, 2, "dirige")     # fort 1:1
    predire_cardinalite(modele, 6, 5, "utilise")    # ambigu
    predire_cardinalite(modele, 4, 4, "gere")       # ambigu
    predire_cardinalite(modele, 8, 8, "associe")    # très ambigu
