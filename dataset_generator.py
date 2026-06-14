# Générateur de jeu de données synthétique pour classification Merise
# Les résultats sont réutilisables sans fuite de données réelles.
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# Vocabulaire des relations avec probabilité d'appartenance à chaque type
# Les scores sont utilisés comme attributs sémantiques continus
# Ils ne sont pas interprétés comme des classes exclusives.

VOCABULAIRE = {
    # ── 1:1 forts ──────────────────────────────────────────────────────
    "dirige":              (0.90, 0.15, 0.05),
    "est_marie_a":         (0.95, 0.05, 0.05),
    "correspond":          (0.85, 0.20, 0.10),
    "est_assigne_a":       (0.80, 0.25, 0.10),
    "possede_unique":      (0.85, 0.20, 0.05),
    "represente":          (0.75, 0.30, 0.15),
    "est_dirige_par":      (0.82, 0.22, 0.08),
    "a_pour_chef":         (0.88, 0.18, 0.06),

    # ── 1:N forts ──────────────────────────────────────────────────────
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

    # ── N:M forts ──────────────────────────────────────────────────────
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

    # Relations ambiguës qui ne correspondent pas à un seul type clair
    # Le modèle doit s’appuyer sur les nombres d’attributs et le contexte.
    "gere":                (0.60, 0.55, 0.20),  # possible 1:1 ou 1:N
    "possede":             (0.40, 0.65, 0.25),  # généralement 1:N
    "utilise":             (0.15, 0.55, 0.60),  # 1:N ou N:M
    "contient":            (0.10, 0.60, 0.55),  # 1:N ou N:M
    "associe":             (0.30, 0.45, 0.55),  # relation très ambiguë
    "concerne":            (0.20, 0.50, 0.55),  # relation ambiguë
    "traite":              (0.25, 0.55, 0.50),  # relation ambiguë
    "relie":               (0.20, 0.45, 0.60),  # relation ambiguë
    "implique":            (0.20, 0.45, 0.60),  # relation ambiguë
    "inclut":              (0.10, 0.60, 0.50),  # relation ambiguë
    "achete":              (0.05, 0.45, 0.70),  # plutôt N:M
    "produit":             (0.15, 0.65, 0.45),  # plutôt 1:N
    "couvre":              (0.15, 0.55, 0.55),  # relation ambiguë
}

# Séparation des mots par type dominant
MOTS_FORTS = {
    0: [k for k, v in VOCABULAIRE.items() if v[0] >= 0.75],
    1: [k for k, v in VOCABULAIRE.items() if v[1] >= 0.75],
    2: [k for k, v in VOCABULAIRE.items() if v[2] >= 0.75],
}
MOTS_AMBIGUS = [
    k for k, v in VOCABULAIRE.items()
    if max(v) < 0.75 or sorted(v, reverse=True)[1] >= 0.40
]


def _bruit(val, sigma=0.9):
    """Applique un bruit normal et force la valeur à rester positive."""
    return max(1, int(round(val + np.random.normal(0, sigma))))


def generer_exemple(type_relation, taux_ambiguite=0.20):
    """Crée un exemple de relation avec attributs numériques et scores sémantiques."""
    # Choix du mot de relation, en privilégiant les mots dominants
    if random.random() < taux_ambiguite and MOTS_AMBIGUS:
        mot = random.choice(MOTS_AMBIGUS)
    else:
        candidats = MOTS_FORTS.get(type_relation, list(VOCABULAIRE.keys()))
        mot = random.choice(candidats)

    scores = VOCABULAIRE.get(mot, (0.33, 0.34, 0.33))

    # Simule la taille des deux entités selon le type de relation
    if type_relation == 0:       # 1:1 : tailles équilibrées entre les deux entités
        nb1 = _bruit(random.randint(2, 6))
        nb2 = _bruit(random.randint(2, 6))
    elif type_relation == 1:     # 1:N : entité 1 plus riche en attributs qu'entité 2
        nb1 = _bruit(random.randint(4, 10))
        nb2 = _bruit(random.randint(2, 7))
    else:                        # N:M : les deux entités ont beaucoup d'attributs
        nb1 = _bruit(random.randint(4, 12))
        nb2 = _bruit(random.randint(4, 12))

    # Ajoute un léger bruit aux scores sémantiques pour refléter l'incertitude
    s1_1 = float(np.clip(scores[0] + np.random.normal(0, 0.05), 0.0, 1.0))
    s1_n = float(np.clip(scores[1] + np.random.normal(0, 0.05), 0.0, 1.0))
    sn_m = float(np.clip(scores[2] + np.random.normal(0, 0.05), 0.0, 1.0))

    return {
        'nb_attr_e1':    nb1,
        'nb_attr_e2':    nb2,
        'ratio_attrs':   round(nb1 / max(nb2, 1), 3),
        'max_attrs':     max(nb1, nb2),
        'total_attrs':   nb1 + nb2,
        'longueur_nom':  len(mot),
        'score_1_1':     round(s1_1, 3),
        'score_1_N':     round(s1_n, 3),
        'score_N_M':     round(sn_m, 3),
        'label':         type_relation,
    }


def generer_dataset(n=3000, taux_bruit_labels=0.07):
    """Génère un jeu de données complet et l'enregistre dans dataset_mcd.csv."""
    data = []
    par_classe = n // 3

    for type_rel in [0, 1, 2]:
        for _ in range(par_classe):
            ex = generer_exemple(type_rel)
            # Injecte un petit niveau d'erreur de label pour simuler l'ambiguïté réelle
            if random.random() < taux_bruit_labels:
                autres = [t for t in [0, 1, 2] if t != type_rel]
                ex['label'] = random.choice(autres)
            data.append(ex)

    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv('dataset_mcd.csv', index=False)

    print(f"Dataset V2.0 genere : {len(df)} exemples - dataset_generator.py:142")
    print(f"Distribution :\n{df['label'].value_counts().sort_index()} - dataset_generator.py:143")
    return df


if __name__ == "__main__":
    generer_dataset(3000)
