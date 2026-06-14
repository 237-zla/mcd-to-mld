# main.py
import os
import sys
from dataset_generator import generer_dataset
from ml_model import entrainer_modele, charger_modele
from rules_engine import MoteurRegles
from mld_generator import GenerateurMLD

# ───────────────────────────────────────────────
#  3 MCD d'exemple pour la démonstration
# ───────────────────────────────────────────────

MCD_BIBLIOTHEQUE = {
    "entites": [
        {"nom": "ADHERENT",  "cle": "id_adherent",
         "attributs": ["id_adherent", "nom", "prenom", "email", "date_naissance"]},
        {"nom": "LIVRE",     "cle": "isbn",
         "attributs": ["isbn", "titre", "annee", "nb_pages"]},
        {"nom": "AUTEUR",    "cle": "id_auteur",
         "attributs": ["id_auteur", "nom_auteur", "nationalite"]},
        {"nom": "CATEGORIE", "cle": "id_categorie",
         "attributs": ["id_categorie", "libelle"]},
    ],
    "associations": [
        {"nom": "emprunte",   "entites": ["ADHERENT", "LIVRE"],
         "cardinalites": ["0,N", "0,N"],
         "attributs": ["date_emprunt", "date_retour"]},
        {"nom": "redige",     "entites": ["AUTEUR", "LIVRE"],
         "cardinalites": ["1,N", "0,N"], "attributs": []},
        {"nom": "appartient", "entites": ["CATEGORIE", "LIVRE"],
         "cardinalites": ["1,N", "1,1"], "attributs": []},
    ]
}

MCD_UNIVERSITE = {
    "entites": [
        {"nom": "ETUDIANT",   "cle": "matricule",
         "attributs": ["matricule", "nom", "prenom", "filiere"]},
        {"nom": "COURS",      "cle": "id_cours",
         "attributs": ["id_cours", "intitule", "credits", "volume_horaire"]},
        {"nom": "ENSEIGNANT", "cle": "id_enseignant",
         "attributs": ["id_enseignant", "nom", "grade", "specialite"]},
        {"nom": "SALLE",      "cle": "id_salle",
         "attributs": ["id_salle", "capacite", "batiment"]},
    ],
    "associations": [
        {"nom": "inscrit",    "entites": ["ETUDIANT", "COURS"],
         "cardinalites": ["0,N", "0,N"],
         "attributs": ["note", "semestre"]},
        {"nom": "enseigne",   "entites": ["ENSEIGNANT", "COURS"],
         "cardinalites": ["0,N", "0,N"], "attributs": []},
        {"nom": "reserve",    "entites": ["SALLE", "COURS"],
         "cardinalites": ["1,N", "1,1"],
         "attributs": ["horaire"]},
    ]
}

MCD_HOPITAL = {
    "entites": [
        {"nom": "PATIENT",   "cle": "id_patient",
         "attributs": ["id_patient", "nom", "prenom", "date_naissance", "adresse"]},
        {"nom": "MEDECIN",   "cle": "id_medecin",
         "attributs": ["id_medecin", "nom", "specialite", "telephone"]},
        {"nom": "SERVICE",   "cle": "id_service",
         "attributs": ["id_service", "nom_service", "etage", "capacite"]},
        {"nom": "MEDICAMENT","cle": "id_medicament",
         "attributs": ["id_medicament", "nom_medicament", "dosage", "prix"]},
    ],
    "associations": [
        {"nom": "consulte",   "entites": ["PATIENT", "MEDECIN"],
         "cardinalites": ["0,N", "0,N"],
         "attributs": ["date_consultation", "diagnostic"]},
        {"nom": "appartient", "entites": ["SERVICE", "MEDECIN"],
         "cardinalites": ["1,N", "1,1"], "attributs": []},
        {"nom": "utilise",    "entites": ["MEDECIN", "MEDICAMENT"],
         "cardinalites": ["0,N", "0,N"],
         "attributs": ["quantite"]},
    ]
}

def preparer_modele():
    """Génère le dataset et entraîne le modèle si nécessaire."""
    if not os.path.exists('dataset_mcd.csv'):
        print("Génération du dataset...")
        generer_dataset(600)
    if not os.path.exists('modele_rf.pkl'):
        print("Entraînement du modèle...")
        entrainer_modele('dataset_mcd.csv')
    return charger_modele('modele_rf.pkl')

def transformer_et_afficher(mcd, modele):
    """Transforme un MCD et affiche le MLD + SQL."""
    moteur = MoteurRegles()
    tables = moteur.transformer_mcd(mcd, modele)
    gen = GenerateurMLD(tables)
    gen.afficher_tout()

def menu_principal():
    """Menu interactif en mode terminal."""
    modele = preparer_modele()

    while True:
        print("\n" + "=" * 50)
        print("   MCD -> MLD  |  Système Hybride IA + Règles")
        print("=" * 50)
        print("  1. Bibliothèque")
        print("  2. Université")
        print("  3. Hôpital")
        print("  4. Quitter")
        print("=" * 50)

        choix = input("  Votre choix : ").strip()

        if choix == '1':
            print("\n  >>> Transformation du MCD : BIBLIOTHEQUE")
            transformer_et_afficher(MCD_BIBLIOTHEQUE, modele)
        elif choix == '2':
            print("\n  >>> Transformation du MCD : UNIVERSITE")
            transformer_et_afficher(MCD_UNIVERSITE, modele)
        elif choix == '3':
            print("\n  >>> Transformation du MCD : HOPITAL")
            transformer_et_afficher(MCD_HOPITAL, modele)
        elif choix == '4':
            print("\n  Au revoir !")
            sys.exit(0)
        else:
            print("  Choix invalide, réessayez.")

if __name__ == "__main__":
    menu_principal()