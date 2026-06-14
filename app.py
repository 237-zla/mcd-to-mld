import gradio as gr
import json
import os
import sys
import io
import contextlib

# ── Initialisation du modèle au démarrage ────────────────────────────────────
from dataset_generator import generer_dataset
from ml_model          import entrainer_modele, charger_modele
from rules_engine      import MoteurRegles
from mld_generator     import GenerateurMLD

_modele = None

def _init_modele():
    global _modele
    if _modele is not None:
        return
    if os.path.exists('modele_rf.pkl'):
        _modele = charger_modele()
    else:
        generer_dataset(3000)
        _modele = entrainer_modele('dataset_mcd.csv')

_init_modele()


# ── Inférence de type SQL ─────────────────────────────────────────────────────
def _sql_type(attr, est_pk, est_fk):
    if est_pk and not est_fk:
        return 'INTEGER NOT NULL' if attr.startswith('id_') else 'VARCHAR(50) NOT NULL'
    if est_fk:
        return 'INTEGER' if attr.startswith('id_') else 'VARCHAR(50)'
    a = attr.lower()
    if any(k in a for k in ['date', 'naissance']):          return 'DATE'
    if any(k in a for k in ['montant', 'prix', 'salaire',
                              'note', 'taux', 'budget']):   return 'DECIMAL(10,2)'
    if any(k in a for k in ['age', 'nb_', 'credits', 'stock',
                              'quantite', 'annee', 'volume',
                              'capacite', 'pages', 'horaire',
                              'etage']):                    return 'INTEGER'
    return 'VARCHAR(100)'


def _construire_sql(tables, pk_map):
    blocs = []
    for nom, tbl in tables.items():
        pk    = tbl['cle_primaire']
        fks   = tbl['cles_etrangeres']
        cols  = []
        for attr in tbl['attributs']:
            ac   = attr.lstrip('#')
            e_fk = attr.startswith('#') or attr in fks
            if '(' in pk:
                e_pk = ac in [k.strip() for k in pk.strip('()').split(',')]
            else:
                e_pk = (ac == pk)
            cols.append(f'    {ac} {_sql_type(ac, e_pk, e_fk)},')
        if '(' in pk:
            parts = ', '.join(k.strip() for k in pk.strip('()').split(','))
            cols.append(f'    PRIMARY KEY ({parts}),')
        else:
            cols.append(f'    PRIMARY KEY ({pk}),')
        for fk in fks:
            ac  = fk.lstrip('#')
            ref = pk_map.get(ac, 'INCONNUE')
            cols.append(f'    FOREIGN KEY ({ac}) REFERENCES {ref}({ac}),')
        cols[-1] = cols[-1].rstrip(',')
        blocs.append(f'CREATE TABLE {nom} (\n' + '\n'.join(cols) + '\n);')
    return '\n\n'.join(blocs)


# ── Transformation principale ─────────────────────────────────────────────────
def transformer(json_input: str):
    """
    Prend un MCD au format JSON et retourne :
    - La notation MLD Merise
    - Le code SQL DDL
    - Un résumé de la transformation
    """
    try:
        mcd = json.loads(json_input)
    except json.JSONDecodeError as e:
        return (
            f"❌ JSON invalide : {e}",
            "",
            "Erreur de parsing — vérifiez la syntaxe JSON."
        )

    if 'entites' not in mcd or not mcd['entites']:
        return (
            "❌ Champ 'entites' manquant ou vide.",
            "",
            ""
        )

    if any(not e.get('nom') or not e.get('cle') for e in mcd['entites']):
        return (
            "❌ Chaque entité doit avoir 'nom' et 'cle'.",
            "",
            ""
        )

    try:
        moteur = MoteurRegles()
        logs_moteur = io.StringIO()
        with contextlib.redirect_stdout(logs_moteur):
            tables = moteur.transformer_mcd(mcd, _modele)
        details_ml = []
        association = None
        for ligne in logs_moteur.getvalue().splitlines():
            texte = ligne.strip()
            if texte.startswith("-> '"):
                association = texte.split("'")[1]
            elif "ML predit :" in texte:
                prefixe = f"{association} : " if association else ""
                details_ml.append(prefixe + texte)

        gen  = GenerateurMLD(tables)

        # ── MLD Notation ──
        lignes = []
        for nom, tbl in tables.items():
            pk  = tbl['cle_primaire']
            fks = tbl['cles_etrangeres']
            fmt = [gen._formater_attribut(a, pk, fks)
                   for a in tbl['attributs']]
            lignes.append(f"{nom}({', '.join(fmt)})")

        mld_text = "\n".join(lignes)
        legende  = "\n* = Clé Primaire  |  # = Clé Étrangère  |  *# = PK Composite"
        mld_out  = mld_text + "\n" + "─" * 55 + legende

        # ── SQL ──
        sql_out = _construire_sql(tables, gen._pk_map)

        # ── Résumé ──
        nb_tables     = len(tables)
        nb_jonctions  = sum(
            1 for t in tables.values()
            if len(t['cles_etrangeres']) >= 2
            and '(' in t['cle_primaire']
        )
        nb_fk         = sum(len(t['cles_etrangeres']) for t in tables.values())
        resume = (
            f"Transformation reussie\n\n"
            f"Tables générées     : {nb_tables}\n"
            f"Tables de jonction  : {nb_jonctions}\n"
            f"Clés étrangères     : {nb_fk}\n"
            f"Entités en entrée   : {len(mcd['entites'])}\n"
            f"Associations        : {len(mcd.get('associations', []))}"
        )
        if details_ml:
            resume += "\n\nConfiance ML :\n" + "\n".join(details_ml)

        return mld_out, sql_out, resume

    except Exception as e:
        return (
            f"Erreur durant la transformation : {str(e)}",
            "",
            ""
        )


# ── Exemples ──────────────────────────────────────────────────────────────────
EX_BIBLIOTHEQUE = json.dumps({
    "entites": [
        {"nom": "ADHERENT",  "attributs": ["id_adherent","nom","prenom","email","date_naissance"], "cle": "id_adherent"},
        {"nom": "LIVRE",     "attributs": ["isbn","titre","annee","nb_pages"],                     "cle": "isbn"},
        {"nom": "AUTEUR",    "attributs": ["id_auteur","nom_auteur","nationalite"],                "cle": "id_auteur"},
        {"nom": "CATEGORIE", "attributs": ["id_categorie","libelle"],                              "cle": "id_categorie"}
    ],
    "associations": [
        {"nom": "emprunte",   "entites": ["ADHERENT","LIVRE"],     "cardinalites": ["0,N","0,N"], "attributs": ["date_emprunt","date_retour"]},
        {"nom": "redige",     "entites": ["AUTEUR","LIVRE"],       "cardinalites": ["1,N","0,N"], "attributs": []},
        {"nom": "appartient", "entites": ["LIVRE","CATEGORIE"],    "cardinalites": ["1,1","1,N"], "attributs": []}
    ]
}, indent=2, ensure_ascii=False)

EX_UNIVERSITE = json.dumps({
    "entites": [
        {"nom": "ETUDIANT",   "attributs": ["matricule","nom","prenom","filiere"],              "cle": "matricule"},
        {"nom": "COURS",      "attributs": ["id_cours","intitule","credits","volume_horaire"],  "cle": "id_cours"},
        {"nom": "ENSEIGNANT", "attributs": ["id_enseignant","nom","grade","specialite"],        "cle": "id_enseignant"},
        {"nom": "SALLE",      "attributs": ["id_salle","capacite","batiment"],                  "cle": "id_salle"}
    ],
    "associations": [
        {"nom": "inscrit",  "entites": ["ETUDIANT","COURS"],      "cardinalites": ["0,N","1,N"], "attributs": ["note","semestre"]},
        {"nom": "enseigne", "entites": ["ENSEIGNANT","COURS"],    "cardinalites": ["1,N","0,N"], "attributs": []},
        {"nom": "utilise",  "entites": ["COURS","SALLE"],         "cardinalites": ["1,1","0,N"], "attributs": ["horaire"]}
    ]
}, indent=2, ensure_ascii=False)

EX_HOPITAL = json.dumps({
    "entites": [
        {"nom": "PATIENT",    "attributs": ["id_patient","nom","prenom","date_naissance","tel"],  "cle": "id_patient"},
        {"nom": "MEDECIN",    "attributs": ["id_medecin","nom","specialite","bureau"],            "cle": "id_medecin"},
        {"nom": "SERVICE",    "attributs": ["id_service","nom_service","etage"],                  "cle": "id_service"},
        {"nom": "MEDICAMENT", "attributs": ["id_medicament","nom_medicament","dosage","prix"],    "cle": "id_medicament"}
    ],
    "associations": [
        {"nom": "consulte",   "entites": ["PATIENT","MEDECIN"],      "cardinalites": ["0,N","0,N"], "attributs": ["date_consultation","diagnostic"]},
        {"nom": "appartient", "entites": ["SERVICE","MEDECIN"],      "cardinalites": ["1,N","1,1"], "attributs": []},
        {"nom": "utilise",    "entites": ["MEDECIN","MEDICAMENT"],   "cardinalites": ["0,N","0,N"], "attributs": ["quantite"]}
    ]
}, indent=2, ensure_ascii=False)


# ── Interface Gradio ──────────────────────────────────────────────────────────
CSS = """
.title    { font-size: 1.6rem; font-weight: 800; }
.subtitle { color: #64748b; font-size: 0.9rem; }
footer    { display: none !important; }
"""

with gr.Blocks(title="MCD → MLD Transformer") as demo:

    gr.Markdown(
        """
# MCD → MLD Transformer
**Système de transformation automatique Merise — Architecture hybride IA + Règles déterministes**
Renseignez votre MCD au format JSON pour obtenir le MLD Merise et le code SQL DDL correspondant.
---
        """
    )

    with gr.Row():

        # ── Colonne gauche : Entrée ───────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Modèle Conceptuel de Données (MCD)")

            json_input = gr.Code(
                language="json",
                label="MCD au format JSON",
                value=EX_BIBLIOTHEQUE,
                lines=28,
            )

            with gr.Row():
                btn_bib  = gr.Button("Gestion Bibliothèque", size="sm", variant="secondary")
                btn_univ = gr.Button("Gestion Universitaire", size="sm", variant="secondary")
                btn_hop  = gr.Button("Gestion Hospitalière",  size="sm", variant="secondary")

            btn_transform = gr.Button(
                "Transformer MCD → MLD",
                variant="primary",
                size="lg",
            )

        # ── Colonne droite : Résultats ────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Modèle Logique de Données (MLD) + SQL DDL")

            out_resume = gr.Textbox(
                label="Résumé",
                lines=12,
                interactive=False,
                placeholder="Le résumé apparaîtra ici après la transformation."
            )

            out_mld = gr.Textbox(
                label="Notation MLD Merise",
                lines=10,
                interactive=False,
                placeholder="TABLE(*pk, attribut, #fk)"
            )

            out_sql = gr.Code(
                language="sql",
                label="Code SQL DDL",
                lines=20,
                interactive=False,
            )

    # ── Accordéon : Format JSON ───────────────────────────────────────────────
    with gr.Accordion("Format JSON attendu", open=False):
        gr.Markdown("""
```json
{
  "entites": [
    {
      "nom": "NOM_ENTITE",
      "attributs": ["attribut1", "attribut2", "..."],
      "cle": "attribut_cle_primaire"
    }
  ],
  "associations": [
    {
      "nom": "nom_association",
      "entites": ["ENTITE1", "ENTITE2"],
      "cardinalites": ["1,N", "0,N"],
      "attributs": ["attribut_asso_optionnel"]
    }
  ]
}
```
**Cardinalités supportées :** `0,1` · `1,1` · `0,N` · `1,N`

**Règles Merise appliquées :**
- `1:1` → Règle 4 : FK vers l'entité optionnelle
- `1:N` → Règle 2 : FK migre vers l'enfant
- `N:M` → Règle 3 : Table de jonction
        """)

    with gr.Accordion("À propos du projet", open=False):
        gr.Markdown("""
**Auteur :** OURBANI DOUBLA MAMOUDOU Ibrahima · SDIA L3 · ENSPD Douala, Cameroun

**Architecture :**
- Dataset synthétique 3000 exemples (généré automatiquement)
- Classificateur Random Forest (scikit-learn)
- Moteur de règles Merise déterministe
- Générateur SQL avec inférence de types

**Limitations connues :**
Le modèle ML utilise des scores sémantiques sur un vocabulaire français défini.
Pour des mots hors vocabulaire, le système se base sur les cardinalités explicites
du MCD (qui ont toujours l'autorité finale).

**Code source :** [github.com/237-zla/mcd-to-mld](https://github.com/237-zla/mcd-to-mld)
        """)

    # ── Événements ───────────────────────────────────────────────────────────
    btn_transform.click(
        fn=transformer,
        inputs=[json_input],
        outputs=[out_mld, out_sql, out_resume]
    )
    
    # Exemple buttons: update JSON and trigger transformation
    btn_bib.click(
        fn=lambda: EX_BIBLIOTHEQUE,
        outputs=[json_input]
    ).then(
        fn=transformer,
        inputs=[json_input],
        outputs=[out_mld, out_sql, out_resume]
    )
    
    btn_univ.click(
        fn=lambda: EX_UNIVERSITE,
        outputs=[json_input]
    ).then(
        fn=transformer,
        inputs=[json_input],
        outputs=[out_mld, out_sql, out_resume]
    )
    
    btn_hop.click(
        fn=lambda: EX_HOPITAL,
        outputs=[json_input]
    ).then(
        fn=transformer,
        inputs=[json_input],
        outputs=[out_mld, out_sql, out_resume]
    )

    # Transformation automatique au chargement avec l'exemple par défaut
    demo.load(
        fn=transformer,
        inputs=[json_input],
        outputs=[out_mld, out_sql, out_resume]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        show_error=True
    )
