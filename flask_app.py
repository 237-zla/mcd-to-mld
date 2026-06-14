# app.py
import os
import json
from flask import Flask, render_template, request, jsonify
from dataset_generator import generer_dataset
from ml_model import entrainer_modele, charger_modele
from rules_engine import MoteurRegles
from mld_generator import GenerateurMLD

app = Flask(__name__)

modele_ml = None

def initialiser_modele():
    global modele_ml
    if modele_ml is not None:
        return modele_ml
    if not os.path.exists('dataset_mcd.csv'):
        generer_dataset(600)
    if not os.path.exists('modele_rf.pkl'):
        entrainer_modele('dataset_mcd.csv')
    modele_ml = charger_modele('modele_rf.pkl')
    return modele_ml

# ─── Préparer le modèle ML au démarrage ───
initialiser_modele()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/transformer', methods=['POST'])
def transformer():
    try:
        mcd = request.get_json()
        if not mcd or 'entites' not in mcd or 'associations' not in mcd:
            return jsonify({'erreur': 'Format MCD invalide'}), 400

        # Transformation MCD → MLD
        moteur = MoteurRegles()
        tables = moteur.transformer_mcd(mcd, modele_ml)

        # Génération MLD + SQL
        gen = GenerateurMLD(tables)
        lignes_mld = gen.afficher_mld()
        instructions_sql = gen.generer_sql()

        # Construire la réponse
        tables_json = {}
        for nom_table, table in tables.items():
            pk   = table['cle_primaire']
            fks  = table['cles_etrangeres']
            attrs = table['attributs']
            attrs_formates = [
                gen._formater_attribut(a, pk, fks) for a in attrs
            ]
            tables_json[nom_table] = {
                'attributs'      : attrs_formates,
                'cle_primaire'   : pk,
                'cles_etrangeres': [fk.lstrip('#') for fk in fks]
            }

        return jsonify({
            'mld_texte'  : lignes_mld,
            'sql'        : instructions_sql,
            'tables'     : tables_json,
            'nb_tables'  : len(tables)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'erreur': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)