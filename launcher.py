"""
Lanceur graphique — MCD vers MLD
Démarre le serveur Flask et ouvre le navigateur automatiquement.
"""

import tkinter as tk
import threading
import webbrowser
import time
import os
import sys

# S'assurer que le répertoire du script est le cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Lanceur:

    _C = {
        'bg':    '#0f0f1a',
        'rouge': '#e94560',
        'vert':  '#4ecca3',
        'texte': '#eaeaea',
        'gris':  '#64748b',
        'jaune': '#f0a500',
    }

    def __init__(self):
        self.root   = tk.Tk()
        self._actif = False
        self._configurer()
        self._construire()

    def _configurer(self):
        c = self._C
        self.root.title('MCD → MLD  |  Système IA')
        self.root.geometry('480x310')
        self.root.resizable(False, False)
        self.root.configure(bg=c['bg'])
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f'480x310+{(sw-480)//2}+{(sh-310)//2}')

    def _construire(self):
        c = self._C

        tk.Label(
            self.root, text='MCD  →  MLD',
            font=('Segoe UI', 30, 'bold'),
            fg=c['rouge'], bg=c['bg']
        ).pack(pady=(34, 4))

        tk.Label(
            self.root,
            text='Transformation automatique par Intelligence Artificielle',
            font=('Segoe UI', 9), fg=c['gris'], bg=c['bg']
        ).pack()

        tk.Label(
            self.root,
            text='SDIA Licence 3  ·  Bases de Données  ·  Session Normale',
            font=('Segoe UI', 8), fg=c['gris'], bg=c['bg']
        ).pack(pady=(2, 22))

        tk.Frame(self.root, bg=c['rouge'], height=1).pack(
            fill='x', padx=60, pady=(0, 18)
        )

        self._var = tk.StringVar(value='Cliquez pour démarrer')
        self._lbl = tk.Label(
            self.root, textvariable=self._var,
            font=('Segoe UI', 9), fg=c['gris'], bg=c['bg']
        )
        self._lbl.pack(pady=(0, 12))

        self._btn = tk.Button(
            self.root,
            text="     Lancer l'application     ",
            font=('Segoe UI', 10, 'bold'),
            fg='white', bg=c['rouge'],
            activebackground='#c73652',
            activeforeground='white',
            relief='flat', bd=0,
            padx=20, pady=10,
            cursor='hand2',
            command=self._lancer
        )
        self._btn.pack()

        tk.Label(
            self.root,
            text="L'interface s'ouvrira dans votre navigateur",
            font=('Segoe UI', 8), fg=c['gris'], bg=c['bg']
        ).pack(pady=(8, 0))

    # ── Logique ──────────────────────────────────────────────────────

    def _lancer(self):
        if self._actif:
            webbrowser.open('http://localhost:5000')
            return

        self._btn.config(state='disabled', text='     Initialisation...     ')
        self._statut('Chargement du modèle IA...', self._C['jaune'])

        threading.Thread(target=self._demarrer, daemon=True).start()

    def _demarrer(self):
        try:
            from flask_app import app, initialiser_modele
            from dataset_generator import generer_dataset
            from ml_model import entrainer_modele, charger_modele

            # Préparer le modèle
            if not os.path.exists('dataset_mcd.csv'):
                generer_dataset(600)
            if not os.path.exists('modele_rf.pkl'):
                entrainer_modele('dataset_mcd.csv')
            initialiser_modele()

            threading.Thread(
                target=lambda: app.run(
                    host='127.0.0.1', port=5000,
                    debug=False, use_reloader=False
                ),
                daemon=True
            ).start()

            time.sleep(1.5)
            self._actif = True
            self.root.after(0, self._pret)

        except Exception as e:
            self.root.after(0, lambda: self._erreur(str(e)))

    def _pret(self):
        self._statut(
            'Serveur actif  →  http://localhost:5000',
            self._C['vert']
        )
        self._btn.config(
            state='normal',
            text='     Ouvrir dans le navigateur     '
        )
        webbrowser.open('http://localhost:5000')

    def _erreur(self, msg):
        self._statut(f'Erreur : {msg[:55]}', self._C['rouge'])
        self._btn.config(state='normal', text='     Réessayer     ')

    def _statut(self, texte, couleur):
        self._var.set(texte)
        self._lbl.config(fg=couleur)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    Lanceur().run()