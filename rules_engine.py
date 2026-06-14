# rules_engine.py

class MoteurRegles:
    def __init__(self):
        self.tables = {}

    def reinitialiser(self):
        self.tables = {}

    def regle1_entite_vers_table(self, entite):
        self.tables[entite['nom']] = {
            'attributs'       : list(entite['attributs']),
            'cle_primaire'    : entite['cle'],
            'cles_etrangeres' : []
        }

    def regle2_association_1_N(self, asso):
        e1, e2 = asso['entites']
        c1, c2 = asso['cardinalites']
        if 'N' in c1 or 'n' in c1:
            parent, enfant = e1, e2
        else:
            parent, enfant = e2, e1
        pk_parent = self.tables[parent]['cle_primaire']
        fk        = f"#{pk_parent}"
        if fk not in self.tables[enfant]['cles_etrangeres']:
            self.tables[enfant]['cles_etrangeres'].append(fk)
            self.tables[enfant]['attributs'].append(fk)
        for attr in asso.get('attributs', []):
            if attr not in self.tables[enfant]['attributs']:
                self.tables[enfant]['attributs'].append(attr)
        print(f"      PK '{pk_parent}'  ->  FK dans '{enfant}'")

    def regle3_association_N_M(self, asso):
        e1, e2   = asso['entites']
        pk_e1    = self.tables[e1]['cle_primaire']
        pk_e2    = self.tables[e2]['cle_primaire']
        fk1, fk2 = f"#{pk_e1}", f"#{pk_e2}"
        attributs = [fk1, fk2] + list(asso.get('attributs', []))
        self.tables[asso['nom'].upper()] = {
            'attributs'       : attributs,
            'cle_primaire'    : f"({pk_e1}, {pk_e2})",
            'cles_etrangeres' : [fk1, fk2]
        }
        print(f"      Table de jonction '{asso['nom'].upper()}' créée")
        print(f"      Clé composite : ({pk_e1}, {pk_e2})")

    def regle4_association_1_1(self, asso):
        e1, e2 = asso['entites']
        c1, c2 = asso['cardinalites']
        if c1.startswith('0'):
            receveur, donneur = e1, e2
        elif c2.startswith('0'):
            receveur, donneur = e2, e1
        else:
            receveur, donneur = e2, e1
        pk_donneur = self.tables[donneur]['cle_primaire']
        fk         = f"#{pk_donneur}"
        if fk not in self.tables[receveur]['cles_etrangeres']:
            self.tables[receveur]['cles_etrangeres'].append(fk)
            self.tables[receveur]['attributs'].append(fk)
        for attr in asso.get('attributs', []):
            if attr not in self.tables[receveur]['attributs']:
                self.tables[receveur]['attributs'].append(attr)
        print(f"      PK '{pk_donneur}'  ->  FK dans '{receveur}'")

    def transformer_mcd(self, mcd, modele_ml):
        from ml_model import predire_cardinalite
        self.reinitialiser()
        entites_dict = {e['nom']: e for e in mcd['entites']}

        print("\n" + "=" * 54)
        print("  ETAPE 1 - Regle 1 : Entites -> Tables")
        print("=" * 54)
        for entite in mcd['entites']:
            self.regle1_entite_vers_table(entite)
            print(f"  [OK]  {entite['nom']:15s}  (PK = {entite['cle']})")

        print("\n" + "=" * 54)
        print("  ETAPE 2 - Associations -> Regles 2 / 3 / 4")
        print("=" * 54)
        for asso in mcd['associations']:
            e1, e2 = asso['entites']
            c1, c2 = asso['cardinalites']
            print(f"\n  -> '{asso['nom'].upper()}'  "
                  f"({e1} [{c1}] --- [{c2}] {e2})")
            nb1 = len(entites_dict[e1]['attributs'])
            nb2 = len(entites_dict[e2]['attributs'])
            _, label_ml, conf = predire_cardinalite(
                modele_ml, nb1, nb2, asso['nom'].lower()
            )
            both_N = (('N' in c1 or 'n' in c1) and ('N' in c2 or 'n' in c2))
            one_N  = (('N' in c1 or 'n' in c1) or ('N' in c2 or 'n' in c2))
            if both_N:
                type_reel = 'N:M'
            elif one_N:
                type_reel = '1:N'
            else:
                type_reel = '1:1'
            accord = "[OK] Accord" if type_reel == label_ml else "[!] Ecart"
            print(f"     Type MCD  : {type_reel}  |  "
                  f"ML predit : {label_ml} ({conf:.0f}%)  {accord}")
            if type_reel == '1:1':
                print("     -> Regle 4 appliquee (1:1)")
                self.regle4_association_1_1(asso)
            elif type_reel == '1:N':
                print("     -> Regle 2 appliquee (1:N)")
                self.regle2_association_1_N(asso)
            else:
                print("     -> Regle 3 appliquee (N:M)")
                self.regle3_association_N_M(asso)

        print("\n" + "=" * 54)
        print("  TRANSFORMATION TERMINEE")
        print("=" * 54)
        return self.tables
