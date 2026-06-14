# mld_generator.py

class GenerateurMLD:
    def __init__(self, tables):
        self.tables  = tables
        self._pk_map = self._construire_pk_map()

    def _construire_pk_map(self):
        pk_map = {}
        for nom_table, table in self.tables.items():
            pk = table['cle_primaire']
            if '(' not in pk:
                pk_map[pk] = nom_table
        return pk_map

    def _formater_attribut(self, attr, cle_primaire, cles_etrangeres):
        attr_clean = attr.lstrip('#')
        est_fk = attr.startswith('#') or attr in cles_etrangeres
        pk = cle_primaire
        if '(' in pk:
            pks = [k.strip() for k in pk.strip('()').split(',')]
            est_pk = attr_clean in pks
        else:
            est_pk = (attr_clean == pk)
        if est_pk and est_fk:   return f"*#{attr_clean}"
        elif est_pk:            return f"*{attr_clean}"
        elif est_fk:            return f"#{attr_clean}"
        else:                   return attr_clean

    def afficher_mld(self):
        lignes_mld = []
        print("\n - mld_generator.py:32" + "=" * 60)
        print("MODELE LOGIQUE DE DONNEES (MLD) - mld_generator.py:33")
        print("= - mld_generator.py:34" * 60)
        print("Legende : *=Cle Primaire | #=Cle Etrangere\n - mld_generator.py:35")
        for nom_table, table in self.tables.items():
            pk    = table['cle_primaire']
            fks   = table['cles_etrangeres']
            attrs = table['attributs']
            attrs_formates = [
                self._formater_attribut(a, pk, fks) for a in attrs
            ]
            ligne = f"  {nom_table}({', '.join(attrs_formates)})"
            lignes_mld.append(ligne)
            print(ligne)
        print("\n - mld_generator.py:46" + "=" * 60)
        return lignes_mld

    def generer_sql(self):
        instructions = []
        print("\n - mld_generator.py:51" + "=" * 60)
        print("CODE SQL DDL GENERE - mld_generator.py:52")
        print("= - mld_generator.py:53" * 60)
        for nom_table, table in self.tables.items():
            pk          = table['cle_primaire']
            fks_liste   = table['cles_etrangeres']
            attrs_bruts = table['attributs']
            lignes_sql  = [f"CREATE TABLE {nom_table} ("]
            for attr in attrs_bruts:
                ac     = attr.lstrip('#')
                est_fk = (attr.startswith('#') or attr in fks_liste)

                if ac == pk:
                    if ac.startswith('id_'):
                        type_sql = "INTEGER NOT NULL"
                    else:
                        type_sql = "VARCHAR(50) NOT NULL"
                elif est_fk:
                    if ac.startswith('id_'):
                        type_sql = "INTEGER"
                    else:
                        type_sql = "VARCHAR(50)"
                elif any(m in ac for m in ['date', 'naissance']):
                    type_sql = "DATE"
                elif any(m in ac for m in ['montant', 'prix', 'salaire',
                                            'note', 'taux', 'budget']):
                    type_sql = "DECIMAL(10, 2)"
                elif any(m in ac for m in ['age', 'nb_', 'credits', 'stock',
                                            'quantite', 'annee', 'volume',
                                            'capacite', 'pages', 'horaire']):
                    type_sql = "INTEGER"
                else:
                    type_sql = "VARCHAR(100)"

                lignes_sql.append(f"    {ac} {type_sql},")
            if '(' in pk:
                cols = ', '.join(k.strip() for k in pk.strip('()').split(','))
                lignes_sql.append(f"    PRIMARY KEY ({cols}),")
            else:
                lignes_sql.append(f"    PRIMARY KEY ({pk}),")
            for fk in fks_liste:
                fk_clean  = fk.lstrip('#')
                ref_table = self._pk_map.get(fk_clean, 'INCONNUE')
                lignes_sql.append(
                    f"    FOREIGN KEY ({fk_clean}) "
                    f"REFERENCES {ref_table}({fk_clean}),"
                )
            lignes_sql[-1] = lignes_sql[-1].rstrip(',')
            lignes_sql.append(");\n")
            bloc = '\n'.join(lignes_sql)
            instructions.append(bloc)
            print(bloc)
        return instructions

    def afficher_tout(self):
        self.afficher_mld()
        self.generer_sql()