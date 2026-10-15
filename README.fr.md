# ava — Repo implicite Totem : contrôle et détection des courbes aberrantes

[English](README.md) | **Français**

Les paramètres de repo utilisés pour le contrôle indépendant des prix sont déduits de Totem : les forwards à partir des
prix de consensus des calls et des puts, les dividendes à partir du service dividendes de Totem, qui est distinct. Comme
les deux services n'ont pas les mêmes contributeurs, la courbe de repo implicite présente parfois des formes
irrégulières ou des sauts que le marché ne justifie pas.

Ce dépôt contient la note méthodologique et un prototype Python :

- **Note méthodologique :** [`docs/totem_repo_note_fr.pdf`](docs/totem_repo_note_fr.pdf) (français) et
  [`docs/totem_repo_note_en.pdf`](docs/totem_repo_note_en.pdf) (anglais). La note énonce ses trois objectifs dès la
  première page (comprendre les courbes aberrantes, contrôler la position repo indépendamment des contributeurs
  dividendes, détecter les courbes aberrantes), suit un même exemple du début à la fin et donne les démonstrations.
  Les sources sont dans `docs/latex/`.
- **Package :** `repo_control`, qui ne dépend que de numpy et scipy.

## Ce que fait le prototype

| Étape | Module |
|---|---|
| Forwards à partir des prix de consensus des calls et des puts (zéro de la parité call-put, facteur d'actualisation implicite comme diagnostic) | `parity` |
| Allocation des périodes de dividendes Totem sur les dates de détachement ; bootstrap exact du repo implicite | `buckets`, `curves` |
| Modèle de bruit de l'écart de portage Totem − FO à partir des dispersions et des nombres de contributeurs publiés par Totem (panels emboîtés, écart de panel, sauts aux sorties de contributeurs) | `noise`, `pipeline` |
| Détecteur structurel : tests par filtre adapté des formes types (niveau, marche, pic, saison) après retrait d'un écart de repo lisse, taux de fausses alertes calibré, explication en unités naturelles, courbe réparée, test d'erreur de marquage FO | `detect` |
| Lignes de contrôle : lignes dividendes, repo, forward (forwards Totem avec dividendes FO) et non-forward ; matérialité de la répartition entre repo et dividendes à partir des sensibilités | `control` |
| Pricer minimal pour les sensibilités utilisées dans les exemples | `pricer` |
| Marché Totem synthétique construit contributeur par contributeur, scénarios, banc d'essai à taux de fausses alertes égal contre des contrôles naïfs et B-spline + mélange gaussien | `synthetic`, `benchmark`, `baselines` |

## Utilisation

```bash
pip install -e .[benchmark,report,dev]
pytest                                                   # 22 tests
python scripts/run_benchmark.py                          # ~5 min, écrit results/benchmark.json.gz
PYTHONPATH=. python scripts/build_latex.py               # figures, tableaux et les deux notes (TeX Live requis)
```

Sur données réelles, construire un `TotemSnapshot` et un `FOSnapshot` (voir `repo_control/pipeline.py`), appeler
`imply_totem_repo`, puis lancer `StructuralDetector.detect` sur `implied.delta_carry` et `implied.cov`. L'échelle de
variance du détecteur se calibre sur un historique sain avec `null_statistics` et `calibrate_to_fpr`.
