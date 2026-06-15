# 🎯 Labo scénarios — fiche de documentation

> Onglet `with tab_scenario:` dans `app.py`. **Scorer n'importe quel schéma de
> netting** que l'on veut tester, plus le bac à sable à deux nœuds (forme fermée).

## But de l'onglet

Donner la main : un scénario est une **matrice de labels entiers (M × K)** sur la
grille maturité × strike — les cellules de même label forment un ensemble. On l'édite,
on la charge, ou on part d'un preset / de la solution optimale ; le moteur renvoie le
**score de variance R²**, le verdict EBA (test 6 + plancher 7) et l'impact AVA.

## Ce qui tourne à l'intérieur

1. **Point de départ** : `preset_labels(name, M, K)` (`ebanetting/scenario.py:70`)
   produit `per_tenor` (un ensemble par ligne de maturité, on écrase le smile),
   `per_strike`, `quadrants`, `global`, `singletons` ; ou les labels optimaux repris de
   `st.session_state["optimal_labels"]` ; ou une matrice chargée en JSON.
2. **Édition** : `st.data_editor` sur un `DataFrame` indexé par maturités × strikes.
3. **Scoring** : `score_scenario(bundle, labels, alpha, kappa, weighting)`
   (`scenario.py:58`) construit `NettingScheme.from_labels` puis appelle `.evaluate`
   (même machinerie que l'onglet Netting optimal : Propriété 1, test 6, plancher 7).
   `ScenarioReport` expose `variance_score = R²` et `verdict` (« ADMISSIBLE » ou
   « REJECTED — test de variance / plancher »).
4. **Affichage** : `r2_gauge` (R² vs α), badge de verdict, AVA du scénario (−% vs
   add-up), TE² vs budget, badges test/plancher, `partition_figure`. Bouton d'export
   des labels en JSON. Les résultats sont stockés dans `session_state`
   (`scenario_eval`, `scenario_labels`) pour l'onglet Audit.
5. **Bac à sable à deux nœuds** (expander) : `two_bucket(ν_i, ν_j, s_i, s_j, ρ, var,
   α, κ)` (`ebanetting/netting.py:182`) — Théorème 2 :
   - `TE² = ν_j²·(s_i² + s_j² − 2ρ s_i s_j)` (Th. 2 (i)),
   - seuil d'admissibilité `ρ_min` (éq. 8),
   - gain d'AVA (Th. 2 (iii)).
   Le choix « variance de la paire / du portefeuille » fixe le `Var(ΔΠ)` au membre de
   droite de (6). `two_bucket_figure` trace la frontière d'admissibilité `ρ_min(|ν_j|)`
   et y place votre paire.

## La raison derrière

- **Pourquoi un scorer libre.** Permet de confronter au test la convention de netting
  actuelle du desk (via une `ScenarioSource`), une hypothèse, ou la solution de
  l'optimiseur — sans présupposer qu'elle est admissible. Le verdict est objectif.
- **Pourquoi R² comme score principal.** Le test de variance (Déf. 3) équivaut à
  `R² ≥ α` : c'est l'analogue prudentiel des tests d'attribution de P&L FRTB. La jauge
  rend ce seuil lisible d'un coup d'œil.
- **Pourquoi la forme fermée à deux nœuds.** Elle bâtit l'intuition que l'optimiseur
  applique en grand : netter exige une corrélation élevée **et** des incertitudes
  homogènes ; plus le vega netté `|ν_j|` est grand relativement au risque total, plus
  le seuil `ρ_min` est exigeant — on ne cache pas un gros short derrière un proxy
  approximatif. Les fusions de même signe sont admissibles mais inutiles.
- **Pourquoi les presets.** Ce sont des partitions interprétables servant de points de
  départ naturels (« netting par maturité », « quadrants ») avant édition manuelle.

## Comment lire l'onglet

- La **jauge R²** au-dessus/sous α et les deux badges donnent immédiatement
  l'admissibilité.
- **AVA −% vs add-up** quantifie le gain du scénario ; **TE² vs budget** dit de combien
  on est sous (ou au-dessus) de la contrainte de fidélité.
- Dans le bac à sable, faites varier `ρ` et `|ν_j|` : la paire passe au vert dès
  qu'elle franchit la frontière `ρ_min` — la lecture géométrique de l'éq. (8).
