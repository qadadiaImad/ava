# 🧠 Netting optimal — fiche de documentation

> Onglet `with tab_optimal:` dans `app.py`. Le **glouton joint** de la note (§7.3) :
> trouver la partition rectangulaire qui minimise l'AVA sous le test de variance et le
> plancher.

## But de l'onglet

Résoudre, de façon contrôlée, le problème d'optimisation (10) : la partition des nœuds
qui minimise `AVA(P) = κ Σ_r |m_r| s̃_r` sous la contrainte
`TE²(P) ≤ (1−α)·Var(ΔΠ)` (éq. 6) et le plancher (éq. 7). Comme (10) est NP-difficile
(combinatoire des nombres de Bell), on restreint l'espace de recherche aux **pavages
rectangulaires contigus** et on agglomère glouton­nement.

## Ce qui tourne à l'intérieur

1. **Glouton de base.** `cached_greedy(...)` → `greedy_netting(model, alpha, weighting)`
   (`ebanetting/optimizer.py:142`). Le moteur `_greedy` (`optimizer.py:83`) :
   - part des **singletons** (chaque bucket est un rectangle 1×1) ;
   - à chaque itération, parcourt toutes les paires de rectangles **adjacents et
     fusionnables** (`_mergeable`, `_union`) dont l'union reste un rectangle ;
   - évalue chaque essai via `NettingScheme.evaluate` (voir ci-dessous) ;
   - **priorise** selon l'objectif (`objective`, voir le toggle ci-dessous) :
     - `"ava"` (défaut) : d'abord les fusions à coût nul (`cost ≤ eps` et `gain ≥ 0`),
       puis celles maximisant `gain/cost` (réduction d'AVA par unité de variance
       résiduelle) ;
     - `"te"` : classe les fusions par coût en variance croissant (`−cost`, les moins
       chères d'abord, coût nul en tête), **sans** pondérer par le gain d'AVA ;
   - rejette toute fusion qui ferait dépasser le budget `B = (1−α)·Var` ;
   - enregistre chaque fusion retenue dans un `MergeStep`.
   - **Étape 3 (roll-back du plancher)** : tant que `passes_floor` est faux, on retire
     la dernière fusion et on rejoue (`_replay`) — `rolled_back` compte les retraits.
2. **Évaluation d'un schéma.** `NettingScheme.evaluate(model, alpha)`
   (`ebanetting/netting.py:87`) :
   - `weight_grids` construit un choc représentatif par ensemble selon `weighting`
     (`pivot` = cellule de risque dominant `max |N|·s`, `vega` = pondéré par |vega|,
     `equal`) ;
   - `m_r = Σ vega` sur l'ensemble ; `s̃_r = √Var(W_r)` (Propriété 1) ;
   - `proxy = Σ_r m_r W_r`, `residual = N − proxy`, `te2 = Var(residual)` ;
   - `R² = 1 − te2/var_total`, `AVA = κ Σ|m_r| s̃_r` ;
   - `passes_variance` (te2 ≤ budget), `passes_floor` (AVA ≥ ava_full), `set_stats`.
3. **Objectif d'optimisation** (toggle « 💶 Inclure le coût AVA dans l'objectif »).
   Activé → `objective="ava"` (comportement par défaut ci-dessus) ; désactivé →
   `objective="te"` (optimisation selon la tracking error seule). Le paramètre est
   transmis à `greedy_netting` et, en mode robuste, à chaque run de `robust_netting`.
   Le budget de variance et le roll-back du plancher (7) s'appliquent dans les deux cas.
4. **Mode robuste** (toggle). `cached_robust(...)` → `robust_netting(bundle, alpha,
   kappa, deltas, weighting)` (`optimizer.py:200`) : rejoue le glouton sous
   `stress_bundle` (ρ → max(ρ−δ, −1), `optimizer.py:175`) pour chaque δ, puis prend le
   **raffinement commun** (deux buckets restent ensemble seulement s'ils le sont dans
   *tous* les régimes) et le réévalue sous le modèle de base.

Affichage : 5 métriques (ensembles, R², AVA nettée, budget consommé, verdicts),
`partition_figure` (pavage rectangulaire), `ava_waterfall` (add-up → nettée vs
plancher), `merge_history_figure` (chemin de TE² vs budget et AVA), table des régimes
de stress, expander du détail par ensemble. Les résultats sont mémorisés dans
`st.session_state` (`optimal_labels`, `optimal_eval`, `optimal_history`,
`robust_pack`) pour les onglets Labo scénarios et Audit.

## La raison derrière

- **Pourquoi des rectangles contigus.** Ils sont interprétables et documentables au
  titre de l'art. 9(5) (« cet ensemble regroupe les maturités 1M–3M sur les strikes
  0,9–1,1 »). Ils réduisent aussi drastiquement l'espace combinatoire.
- **Pourquoi prioriser le coût nul puis `gain/cost`.** Les fusions gratuites
  (compensation parfaite, aucune variance résiduelle) sont prises d'abord. Ensuite, on
  achète la réduction d'AVA la plus efficace par unité de budget de variance dépensé —
  l'analogue prudentiel d'un ratio coût/bénéfice.
- **Pourquoi un objectif « TE seule » optionnel.** L'objectif AVA ignore les fusions
  sans gain d'AVA (`gain ≤ 0`), même quand elles ne coûtent presque rien en fidélité
  (p. ex. des vegas de même signe). L'objectif TE seule les retient : il agrège le plus
  possible tant que le budget de variance le permet, en classant par coût croissant.
  Il répond à « quelle est la maille de netting la plus grossière justifiable par la
  seule fidélité ? » plutôt qu'à « quelle partition minimise l'AVA ? ». La partition
  obtenue n'est pas garantie AVA-optimale, d'où le toggle (défaut = objectif AVA).
- **Pourquoi le roll-back du plancher.** Le test (6) contrôle la *fidélité* ; le
  plancher (7) contrôle le *niveau*. Une partition peut passer (6) mais descendre
  l'AVA sous `κ√Var` — interdit. On défait alors les dernières fusions jusqu'à
  respecter (7).
- **Pourquoi le mode robuste.** Étape 4 de l'algorithme : une fusion n'est défendable
  en revue de modèle que si elle survit à un stress de corrélation *adverse au
  netting*. Le raffinement commun ne garde que les fusions stables — incarnation de
  l'asymétrie réglementaire.
- **Pourquoi le pivot par défaut.** Le choc représentatif est porté par la cellule de
  risque dominant (`|N|·s` max) : conservateur et stable.

## Comment lire l'onglet

- **R² vs α** et les deux badges (test 6 / plancher 7) donnent le verdict.
- Le **waterfall** montre combien d'AVA le netting récupère, et que l'on reste
  au-dessus du plancher de pleine diversification.
- Le **chemin de fusion** : la courbe TE² doit rester sous la ligne de budget ; chaque
  marche est une fusion. Un `rolled back` signalé veut dire que le plancher a mordu.
- En mode robuste, comparez la ligne « robuste (raffinement commun) » aux régimes
  stressés : les ensembles qui disparaissent sous stress étaient fragiles.
