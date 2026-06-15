# 🌳 Structure & Stabilité — fiche de documentation

> Onglet `with tab_structure:` dans `app.py`. L'**architecture découplée** de la note
> (§7.5) : la structure par la variance (Run 1), le niveau par l'AVA (Run 2), plus le
> protocole de stabilité entre familles de chocs (§7.4).

## But de l'onglet

Séparer ce qui doit être lent et justifié (la *structure* des ensembles) de ce qui est
rapide et quotidien (le *niveau* d'AVA). Le glouton joint de l'onglet précédent mêle un
objectif (l'AVA, qui dépend du book, de `s` et de κ) à une contrainte (le coût en
variance, qui ne dépend que de Σ). Ici on les sépare en deux runs.

## Ce qui tourne à l'intérieur

1. **Run 1 — le dendrogramme, sans portefeuille.** `cached_dendrogram(...)` →
   `build_dendrogram(bundle)` (`ebanetting/clustering.py:153`) :
   - calcule la **distance de risque de base** `base_risk_distance`
     (`clustering.py:67`) : `d_ij = √(s_i² + s_j² − 2 ρ_ij s_i s_j)` entre nœuds —
     exactement la quantité du Théorème 2 (i) (fusionner j sur le pivot i coûte
     `TE² = ν_j²·d_ij²`) ;
   - agglomère, en se limitant aux rectangles contigus, en fusionnant la paire dont
     l'union a le plus petit `ε` au sens de la **Propriété 8** (`_set_epsilon` :
     `max_j d_{j,pivot}/s_j` minimisé sur le choix du pivot) ;
   - ne dépend que de Σ (`s`, `ρ`) — **jamais du book**.
2. **Run 2 — le niveau, par l'AVA.** Le slider ε puis `decoupled_netting(bundle,
   alpha, kappa, epsilon, dendrogram)` (`clustering.py:295`) :
   - coupe l'arbre à la hauteur ε (`n_merges_at`), évalue le book du jour via
     `evaluate_cut` (`clustering.py:209`) en prenant pour choc représentatif le
     **pivot de structure** (`cut_pivots`, indépendant du book) ;
   - **contrôle obligatoire par book** : si le test (6) ou le plancher (7) échoue, on
     *abaisse la coupe* (on défait des fusions) jusqu'à ce que les deux passent —
     `lowered` compte les fusions défaites ;
   - calcule la borne Prop. 8 `te_bound = ε_réalisé · Σ|ν|s`.
3. **Affichage Run 1/2** : 5 métriques (ensembles, ε réalisé, borne Prop. 8 sur TE vs
   TE réelle, R², AVA), `dendrogram_figure` (hauteurs de fusion + coupe),
   `partition_figure`.
4. **Stabilité (§7.4).** Le multiselect de régimes adverses construit des
   `stress_bundle` ; on peut aussi charger un bundle de covariance alternative
   (Σ_daily restreinte aux piliers, Σ_bidask). `stable_cut(bundle, alternatives,
   alpha, kappa, epsilon, dendrogram)` (`clustering.py:366`) garde la coupe la plus
   profonde ≤ ε dont la partition passe le **test de variance dans *chaque* régime** et
   le plancher sous la covariance de base ; `undone` compte les fusions instables
   défaites. Table par régime + sauvegarde de `stability_pack` dans `session_state`.

## La raison derrière

- **Pourquoi découpler structure et niveau.** Une partition qui *bouge avec le book*
  est un signal d'alerte en validation : re-optimiser la combinatoire chaque jour avec
  les vegas du jour est suspect. La structure (Run 1) ne dépend que de Σ et est donc
  stable d'un portefeuille à l'autre ; seul le niveau (Run 2) est recalculé par book.
- **Pourquoi `d_ij` comme distance.** C'est le coût exact en variance résiduelle de
  netter deux nœuds (Th. 2 (i)) : agglomérer par `d_ij` croissant, c'est littéralement
  fusionner d'abord ce qui coûte le moins de fidélité.
- **Pourquoi la Propriété 8.** Si chaque nœud d'un ensemble vérifie
  `d_{j,pivot} ≤ ε·s_j`, alors pour **n'importe quel book** `TE ≤ ε·AVA_brut/κ`. Cela
  donne une garantie *portfolio-free* : la structure est défendable avant même de
  connaître le book. La métrique « borne Prop. 8 vs TE réelle » vérifie que la borne
  tient.
- **Pourquoi le contrôle par book et l'abaissement de coupe.** La garantie Prop. 8 est
  une majoration ; le test exact (6) sur le book du jour peut demander une coupe plus
  fine. On abaisse alors la coupe pour ce book — jamais on ne garde une coupe qui
  échoue.
- **Pourquoi la stabilité ne teste que le test de variance dans tous les régimes.** Le
  plancher (7) est un niveau prudentiel, qui n'a de sens que sous la covariance de base
  (Σ_daily n'est *pas* un niveau prudentiel, c'est l'estimateur de `ρ` le plus dense et
  un test de stabilité). Asymétrie : refuser une fusion valide est permis, conserver
  une fusion invalidée par un régime ne l'est pas.

## Comment lire l'onglet

- Le **slider ε** monte/descend dans l'arbre : plus ε est grand, moins d'ensembles
  (netting plus agressif). « ε réalisé » et « fusions défaites » indiquent si le
  contrôle par book a dû reculer.
- La **borne Prop. 8** doit être ≥ la TE réelle — sinon la garantie serait violée
  (elle ne l'est pas par construction).
- Dans la table de stabilité, une ligne « FAIL » sur un régime explique pourquoi des
  fusions ont été défaites ; « la structure est stable » signifie que toutes survivent.
