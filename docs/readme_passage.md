# 🔁 Passage — fiche de documentation

> Onglet `with tab_passage:` dans `app.py`. Implémente l'**étape 1** de la note
> (§3) : transporter le vega granulaire sur les piliers du consensus.

## But de l'onglet

L'incertitude (`s`, `ρ`) n'est observable qu'à quelques piliers (dispersions Totem,
fourchettes brokers). Le vega vit sur une grille plus fine. L'onglet transporte le
vega vers les piliers, vérifie qu'aucune masse n'est perdue, et mesure la *tracking
error* induite par le choix de convention de passage.

## Ce qui tourne à l'intérieur

L'activation se fait par le toggle « Projeter le vega granulaire… » de la barre
latérale, qui appelle `project_bundle(bundle, pillar_t, pillar_k, convention)`
(`ebanetting/passage.py:119`). Tout l'aval de l'application tourne ensuite sur la
grille de nœuds projetée.

`project_bundle` fait, dans l'ordre :

1. Construit, **pour les trois conventions**, les matrices de passage 1-D par axe via
   `passage_matrix` (`passage.py:54`) — lignes positives sommant à 1 (Déf. 1) :
   - `interp` : poids d'interpolation linéaire,
   - `equal` : ½/½ sur les deux piliers encadrants,
   - `quadrant` : tout le poids sur le pilier le plus proche.
2. Applique le **sandwich 2-D** `project` (`passage.py:94`) : `Ñ = A_Tᵀ · N · A_K` —
   deux produits matriciels ordinaires.
3. **Restreint** `s` et les corrélations aux indices des piliers (`np.ix_`) : c'est là
   que l'incertitude est mesurée. Construit le `node_bundle`.
4. Calcule la **tracking error de chaque convention** contre les poids d'interp
   (éq. 5) : `E = B_Tᵀ N B_K − A_Tᵀ N A_K`, puis `TE² = Var(⟨E, Δσ_pil⟩)` —
   stockée dans `te2_by_convention`. Pour `interp`, `TE² = 0` exactement (Théorème 1).

L'onglet affiche ensuite :
- **4 métriques** : nombre de nœuds, conservation du vega (Prop. 3 :
  `Σ N == Σ Ñ`, propriété `conserves_vega`), `TE²_passage` (éq. 5) et sa part de
  budget, convention retenue. Un `st.warning` si la convention n'est pas `interp`.
- **2 heatmaps** : vega granulaire vs vega projeté.
- **un tableau** comparant la TE² des trois conventions et un drapeau « sans perte ».
- **un expander** affichant les matrices `A_T` et `A_K` (lignes sommant à 1).

> §3.5 — quand les chocs sont *plus fins* que les sensis (variations quotidiennes de
> la surface système), le module fournit aussi `restriction_matrix`, `restrict_shocks`
> et `offgrid_residual` (Prop. 5, Remarque 4). Ces fonctions ne sont pas câblées dans
> l'onglet mais font partie du même module `passage.py`.

## La raison derrière

- **Pourquoi un passage du tout.** On ne peut tester le netting que là où
  l'incertitude est observée. Le sandwich `A_Tᵀ N A_K` (Déf. 2) amène le vega aux
  piliers sans jamais assembler un opérateur (MK × MK).
- **Théorème 1 — pourquoi `interp` est sans perte.** Quand les poids de passage
  coïncident avec les poids d'interpolation de la *construction de la surface de
  pricing*, le passage ne détruit aucune information au premier ordre :
  `TE_passage = 0`. Toute autre convention (quadrant, équipondérée) crée une erreur
  quantifiable qui **consomme le budget du test de variance avant tout netting**.
- **Remarque 3 — pourquoi les matrices ne sont pas optimisées.** Elles sont *fixées*
  par la construction de la surface, pas un degré de liberté. Seule la partition en
  aval est optimisée. C'est pourquoi l'onglet *mesure* la TE de passage au lieu de la
  *minimiser*.
- **Pourquoi un avertissement sur les conventions non-interp.** La TE de passage est
  une perte sèche de budget : il faut la documenter et la déduire (note §8), sinon le
  netting apparaît plus permissif qu'il ne l'est.

## Comment lire l'onglet

- Si **conservation = OK** et **TE²_passage = 0** (convention `interp`), le passage est
  neutre : la suite travaille sur une grille plus petite sans coût.
- Si vous changez de convention, regardez la **part de budget** consommée : c'est
  autant de marge en moins pour le netting réel dans les onglets suivants.
- Les matrices `A_T` / `A_K` permettent d'auditer exactement comment chaque bucket fin
  a été réparti sur les piliers.
