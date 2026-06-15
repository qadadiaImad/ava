# 🏛️ Théorie — fiche de documentation

> Onglet `with tab_theory:` dans `app.py`. **Aucun calcul** ne tourne ici : l'onglet
> est la *référence théorique* que tout le reste de l'application implémente.

## But de l'onglet

Poser, de la réglementation à l'algorithme, le cadre complet du netting des
sensibilités vega sous le test de variance EBA. C'est le contrat : chaque équation
affichée ici a une implémentation ailleurs dans le code. L'onglet sert de point
d'entrée pédagogique et de justification documentaire (art. 9(5)).

## Ce qui tourne à l'intérieur

Rien d'autre que des appels `st.markdown(...)` et `st.latex(...)`, disposés en deux
colonnes puis, après un `st.divider()`, deux colonnes de plus. Pas de fonction du
moteur `ebanetting`, pas d'état, pas de lecture de données. Le contenu est statique.

Les sections affichées et **leur implémentation réelle** :

| Section affichée | Contenu | Où c'est réellement calculé |
|---|---|---|
| §1 Cadre réglementaire | RTS 2016/101, art. 9(5) (netting) et art. 89 (90 %) | — (contexte) |
| §2 Modèle d'incertitude | ΔΠ = ⟨N, Δσ⟩, Var par Propriété 1, AVA add-up (2) / pleine diversification (3) | `ebanetting/model.py` — `UncertaintyModel` |
| §3 Passage aux nœuds, Th. 1 | sandwich Ñ = AᵀT·N·AK | `ebanetting/passage.py` — `project_bundle()` |
| §3.5 Grilles de chocs différentes | restriction sans perte (Prop. 5) | `passage.py` — `restriction_matrix()`, `offgrid_residual()` |
| §3b Hypothèse de découplage | ρ₂D = ρ_mat·ρ_strike terme à terme | `datasource.py` — `MarketDataBundle.covariance_of()` |
| §4 Netting comme opérateur, Déf. 4 | partition, m_r, choc représentatif | `ebanetting/netting.py` — `NettingScheme` |
| §5 Test de variance, Déf. 3, éq. 6 | TE² ≤ (1−α)·Var, R² ≥ α | `NettingScheme.evaluate()` |
| §6 Netting optimal, éq. 10 | problème NP-difficile | `ebanetting/optimizer.py` — `greedy_netting()` |
| Th. 2, éq. 8 | forme fermée à deux nœuds | `netting.py` — `two_bucket()` |
| §6.1 Diagnostic spectral, Th. 3 | facteurs de Σ, plancher de queue | `ebanetting/spectral.py` — `spectral_diagnostic()` |
| §6.2 Décomposition du smile, Th. 4 | niveau / RR / FLY | `ebanetting/scenario.py` — `smile_decomposition()` |
| §7.3 Algorithme glouton | budget, fusions rectangulaires | `optimizer.py` — `_greedy()` |
| §7.4 / §7.5 Architecture découplée, Prop. 8 | structure ⟂ niveau | `ebanetting/clustering.py` |
| §8 Points IPV | s, ρ, asymétrie, documentation, non-linéarité | (recommandations de mise en œuvre) |

## La raison derrière

- **Pourquoi un test plutôt qu'une formule.** L'art. 9(5) n'accorde pas la pleine
  diversification (éq. 3) par défaut : l'établissement doit *démontrer* que les
  positions se compensent réellement face à l'incertitude. Le test de variance est
  cette démonstration, appliquée à des agrégations partielles et interprétables.
- **Pourquoi deux extrêmes d'AVA.** L'add-up (éq. 2, κΣ|N|s, aucune compensation) et
  la pleine diversification (éq. 3, κ√Var) bornent tout résultat. L'écart entre les
  deux est le bénéfice de netting théorique maximal — l'enjeu économique de tout
  l'exercice.
- **Pourquoi pas de produit de Kronecker.** Sous l'hypothèse de découplage, chaque
  variance se calcule en forme matricielle (Propriété 1) avec des produits
  matriciels usuels et une réduction par somme. La covariance (MK × MK) n'est jamais
  assemblée — gain de coût, de mémoire et d'interprétabilité.
- **Pourquoi l'asymétrie réglementaire.** Le test protège contre la *sous*-estimation
  de l'AVA. Refuser une fusion valide est permis (conservateur) ; conserver une
  fusion invalidée ne l'est pas. Ce principe gouverne les stress et la stabilité.

## Comment lire l'onglet

C'est une lecture linéaire. Les numéros d'équation entre parenthèses (1)–(10) et les
références (Th. 1–4, Prop. 5–8, Déf. 1–4) sont repris **tels quels** dans les libellés
des métriques et des graphiques des autres onglets : c'est la clé de correspondance
entre la note technique et l'écran.
