# 🔬 Spectral & Smile — fiche de documentation

> Onglet `with tab_spectral:` dans `app.py`. Le **diagnostic spectral** (§6.1 — le
> spectre mesure, la partition nette) et la **décomposition du smile** par tranche
> (§6.2).

## But de l'onglet

Diagnostiquer le book *avant* de partitionner : combien de modes de risque
indépendants porte-t-il, lesquels sont compressibles, et quel est le nombre minimal
d'ensembles de netting que la structure autorise. Puis, par tranche, séparer ce qui se
nette parfaitement (le niveau) de ce qui porte le résidu (RR / FLY).

## Ce qui tourne à l'intérieur

1. **Diagnostic spectral.** `spectral_diagnostic(model, alpha)`
   (`ebanetting/spectral.py:95`) :
   - assemble `Σ_pil` via `node_covariance` (`spectral.py:53`) = `D ρ D` (paramétrage
     découplé terme à terme, ou `corr_full`) — petite matrice (q × q), pas de Kronecker ;
   - diagonalise (`np.linalg.eigh`), trie les valeurs propres `λ_l` décroissantes ;
   - `c_l = ⟨ν, u_l⟩` (projection du vega, Prop. 7), `loadings2 = c_l²·λ_l` (la **carte
     de risque**, Prop. 7) ;
   - `residual_curve[L] = Σ_{l>L} c_l²·λ_l` — le **plancher du Théorème 3** avec L
     facteurs ;
   - `inertia[L] = τ_L` (compressibilité, U1) ; `k_star = K*(α)` = plus petit L tel que
     le plancher ≤ budget.
2. **4 métriques** : `K*(α)` (plancher spectral sur le nombre d'ensembles), part de
   variance expliquée par les 3 premiers modes, inertie `τ₃`, ratio `λ₁/λ₂`.
3. **`spectral_figure`** : barres de la carte de risque `c_l²λ_l` + courbe de la part
   résiduelle (Th. 3) avec la ligne de budget (1−α) et la verticale `K*(α)`.
4. **3 directions propres dominantes** `u_l` affichées sur la grille via `diag.mode`
   (heatmaps « niveau », « terme/skew », « smile »).
5. **Décomposition du smile.** `smile_decomposition(bundle)` (`scenario.py:90`) : par
   maturité, dans la base `{1, x, x²}` (x = K/F − 1), `level = Σν`, `RR = ν·x`,
   `FLY = ν·x²` ; `te2_line` est le résidu *exact* de l'écrasement de la tranche sur son
   pivot ATM sous `Σ_m = D ρ_strike D`, et `r2_line` le R² par tranche.
   `smile_decomposition_figure` : barres niveau/RR/FLY + courbe du R² par tranche vs α.

## La raison derrière

- **Pourquoi diagonaliser Σ.** Les facteurs `ξ_l = u_lᵀΔσ` sont décorrélés, de
  variances `λ_l` (Prop. 6) ; la variance du book se décompose exactement en
  `Σ c_l²λ_l` (Prop. 7). C'est la « carte de risque » : on voit *où* est le risque, pas
  seulement combien.
- **Théorème 3 — pourquoi un plancher.** Tout schéma dont les chocs représentatifs
  vivent dans le sous-espace des L premiers facteurs (description réaliste des schémas
  grossiers à poids lisses) vérifie `TE² ≥ Σ_{l>L} c_l²λ_l` : la variance de queue est
  incompressible. `K*(α)` est donc un **nombre minimal d'ensembles** en deçà duquel le
  test ne peut pas passer — un diagnostic, avant toute optimisation.
- **La limite, essentielle.** Un facteur n'est **pas** un ensemble de netting : ses
  poids signés denses ne sont pas une exposition de valorisation documentable. Le
  spectre *mesure* la compressibilité ; c'est la partition (rectangulaire,
  interprétable) qui *nette*. L'onglet ne propose donc jamais « nettez selon le
  facteur 1 ».
- **Théorème 4 — pourquoi décomposer le smile.** Le niveau net `m_a` se nette
  parfaitement quelle que soit sa taille ; tout le résidu de l'écrasement est porté par
  RR_a et FLY_a. Une tranche dont le `r2_line` tombe sous α porte un RR/FLY net
  matériel : sa forme de smile doit rester en add-up (étape 1 du netting hiérarchique),
  les niveaux des tranches qui passent allant ensuite au netting de structure par terme.

## Comment lire l'onglet

- **`K*(α)`** : si votre partition (onglets précédents) a moins d'ensembles que `K*`,
  elle ne peut mathématiquement pas passer le test — inutile de chercher, il faut
  raffiner.
- La **carte de risque** (barres) montre les modes coûteux ; la **part résiduelle**
  croisant la ligne de budget en `K*` indique combien de facteurs il faut « expliquer ».
- Les **3 heatmaps** de modes propres lisent le risque comme niveau, terme/skew, smile.
- Sur la décomposition du smile, les tranches sous la ligne α (R²) sont celles à garder
  en add-up sur leur forme.
