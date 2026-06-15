# 🧬 Deux couches — fiche de documentation

> Onglet `with tab_twolayer:` dans `app.py`. La **méthodologie compagnon** :
> sous-jacent (structure) × book (évaluation), avec le résidu signé *exact* qui peut
> autoriser ce que le majorant de couche 1 refusait.

## But de l'onglet

Travailler une tranche (une ligne de maturité) en deux temps. **Couche 1** (par
sous-jacent, sans book) : le smile bouge selon trois modes — niveau, pente ΔS,
courbure ΔC — plus un bruit idiosyncratique ; cela engendre une carte des fusions
possibles. **Couche 2** (par book) : le résidu *exact* du netting d'un groupe sur son
pivot, porté par les agrégats signés RR et FLY, qui décide collapse / extraction /
scission.

## Ce qui tourne à l'intérieur

1. **Modèle de couche 1, éditable.** `SmileModel(s0, σ_S, σ_C, σ_ε)`
   (`ebanetting/twolayer.py:76`). `x = strikes − 1` (offset de moneyness),
   `nu = vega[tranche]`.
2. **Carte de désaccord.** `model_distance(x, smodel)` (`twolayer.py:124`) — Théorème 1 :
   `d_ij² = (x_j−x_i)²σ_S² + (x_j−x_i)²(x_j+x_i)²σ_C² + 2σ_ε²`. C'est l'**écart**
   `|x_j − x_i|` qui compte (pas la distance à l'ATM) ; les ailes amplifient la
   courbure ; l'idio est un plancher.
3. **Courbe de corrélation générée.** `generated_correlation(x, smodel)` (`twolayer.py:148`)
   = `s0 / s_x` (Propriété 2) : la corrélation avec l'ATM n'est pas une donnée
   primitive, le modèle la fabrique.
4. **Dendrogramme de tranche.** `tranche_dendrogram(x, dist, s)` (`twolayer.py:164`) :
   groupes de strikes contigus par désaccord pivot-à-pivot croissant, le pivot d'un
   groupe étant son point le plus liquide (plus petit `s`). Slider ε →
   `cut_tranche(merges, K, ε)` → groupes.
5. **Couche 2 — décisions book.** `evaluate_book(groups, nu, x, smodel, alpha, kappa)`
   (`twolayer.py:378`) :
   - tous les groupes démarrent en **collapse** sur leur pivot barycentre
     (`barycenter_pivot` `x_p = RR_0/m` annule RR exactement, ramené au nœud coté le
     plus proche) ;
   - tant que la TE² **globale** exacte (`_global_te2` : RR et FLY *sommés* entre
     groupes collapsés *avant* d'être élevés au carré, + idio par groupe) dépasse le
     budget → **extraction** du pire groupe (m reste netté, RR/FLY provisionnés en
     add-up avec `s_skew`/`s_fly`) ;
   - si l'idio seul déborde encore → **scission** du pire groupe.
   - Retourne les `decisions`, `te2`, `te2_majorant` (`(Σ|ν_j| d_jp)²`), AVA, plancher.
6. **Affichage** : heatmap de désaccord, courbe de corrélation, dendrogramme ; 5
   métriques (groupes, TE² exact vs majorant, budget, AVA, verdicts) ; un `st.success`
   quand `te2_majorant > budget ≥ te2` (le calcul exact autorise ce que la couche 1
   refusait) ; la table des décisions par groupe.

## La raison derrière

- **Pourquoi séparer désaccord et RR/FLY.** Le désaccord entre points de vol est une
  propriété de l'underlying (il ignore le book) ; RR et FLY sont la *projection du
  book* sur cette structure. Le même dendrogramme sert tous les books du sous-jacent.
- **Théorème 2 — pourquoi exact et pas borne.** Les modes communs (pente, courbure)
  entrent par des sommes **signées** : ils peuvent se compenser. L'idio entre par des
  **carrés** : il ne se compense jamais, c'est le plancher incompressible. La variance
  du résidu est donc une forme fermée, pas une inégalité.
- **Pourquoi le pivot barycentre.** `x_p = RR_0/m` annule exactement le terme de pente
  (RR_p = RR_0 − m·x_p) : le bon choix de représentation supprime une source de résidu
  sans rien approximer.
- **Pourquoi jamais tout-ou-rien (N1–N4).** Plutôt que d'accepter ou rejeter un groupe
  entier, on garde le niveau net netté et on *extrait* seulement la forme (RR/FLY) qui
  pose problème, en la provisionnant avec sa propre incertitude. Seul le netting
  réellement injustifié est abandonné.
- **Pourquoi exact vs majorant.** Le majorant de couche 1, `Σ|ν_j| d_jp` (inégalité
  triangulaire), efface les signes : il **compte deux fois** ce qui s'annule. Le calcul
  exact peut donc autoriser une fusion que le majorant refusait — jamais l'inverse.

## Comment lire l'onglet

- La **heatmap de désaccord** montre quels strikes peuvent se grouper (faible `d`) ;
  la **courbe de corrélation** illustre la « falaise » de courbure dans les ailes.
- **TE² exact vs majorant** : si l'exact est nettement plus bas, le book bénéficie de
  compensations réelles que la borne ne voit pas — d'où l'éventuel message vert.
- La table des décisions indique, par groupe, le niveau net `m`, RR_p, FLY_p, la
  Var(R) exacte, le majorant, et la décision (collapse / extract / split).
