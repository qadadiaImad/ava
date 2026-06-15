# 📊 Données de marché — fiche de documentation

> Onglet `with tab_data:` dans `app.py`. Affiche les **entrées** du problème et
> l'**enveloppe d'AVA**, puis documente le contrat de données.

## But de l'onglet

Montrer les quatre objets qui suffisent à tout le framework — la surface de vega `N`,
la surface d'incertitude `s`, et les corrélations `ρ_mat` / `ρ_strike` — et encadrer
le résultat entre ses deux bornes : l'AVA add-up (aucun netting) et l'AVA pleinement
diversifiée. C'est le tableau de bord d'entrée et le point de branchement des données
de la banque.

## Ce qui tourne à l'intérieur

1. **Construction du modèle.** `UncertaintyModel(bundle=bundle, kappa=float(kappa))`
   (`ebanetting/model.py:26`). C'est un wrapper immuable qui précalcule les formes
   quadratiques.
2. **Les quatre KPI** (`k1..k4.metric`) :
   - `model.var_total` — Var(ΔΠ) via `bundle.variance_of(vega)` →
     `covariance_of` (`datasource.py:127`), c'est-à-dire la **Propriété 1** en forme
     sandwich : `Σ( (V∘s) * (corr_mat @ (V∘s) @ corr_strike) )`. Aucun Kronecker.
   - `model.ava_brut` — éq. (2), `κ·Σ|N_mk|·s_mk` (`model.py:46`).
   - `model.ava_full` — éq. (3), `κ·√Var(ΔΠ)` (`model.py:50`).
   - **Bénéfice de netting max** = `ava_brut − ava_full`, avec le pourcentage de
     l'add-up que cela représente.
3. **Quatre heatmaps** (`ui/charts.py` — `heatmap`) :
   - vega `N` (diverging RdBu, signée),
   - incertitude `s` (Plasma),
   - `ρ_mat` (corrélation entre maturités) et `ρ_strike` (entre moneyness), en Sunset ;
     si le bundle est fourni en `corr_full`, c'est cette matrice (MK × MK) qui s'affiche.
4. **Contrat de données** (expander) : schéma JSON commenté + bouton de téléchargement
   du bundle courant comme gabarit (`bundle.to_json(indent=2)`).

En amont, dans la barre latérale, `bundle.validate()` (`datasource.py:155`) a déjà
rejeté tout bundle invalide (formes, `s > 0`, corrélations symétriques, diagonale
unité, semi-définies positives).

## La raison derrière

- **Pourquoi ces entrées et pas d'autres.** La covariance Σ aux nœuds est l'unique
  input statistique du framework (note §4.1). Tout le reste — test, AVA, structure —
  en dérive. La surface de vega est l'exposition du book ; `s` et `ρ` décrivent
  l'incertitude du consensus.
- **Pourquoi corrélations découplées par axe.** L'annexe pose
  `ρ[(a,b),(a',b')] = ρ_mat[a,a']·ρ_strike[b,b']` : seulement `C(qT,2)+C(qK,2)`
  paramètres au lieu de `C(qT·qK,2)`, chacun interprétable (corrélation de maturité,
  corrélation de moneyness) et stressable indépendamment. `corr_full` reste possible
  et prend la priorité.
- **Pourquoi le moneyness K/F (et pas le strike absolu).** En strike fixe, les points
  dérivent en moneyness avec le spot : la corrélation de l'axe strike est gonflée et
  le netting est **sur-justifié** — un biais non conservateur (note §8). D'où la
  recommandation d'estimer `s, ρ` en coordonnées K/F.
- **Pourquoi gonfler `s` quand peu de contributeurs.** `s` est la dispersion
  inter-contributeurs Totem ; avec peu de contributeurs, il faut l'**élargir**, pas le
  lisser — sinon l'incertitude est sous-estimée (asymétrie réglementaire).
- **Pourquoi les deux bornes d'AVA.** Elles cadrent visuellement le gain disponible :
  aucun schéma de netting ne peut descendre l'AVA sous `ava_full` (c'est le plancher
  de conservatisme, éq. 7), ni la faire monter au-dessus de `ava_brut`.

## Comment lire l'onglet

- Le **bénéfice de netting max** est le « prix » maximal que le netting peut faire
  gagner ; les onglets Netting optimal et Structure indiquent quelle fraction est
  réellement atteignable sous contrainte.
- Sur la heatmap de vega, les zones de signes opposés voisines (rouge/bleu) sont les
  candidats naturels à la compensation ; la heatmap de `s` montre où l'incertitude est
  large (ailes, maturités courtes, longue échéance) — donc où le netting est cher.
- La heatmap `ρ` montre la portée de la dépendance : plus la décroissance est lente,
  plus de buckets pourront se netter.
