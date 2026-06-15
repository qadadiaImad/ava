# 📋 Audit & Export — fiche de documentation

> Onglet `with tab_audit:` dans `app.py`. Le **dossier de preuve art. 9(5)** :
> un JSON empreinté, prêt pour l'archive de validation de modèle.

## But de l'onglet

Produire, en un téléchargement, tout ce que le validateur et le superviseur
demanderont : la partition retenue, le R² réalisé, les résultats de stress et la trace
gloutonne — relié par empreinte au bundle d'entrée exact, par date de calcul.

## Ce qui tourne à l'intérieur

1. **Source de l'évaluation** : un `st.radio` choisit « Schéma optimal » (reprend
   `optimal_eval` / `optimal_history` de `session_state`) ou « Scénario courant »
   (`scenario_eval`). Si l'évaluation n'existe pas encore, un message invite à visiter
   l'onglet correspondant.
2. **Résultats de stress** : si `robust_pack` est présent (mode robuste de l'onglet
   Netting optimal), on en extrait, par δ, le nombre d'ensembles, R², AVA et les
   verdicts.
3. **Assemblage du dossier** : `build_audit_pack(bundle, ev_a, kappa, stress_results,
   history)` (`ebanetting/reporting.py:28`) construit un dict :
   - `generated_at` (UTC), `regulation` (référence RTS) ;
   - `input` : `meta`, **empreinte SHA-256** (`_bundle_fingerprint`, 16 hex du JSON
     trié, `reporting.py:23`), grille ;
   - `parameters` : α, κ, pondération ;
   - `partition` : labels + nombre d'ensembles ;
   - `variance_test` : te2, budget, var_total, R² réalisé, `passes` ;
   - `conservatism_floor` : AVA nettée, plancher, `passes` ;
   - `ava` : add-up, nettée, pleine diversification, économie et % ;
   - `sets` : statistiques par ensemble ;
   - optionnels : `correlation_stress`, `greedy_history`.
   Le `stability_pack` (onglet Structure) est greffé sous `stability_6_4` s'il existe.
4. **Sortie** : `audit_json(pack)` (`reporting.py:105`) sérialise en JSON indenté ;
   boutons de téléchargement (dossier + bundle d'entrée), métrique d'empreinte, et
   `st.json(pack)` pour l'inspection.

## La raison derrière

- **Pourquoi un dossier de preuve.** Le RTS 2016/101 exige une preuve *documentée* que
  les expositions nettées se compensent réellement face à l'incertitude (art. 9(5)).
  Sans cette piste d'audit, le bénéfice de netting n'est pas opposable au superviseur.
- **Pourquoi une empreinte SHA-256.** Elle lie le résultat au bundle d'entrée exact :
  reproductibilité et détection de toute altération. Deux dossiers d'empreintes
  différentes ne portent pas sur les mêmes données.
- **Pourquoi conserver stress et trace.** La note (§8/§9) recommande d'archiver
  {partition retenue, R² réalisé, frontière, résultats de stress} *par date de calcul* :
  c'est exactement le contenu du dossier, de sorte qu'une revue ultérieure peut rejouer
  et contester chaque fusion.
- **Pourquoi deux sources (optimal / scénario).** On peut archiver soit la solution de
  l'optimiseur, soit le schéma effectivement appliqué par le desk — selon ce qui est
  soumis à validation.

## Comment lire l'onglet

- Vérifiez que **les verdicts** (`variance_test.passes`, `conservatism_floor.passes`)
  sont à `true` avant d'archiver.
- L'**empreinte** doit correspondre au bundle téléchargé à côté : c'est le lien de
  preuve.
- La section `greedy_history` (si présente) retrace chaque fusion avec son gain d'AVA
  et son coût en variance — la justification pas-à-pas de la partition.
