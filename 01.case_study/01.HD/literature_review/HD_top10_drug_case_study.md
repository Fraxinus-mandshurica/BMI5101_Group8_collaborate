# Huntington Disease: DREAMwalk Top 10 Drug Case Study

Literature search and assessment date: 22 October 2026. Methods and Discussion updated: 28 October 2026. Target: Huntington disease (HD), MSI CUI `C0020179`.

## 1. Main findings

Among the current top ten candidates, **amantadine, clonazepam, and diazepam have treatment studies involving HD patients; lorazepam has HD prescribing data and reports of catatonia management; phenobarbital has a juvenile-HD seizure case; and orphenadrine has preclinical evidence from an HD-related toxin model.** Evidence located for the other four drugs is mainly from general pharmacology or related drug classes.

The case study supports the following interpretation: **the model ranked several drugs with evidence relevant to symptomatic HD management and generated hypotheses for further investigation. These findings do not establish disease-modifying efficacy.** Negative and conflicting results are retained below. General pharmacology is not counted as clinical validation in HD.

Three candidates are particularly useful for the main discussion:

- **Amantadine:** the most extensive direct HD trial evidence in this list, with conflicting objective chorea results. It illustrates recovery of a previously investigated drug. [R06](https://pubmed.ncbi.nlm.nih.gov/12221159/) [R07](https://pubmed.ncbi.nlm.nih.gov/12873857/) [R08](https://pubmed.ncbi.nlm.nih.gov/12821751/) [R09](https://pubmed.ncbi.nlm.nih.gov/15304616/)
- **Clonazepam:** HD cases, guidelines, and prescribing data support a symptom-specific interpretation, particularly for myoclonus. [R22](https://pubmed.ncbi.nlm.nih.gov/131236/) [R24](https://pubmed.ncbi.nlm.nih.gov/1291167/) [R02](https://pubmed.ncbi.nlm.nih.gov/31333565/) [R01](https://pubmed.ncbi.nlm.nih.gov/39431460/)
- **Orphenadrine:** NMDA receptor pharmacology and results in 3-nitropropionic acid (3-NPA) models support a preclinical repurposing hypothesis that remains to be tested. [R28](https://pubmed.ncbi.nlm.nih.gov/11159722/) [R29](https://pubmed.ncbi.nlm.nih.gov/10340304/) [R30](https://pubmed.ncbi.nlm.nih.gov/8788072/)

## 2. Prediction run and interpretation

### 2.1 Why Huntington disease was selected

HD is a rare neurodegenerative disease, which matches the project's focus on rare-disease drug repurposing. Its motor, cognitive and psychiatric symptoms also provide a clinical setting in which the purpose of a proposed association must be stated clearly. [R02](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2019.00710/full)

HD is already represented in the accepted MSI inputs as `C0020179`, with an existing DREAMwalk embedding and molecular graph connections. The case study can therefore use the completed model without adding a new disease or rebuilding the graph. Published treatment studies and clinical guidelines provide evidence against which the candidates can be assessed, including conflicting and symptom-specific findings.

HD was selected as a practical, literature-supported case study. It is not an unseen disease in the current classifier: all five known MSI HD-drug positives participate in training. Their high training scores are not a reason to claim successful generalization.

### 2.2 Prediction protocol

This report uses the [current top-ten snapshot](../../../reference_results/hd_top10_candidates.csv), rather than the top four from the earlier HD-label-holdout workflow.

The current workflow uses:

1. One fixed set of corrected DREAMwalk MSI embeddings: MeSH2024, embedding seed 0, tau (`τ`) = 0.3, and `pp=1`, with first-hit teleportation, positive-output-type negative sampling (NS), and current-positive-output collision filtering.
2. All 5,926 known drug-disease association (DDA) positives, **including the five known HD positives**, and 5,926 link-level negatives for each XGBoost classifier.
3. Ten pairwise-disjoint negative sets generated with seeds 0–9. The classifier training seed remains 0. Embeddings are not retrained.
4. Predictions for all 1,661 drugs. After excluding the five known MSI HD positives, 1,656 candidates are ranked by their mean score across the ten classifiers. Sample SD uses `ddof=1`.

**The mean is a ranking score, not a calibrated probability of clinical efficacy. SD reflects variation across these ten link-negative sets only.** This is neither ten-fold cross-validation nor ten embedding-training runs. None of the current top-ten pairs was sampled as an HD training negative in these ten sets.

Here, a *novel candidate* means a pair absent from the current MSI HD positive labels. Amantadine, for example, already has HD publications and is not a new discovery in the literature. This assessment is post hoc literature corroboration and does not produce a new held-out performance estimate.

### 2.3 Ranking and evidence overview

| Rank | Drug (DrugBank ID) | Mean ± SD | Most directly relevant evidence located | Appropriate interpretation |
| --- | --- | --- | --- | --- |
| 1 | Amantadine (DB00915) | 0.8945 ± 0.0542 | HD randomized trials with conflicting results | Symptomatic chorea management |
| 2 | Prazepam (DB01588) | 0.8833 ± 0.0655 | Anxiety trials outside HD; BZD-class context in HD | Anxiety hypothesis; no drug-specific HD efficacy evidence located |
| 3 | Lorazepam (DB00186) | 0.8807 ± 0.0618 | HD prescribing cohort and catatonia case synthesis | Specific neuropsychiatric symptoms |
| 4 | Secobarbital (DB00418) | 0.8796 ± 0.0549 | Drug-specific GABA receptor experiments | CNS pharmacology; direct HD efficacy evidence is lacking |
| 5 | Diazepam (DB00829) | 0.8791 ± 0.0623 | Early HD treatment reports and a small study | Symptomatic chorea/anxiety management |
| 6 | Pentobarbital (DB00312) | 0.8790 ± 0.0601 | Drug-specific GABA receptor experiments | CNS pharmacology; direct HD efficacy evidence is lacking |
| 7 | Clonazepam (DB01068) | 0.8768 ± 0.0576 | HD cases, guidelines, and prescribing cohort | Myoclonus and some choreiform symptoms |
| 8 | Phenobarbital (DB01174) | 0.8667 ± 0.0626 | A single juvenile-HD case | Refractory seizures/myoclonus; not generalizable to adult HD |
| 9 | Benperidol (DB12867) | 0.8634 ± 0.0425 | D2-blockade pharmacology, a non-HD review, and class-level HD context | Antidopaminergic hypothesis; drug-specific HD evidence is lacking |
| 10 | Orphenadrine (DB01173) | 0.8583 ± 0.0494 | Cell and rat studies of 3-NPA neurotoxicity | Preclinical hypothesis related to excitotoxicity |

Scores are close. Their ordering alone cannot establish that, for example, rank 2 is more suitable for clinical use than rank 7. The discussion below is organized by evidence content rather than clinical preference inferred from rank.

## 3. Drug-by-drug assessment

### 3.1 Amantadine — Rank 1

**Mechanism and relevance to HD.** Amantadine has NMDA receptor antagonist activity. Glutamatergic excitotoxicity provides a rationale for its investigation in HD, but mechanistic plausibility and clinical outcomes require separate assessment.

**Direct evidence.** The crossover trial by Verhagen Metman et al. enrolled 24 patients with HD, of whom 22 were evaluable. The median reduction in extremity chorea at rest was 36% relative to placebo (`p=0.04`). A separate crossover trial in 24 participants found no average objective improvement in chorea, although subjective outcomes and quality of life were better. Lucetti et al. reported improved chorea in an acute IV study of nine patients. A subsequent small crossover study reported in correspondence by Heckmann et al. enrolled eight patients, with six completing both arms, and did not show consistent motor benefit. [R06](https://pubmed.ncbi.nlm.nih.gov/12221159/) [R07](https://pubmed.ncbi.nlm.nih.gov/12873857/) [R08](https://pubmed.ncbi.nlm.nih.gov/12821751/) [R09](https://pubmed.ncbi.nlm.nih.gov/15304616/)

**Interpretation.** Amantadine is the strongest candidate in this list for comparison with clinical HD literature. The defensible claim is that the model recovered a drug already investigated in HD, rather than that multiple RCTs consistently established efficacy. The 2023 guideline and 2026 expert consensus also discuss it within symptomatic management. [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) [R05](https://pubmed.ncbi.nlm.nih.gov/42446188/)

**References for this drug (6):**

- [R06](https://pubmed.ncbi.nlm.nih.gov/12221159/) — HD crossover trial supporting short-term chorea benefit.
- [R07](https://pubmed.ncbi.nlm.nih.gov/12873857/) — HD crossover trial with a negative objective chorea outcome but better subjective outcomes.
- [R08](https://pubmed.ncbi.nlm.nih.gov/12821751/) — Small acute IV HD trial.
- [R09](https://pubmed.ncbi.nlm.nih.gov/15304616/) — Correspondence reporting a subsequent small HD study and conflicting findings.
- [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) — HD guideline providing clinical context.
- [R05](https://pubmed.ncbi.nlm.nih.gov/42446188/) — 2026 HD expert consensus providing symptomatic-use context.

### 3.2 Prazepam — Rank 2

**Mechanism and relevance to HD.** Prazepam is a benzodiazepine (BZD) that enhances GABAergic inhibition. Symptoms such as anxiety provide a reason to consider this class in HD, but do not establish prazepam-specific efficacy.

**Evidence located.** Weir's multicenter randomized trial included 847 patients with anxiety and reported superiority over placebo. Fabre et al. compared prazepam, diazepam, chlorazepate, and placebo, with results differing across anxiety scales. **These were not HD cohorts.** An additional anxiety trial by Rickels et al. was published in 1977. Its bibliographic record was checked, but the complete results were not reviewed and are not used here to establish efficacy. [R11](https://pubmed.ncbi.nlm.nih.gov/721788/) [R12](https://pubmed.ncbi.nlm.nih.gov/39012/) [R10](https://pubmed.ncbi.nlm.nih.gov/858240/)

**Interpretation.** No prazepam-specific HD therapeutic study was located in this search. It is an indirect candidate with general anxiolytic evidence, which should be distinguished from amantadine's direct HD trials. The HD expert consensus provides anxiety/sleep-management context but does not specifically validate prazepam. [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/)

**References for this drug (4):**

- [R11](https://pubmed.ncbi.nlm.nih.gov/721788/) — Prazepam anxiety RCT outside HD.
- [R12](https://pubmed.ncbi.nlm.nih.gov/39012/) — Comparison with other BZDs and placebo outside HD.
- [R10](https://pubmed.ncbi.nlm.nih.gov/858240/) — Original anxiety-trial record; no outcome extracted in this assessment.
- [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/) — HD neuropsychiatric consensus, used only for class and symptom context.

### 3.3 Lorazepam — Rank 3

**Mechanism and relevance to HD.** Lorazepam is a BZD. Its HD association mainly concerns neuropsychiatric symptom management, including rare cases of catatonia. It is not an established neuroprotective treatment for HD.

**Directly relevant evidence.** The Enroll-HD observational study by Feleus et al. included 8,546 participants with manifest HD and reported lorazepam as the most frequently prescribed BZD in Europe. This documents use, **not a treatment-effect estimate**. A 2026 scoping review identified three articles describing four HD catatonia cases. All received lorazepam, with variable responses, and all also received electroconvulsive therapy (ECT). Outcomes under combined treatment cannot be attributed to lorazepam alone. [R01](https://pubmed.ncbi.nlm.nih.gov/39431460/) [R13](https://pubmed.ncbi.nlm.nih.gov/42155211/)

The original juvenile/Westphal-HD case reported by Merida-Puga et al. also showed that standard BZD/ECT treatment could be insufficient and require additional medications. Bush et al.'s prospective catatonia protocol supports lorazepam's general use for catatonia, but **was not an HD-specific trial**. [R14](https://pubmed.ncbi.nlm.nih.gov/22123587/) [R32](https://pubmed.ncbi.nlm.nih.gov/8686484/)

**Interpretation.** Lorazepam is a candidate supported by prescribing data and symptom-specific reports in HD. These sources do not establish improvement in HD chorea or disease progression. The HD psychiatric consensus also cautions against broad long-term BZD use for sleep solely because of sedative effects. [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/)

**References for this drug (5):**

- [R01](https://pubmed.ncbi.nlm.nih.gov/39431460/) — Enroll-HD cohort documenting prescribing practice.
- [R13](https://pubmed.ncbi.nlm.nih.gov/42155211/) — HD catatonia scoping review: four cases with variable responses.
- [R14](https://pubmed.ncbi.nlm.nih.gov/22123587/) — Original HD catatonia case illustrating the limits of combined and augmented treatment.
- [R32](https://pubmed.ncbi.nlm.nih.gov/8686484/) — Prospective lorazepam/ECT catatonia study, not specific to HD.
- [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/) — HD neuropsychiatric consensus.

### 3.4 Secobarbital — Rank 4

**Mechanism and relevance to HD.** Secobarbital is a barbiturate. Iadarola et al. observed enhanced GABA receptor-coupled responses in biochemical and hippocampal-slice preparations. Peters et al. also showed that secobarbitone, another name for the same drug, enhanced GABA-evoked currents. [R15](https://pubmed.ncbi.nlm.nih.gov/2981311/) [R16](https://pubmed.ncbi.nlm.nih.gov/2850060/)

**Evidence boundary.** These are drug-specific pharmacology experiments, not studies in HD patients or mutant-HTT models. No direct HD therapeutic trial or clear HD case report for secobarbital was located. Pharmacological proximity to other CNS drugs may help explain its ranking, but this is an interpretation hypothesis that has not been tested with model attribution.

**Interpretation.** Secobarbital should remain a mechanistically plausible candidate without direct HD efficacy support. General sedation cannot be presented as evidence of anti-chorea or disease-modifying activity. The HD sleep/neuropsychiatric consensus supplies symptom-management context and does not establish efficacy for this drug. [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/)

**References for this drug (3):**

- [R15](https://pubmed.ncbi.nlm.nih.gov/2981311/) — Direct comparison of GABA-related effects of secobarbital and other barbiturates, outside HD.
- [R16](https://pubmed.ncbi.nlm.nih.gov/2850060/) — Secobarbitone receptor electrophysiology and binding, outside HD.
- [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/) — HD symptom-management context; no evaluation of secobarbital efficacy.

### 3.5 Diazepam — Rank 5

**Direct evidence.** *Valium therapy of Huntington's chorea* was published in 1964. The original record was verified, but no abstract was available; response rates are not inferred from its title. Caraceni et al. compared several drugs in 11 patients with HD and reported improvement in abnormal involuntary movements with diazepam. The small sample and older clinical assessment methods limit the strength of this evidence. [R17](https://pubmed.ncbi.nlm.nih.gov/14154759/) [R18](https://pubmed.ncbi.nlm.nih.gov/6210644/)

**Mechanism and clinical context.** Patch-clamp experiments support diazepam's modulation of GABA receptor currents. The HD guideline and 2026 expert consensus provide context for anxiety management or adjunct symptomatic use in selected situations. This does not establish an effect on disease progression. [R19](https://pubmed.ncbi.nlm.nih.gov/2471436/) [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) [R05](https://pubmed.ncbi.nlm.nih.gov/42446188/)

**Interpretation.** Diazepam has a direct link to HD treatment literature and fits the symptomatic-rediscovery group. Its clinical evidence here is mainly from early and small studies.

**References for this drug (5):**

- [R17](https://pubmed.ncbi.nlm.nih.gov/14154759/) — Early diazepam/Valium treatment report in HD; bibliographic verification only.
- [R18](https://pubmed.ncbi.nlm.nih.gov/6210644/) — Pharmacologic study in 11 HD patients, with limited assessment methods.
- [R19](https://pubmed.ncbi.nlm.nih.gov/2471436/) — Diazepam/phenobarbital GABA receptor-channel comparison, outside HD.
- [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) — HD guideline.
- [R05](https://pubmed.ncbi.nlm.nih.gov/42446188/) — HD chorea expert consensus providing adjunct-use context.

### 3.6 Pentobarbital — Rank 6

**Mechanistic evidence.** Several experiments directly examined pentobarbital, also called pentobarbitone. Iadarola et al. studied GABA-mediated inhibition; Peters et al. studied GABA-evoked currents; MacDonald et al. examined changes in GABA_A channel gating; and Rho et al. distinguished enhancement of GABA responses from direct receptor activation at higher concentrations. [R15](https://pubmed.ncbi.nlm.nih.gov/2981311/) [R16](https://pubmed.ncbi.nlm.nih.gov/2850060/) [R20](https://pubmed.ncbi.nlm.nih.gov/2482885/) [R21](https://pubmed.ncbi.nlm.nih.gov/8961191/)

**Evidence boundary.** These studies do not establish HD efficacy. No direct HD therapeutic study of pentobarbital was located. Sharing the barbiturate class with phenobarbital does not allow the latter's juvenile-HD seizure case to be treated as pentobarbital evidence.

**Interpretation.** Current support consists mainly of drug-specific GABA pharmacology and graph-derived prioritization. Pentobarbital remains a pharmacologically related candidate with weak clinical support for HD. The HD psychiatric consensus provides symptom-management context only. [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/)

**References for this drug (5):**

- [R15](https://pubmed.ncbi.nlm.nih.gov/2981311/) — Barbiturate biochemical/slice comparison, outside HD.
- [R16](https://pubmed.ncbi.nlm.nih.gov/2850060/) — Electrophysiology including pentobarbitone, outside HD.
- [R20](https://pubmed.ncbi.nlm.nih.gov/2482885/) — Pentobarbitone/phenobarbitone GABA_A channel kinetics, outside HD.
- [R21](https://pubmed.ncbi.nlm.nih.gov/8961191/) — Direct receptor-activation experiments, outside HD.
- [R04](https://pubmed.ncbi.nlm.nih.gov/30040737/) — HD symptom-management context, not a pentobarbital efficacy study.

### 3.7 Clonazepam — Rank 7

**Direct HD evidence.** The early case series by Peiris et al. included three patients with Huntington chorea and reported reduced choreiform movements. Aoba et al. reported a substantial response of severe action myoclonus to clonazepam in an adult with HD. Stewart also published an HD treatment letter; only its bibliographic record was checked here, so no efficacy magnitude is attributed to it. [R22](https://pubmed.ncbi.nlm.nih.gov/131236/) [R24](https://pubmed.ncbi.nlm.nih.gov/1291167/) [R23](https://pubmed.ncbi.nlm.nih.gov/2962291/)

**Consistency across evidence types.** The 2019 international guideline mentions clonazepam for functionally impairing HD myoclonus, with Grade C evidence. The 2023 guideline and 2026 consensus provide related symptomatic context. In the Enroll-HD study, 5.6% of the adult-onset manifest-HD subgroup used clonazepam. This documents prescribing, not efficacy. [R02](https://pubmed.ncbi.nlm.nih.gov/31333565/) [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) [R05](https://pubmed.ncbi.nlm.nih.gov/42446188/) [R01](https://pubmed.ncbi.nlm.nih.gov/39431460/)

**Interpretation.** Clonazepam has one of the more complete evidence chains among these candidates, particularly for myoclonus. Improvement in myoclonus should not be generalized to all motor symptoms or disease modification.

**References for this drug (7):**

- [R22](https://pubmed.ncbi.nlm.nih.gov/131236/) — Choreiform-activity case series including three HD patients.
- [R23](https://pubmed.ncbi.nlm.nih.gov/2962291/) — HD treatment letter; bibliographic verification only.
- [R24](https://pubmed.ncbi.nlm.nih.gov/1291167/) — Adult-HD action-myoclonus case.
- [R02](https://pubmed.ncbi.nlm.nih.gov/31333565/) — International guideline: Grade C support for myoclonus management.
- [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) — DGN guideline.
- [R01](https://pubmed.ncbi.nlm.nih.gov/39431460/) — Enroll-HD prescribing cohort.
- [R05](https://pubmed.ncbi.nlm.nih.gov/42446188/) — 2026 HD expert consensus.

### 3.8 Phenobarbital — Rank 8

**Direct evidence.** Ishii and Nagasawa described a boy with juvenile HD who received long-term multidrug treatment including phenobarbital. During a period of severe myoclonic seizures, intensified treatment was followed by seizure suppression. Severe disease, concurrent medications, and respiratory support prevent isolation of phenobarbital's effect. [R25](https://www.jstage.jst.go.jp/article/jsmid/36/3/36_491/_article/-char/en)

**Why the subtype matters.** Cloud et al. conducted a multicenter retrospective study of 90 genetically confirmed juvenile-HD patients; 38% had seizures. This establishes the relevance of seizure management to juvenile HD, but is not an independent phenobarbital efficacy trial. [R26](https://pubmed.ncbi.nlm.nih.gov/23124580/)

GABA_A receptor experiments support phenobarbital's effects on inhibitory neurotransmission, but these are non-HD experiments. [R19](https://pubmed.ncbi.nlm.nih.gov/2471436/) [R20](https://pubmed.ncbi.nlm.nih.gov/2482885/) [R21](https://pubmed.ncbi.nlm.nih.gov/8961191/)

**Interpretation.** Phenobarbital has a case-level association with refractory seizures/myoclonus in juvenile HD. It should not be presented as a validated treatment for adult-HD chorea. A single case also cannot establish slower neurodegeneration.

**References for this drug (5):**

- [R25](https://www.jstage.jst.go.jp/article/jsmid/36/3/36_491/_article/-char/en) — Juvenile-HD phenobarbital case; the main direct evidence.
- [R26](https://pubmed.ncbi.nlm.nih.gov/23124580/) — Juvenile-HD seizure cohort providing subtype and symptom context.
- [R19](https://pubmed.ncbi.nlm.nih.gov/2471436/) — Phenobarbital/diazepam receptor-channel comparison, outside HD.
- [R20](https://pubmed.ncbi.nlm.nih.gov/2482885/) — GABA_A channel kinetics, outside HD.
- [R21](https://pubmed.ncbi.nlm.nih.gov/8961191/) — Barbiturate receptor activation, outside HD.

### 3.9 Benperidol — Rank 9

**Mechanism and relevance to HD.** Benperidol is a butyrophenone antipsychotic with D2 receptor-blocking activity. The local MSI graph links it to `DRD2`. Some antidopaminergic therapies are used for chorea or psychiatric symptoms in HD, providing a class-level rationale. [R27](https://pubmed.ncbi.nlm.nih.gov/15846648/) [R02](https://pubmed.ncbi.nlm.nih.gov/31333565/) [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/)

**Main evidence gap.** No benperidol-specific HD therapeutic study was located. The Cochrane review concerns schizophrenia, and its randomized evidence was itself limited. It cannot be used as evidence for HD efficacy. D2 blockade provides a mechanism, but does not imply that all D2 antagonists have equivalent effects in HD.

**Interpretation.** Benperidol has class-based pharmacological plausibility without drug-specific HD evidence. It is not a strong example of a successfully corroborated prediction in this case study.

**References for this drug (3):**

- [R27](https://pubmed.ncbi.nlm.nih.gov/15846648/) — Benperidol-specific Cochrane review in schizophrenia, not HD.
- [R02](https://pubmed.ncbi.nlm.nih.gov/31333565/) — HD guideline providing antipsychotic-class context, not validation of benperidol.
- [R03](https://pubmed.ncbi.nlm.nih.gov/37968732/) — HD guideline providing context and evidence boundaries for antidopaminergic treatment.

### 3.10 Orphenadrine — Rank 10

**Mechanistic evidence.** Kornhuber et al. demonstrated uncompetitive NMDA receptor antagonist activity using binding and patch-clamp experiments. Sureda et al. subsequently reported protective effects in cell culture and rat excitotoxicity experiments. [R30](https://pubmed.ncbi.nlm.nih.gov/8788072/) [R29](https://pubmed.ncbi.nlm.nih.gov/10340304/)

**The most relevant HD-like model.** Pubill et al. studied 3-NPA-induced neurotoxicity in rat cerebellar granule cells and rats, reporting protection that included lower injury-related markers. The 3-NPA model is relevant to studies of striatal injury and excitotoxicity in HD. However, **it is a toxin model, not a mutant-HTT genetic model or an HD clinical trial**. [R28](https://pubmed.ncbi.nlm.nih.gov/11159722/)

Desaphy et al. also identified effects on voltage-gated sodium channels, showing that orphenadrine acts beyond NMDA receptors. That study concerned analgesia and does not add evidence of HD efficacy. [R31](https://pubmed.ncbi.nlm.nih.gov/19217209/)

**Interpretation.** Orphenadrine is a useful preclinical candidate for discussion: it has drug-specific mechanistic evidence and animal-model findings relevant to HD research. It supports a hypothesis for further testing, rather than established neuroprotection in HD patients.

**References for this drug (4):**

- [R28](https://pubmed.ncbi.nlm.nih.gov/11159722/) — In vitro/in vivo 3-NPA experiments: the most relevant HD-like-model evidence for this drug.
- [R29](https://pubmed.ncbi.nlm.nih.gov/10340304/) — Cell/animal glutamate-excitotoxicity experiments, not genetic HD.
- [R30](https://pubmed.ncbi.nlm.nih.gov/8788072/) — NMDA receptor binding/patch-clamp study.
- [R31](https://pubmed.ncbi.nlm.nih.gov/19217209/) — Sodium-channel blockade/analgesia study supporting multitarget pharmacology, not HD efficacy.

## 4. Interpretation in the context of the local MSI graph

### 4.1 Pharmacological classes cluster in the top ten

The local raw drug-protein edges show:

- Four BZDs: prazepam, lorazepam, diazepam, and clonazepam.
- Three barbiturates: secobarbital, pentobarbital, and phenobarbital.
- **All seven have an annotated edge to GABRA1 (Entrez `2554`).**

This is an observation from the current graph and is consistent with their GABAergic pharmacology. It is **not a statistical enrichment test or proof that GABRA1 caused their rankings**. Several candidates with related pharmacology should not be described as independent disease-modifying mechanisms.

### 4.2 Direct disease-protein overlap does not explain these candidates

HD has 11 direct protein neighbors in the current indication-protein file. None overlaps with the direct protein neighbors of these ten drugs. The local data therefore provide no simple two-edge `drug -> shared protein <- HD` connection for these candidates.

Other graph contexts, including protein-protein interactions, other nodes, and semantic relationships, can still contribute to embeddings. **No path-attribution analysis was performed, so this report does not identify the specific paths responsible for each high score.**

The read-only graph checks used the following files in the working research project:

- `00.Reproduction/00.DREAMwalk_reproduction/data/msi_raw/1_drug_to_protein.tsv`
- `00.Reproduction/00.DREAMwalk_reproduction/data/msi_raw/2_indication_to_protein.tsv`

These raw inputs are not distributed with the code repository.

### 4.3 Points for the project Discussion

1. **The evidence is clearer for symptom management than for disease modification.** Candidates relate to chorea, myoclonus, seizures, or neuropsychiatric symptoms. DDA ranking does not itself distinguish these treatment objectives.
2. **An absent graph label does not establish a new literature finding.** Amantadine and clonazepam have prior HD publications but are absent from the current MSI HD positive labels.
3. **Drug-class clustering limits mechanistic diversity.** Several highly ranked candidates have related pharmacology. Class-level associations should not be counted as ten independent successful predictions.
4. **Disease subtype and evidence level must remain explicit.** A juvenile-HD seizure case, toxin-model neuroprotection, and an adult-HD chorea trial are different forms of evidence and should not be combined into one validation category.
5. **Literature support does not establish the causal effects of the model corrections.** These predictions use corrected embeddings. Explaining why a particular correction changes HD rankings would require a dedicated original-versus-corrected comparison. No such experiment was added here.

### 4.4 What the known HD-drug scores can support

The five known MSI HD-drug pairs below were positive training examples for every classifier. For each drug, the feature is `drug embedding - HD embedding`. XGBoost returns a positive-class score. `mean_score` averages that score over the ten classifiers; `sd_score` is the sample SD across those ten values. `rank_all` is the position among all 1,661 drugs, before known positives are removed.

| Known MSI pair | Mean score | Sample SD | Rank among all drugs |
| --- | ---: | ---: | ---: |
| baclofen (DB00181) | 0.9330 | 0.0238 | 1 |
| pimozide (DB01100) | 0.9152 | 0.0273 | 2 |
| perphenazine (DB00850) | 0.9086 | 0.0255 | 3 |
| tetrabenazine (DB04844) | 0.8418 | 0.0601 | 18 |
| coenzyme Q10 (DB09270) | 0.8025 | 0.0339 | 51 |

These are **in-sample scores**. They describe how the fitted classifiers rank associations they have already seen. They do not demonstrate recovery of withheld HD drugs, and they should not be used as an independent estimate of predictive performance. A score of 0.9330 is not a 93.3% probability of treatment efficacy. A lower score for another known pair also does not establish lower clinical value.

Using all known labels is a practical final-training step for generating candidate rankings. It does not by itself establish overfitting. Overfitting would need to be assessed on data excluded from classifier training. The current table supplies no such comparison, and averaging ten predictions does not change the training status of these five pairs.

The case-study Discussion should therefore focus on the unlabeled candidates and the evidence relevant to them. Candidate scores set the order for investigation. Direct HD trials, case reports, prescribing studies and non-HD pharmacology provide different levels of support. Agreement with prior literature is post hoc corroboration; a high ranking alone is not therapeutic validation.

### 4.5 Planned validation: HD-label holdout

A separate HD-label-holdout evaluation would exclude all HD positive and sampled negative labels from classifier training while retaining the HD node and its existing molecular/semantic embedding. The five known positives would remain excluded from negative sampling. Their positions in the resulting drug ranking could then be reported using individual ranks and a prespecified Recall@K, such as K=10 and K=50. Recall@K would be the number of the five withheld positives in the top K divided by five.

This would test transfer to a disease without classifier-training labels, conditional on its existing graph representation. It would be a stronger check of known-association recovery than the in-sample table above. It would not establish prediction for diseases absent from the graph. No new holdout run was performed for this report update; the earlier single-classifier holdout is a separate historical protocol.

## 5. Suggested report summary

> Using one fixed corrected DREAMwalk embedding and ten disjoint link-negative sets, we ranked unlabeled drug-Huntington disease pairs by their mean classifier score. Literature review identified direct HD treatment studies for amantadine, clonazepam, and diazepam, prescribing and symptom-specific reports for lorazepam, and a juvenile-HD seizure case involving phenobarbital. Orphenadrine had supportive evidence from excitotoxicity and 3-nitropropionic-acid models, but no HD clinical efficacy was established. For prazepam, secobarbital, pentobarbital, and benperidol, the evidence located was predominantly indirect. Seven candidates belonged to benzodiazepine or barbiturate classes and shared a GABRA1 annotation in the local MSI graph. HD was selected because it matches the rare-neurodegenerative-disease objective, is represented in MSI and has clinical literature available for critical assessment. The five known MSI HD-drug pairs were used in training, so their scores provide context only. The candidate ranking highlighted symptom-related associations and preclinical hypotheses for literature assessment; independent known-pair recovery requires a separate holdout evaluation.

This paragraph synthesizes the assessments above. Relevant original studies should accompany its use; the DREAMwalk paper alone is not a source for the drugs' efficacy.

## 6. Search scope and reading limitations

- Zotero was searched first, with no directly relevant entries located for this assessment. The subsequent search used PubMed, Europe PMC, and original journal pages.
- Main queries combined `Huntington + drug name` and `Huntington chorea + drug name`, supplemented by mechanism- or symptom-specific terms such as `myoclonus`, `catatonia`, `seizures`, `NMDA`, and `GABA receptor`.
- This was a targeted case-study literature search, **not a preregistered systematic review**. A study not located in this search is not proven not to exist.
- Drug-specific primary studies were prioritized. Guidelines, cohorts, and reviews are labeled by purpose. Non-HD pharmacology and class-level guidance are not counted as drug-specific HD efficacy evidence.
- Reading covered public abstracts, selected accessible full-text passages, and original correspondence. R10, R17, and R23 were checked at bibliographic-record level only; no unread efficacy results are inferred from them.
- Publications from 2024 and 2026 inform the current assessment. They do not establish prospective discovery in 2023. No temporal validation with a fixed historical cutoff was performed.
- This report translates and organizes existing predictions and literature. No models were retrained, experimental settings changed, or new files uploaded to GitHub. No clinical prescribing assessment was performed.

## 7. Complete references

References R01-R32 are cross-referenced in the accompanying [reference inventory](HD_top10_reference_inventory.csv). Section 3 identifies the role of each reference for each drug.

**R01. Feleus S, Skotnicki LEM, Roos RAC, et al. (2024).** [Medication Use and Treatment Indications in Huntington's Disease; Analyses from a Large Cohort](https://pubmed.ncbi.nlm.nih.gov/39431460/). *Movement disorders clinical practice*, 11(12):1530-1541. DOI: [10.1002/mdc3.14230](https://doi.org/10.1002/mdc3.14230). PMID: 39431460. Reading scope: Relevant full-text passages (Europe PMC).

**R02. Bachoud-Lévi AC, Ferreira J, Massart R, et al. (2019).** [International Guidelines for the Treatment of Huntington's Disease](https://pubmed.ncbi.nlm.nih.gov/31333565/). *Frontiers in neurology*, 10:710. DOI: [10.3389/fneur.2019.00710](https://doi.org/10.3389/fneur.2019.00710). PMID: 31333565. Reading scope: Relevant passages in the journal full text.

**R03. Saft C, Burgunder JM, Dose M, et al. (2023).** [Symptomatic treatment options for Huntington's disease (guidelines of the German Neurological Society)](https://pubmed.ncbi.nlm.nih.gov/37968732/). *Neurological research and practice*, 5(1):61. DOI: [10.1186/s42466-023-00285-1](https://doi.org/10.1186/s42466-023-00285-1). PMID: 37968732. Reading scope: Relevant passages in the journal full text.

**R04. Anderson KE, van Duijn E, Craufurd D, et al. (2018).** [Clinical Management of Neuropsychiatric Symptoms of Huntington Disease: Expert-Based Consensus Guidelines on Agitation, Anxiety, Apathy, Psychosis and Sleep Disorders](https://pubmed.ncbi.nlm.nih.gov/30040737/). *Journal of Huntington's disease*, 7(3):355-366. DOI: [10.3233/jhd-180293](https://doi.org/10.3233/jhd-180293). PMID: 30040737. Reading scope: Relevant full-text passages (Europe PMC).

**R05. Furr Stimming E, Bang J, Brown A, et al. (2026).** [Practical recommendations for the treatment of chorea associated with Huntington's disease: An expert consensus](https://pubmed.ncbi.nlm.nih.gov/42446188/). *Journal of Huntington's disease*. Advance online publication. DOI: [10.1177/18796397261468482](https://doi.org/10.1177/18796397261468482). PMID: 42446188. Reading scope: Relevant passages in the journal full text.

**R06. Verhagen Metman L, Morris MJ, Farmer C, et al. (2002).** [Huntington's disease: a randomized, controlled trial using the NMDA-antagonist amantadine](https://pubmed.ncbi.nlm.nih.gov/12221159/). *Neurology*, 59(5):694-699. DOI: [10.1212/wnl.59.5.694](https://doi.org/10.1212/wnl.59.5.694). PMID: 12221159. Reading scope: Public abstract.

**R07. O'Suilleabhain P, Dewey RB. (2003).** [A randomized trial of amantadine in Huntington disease](https://pubmed.ncbi.nlm.nih.gov/12873857/). *Archives of neurology*, 60(7):996-998. DOI: [10.1001/archneur.60.7.996](https://doi.org/10.1001/archneur.60.7.996). PMID: 12873857. Reading scope: Public abstract and journal page.

**R08. Lucetti C, Del Dotto P, Gambaccini G, et al. (2003).** [IV amantadine improves chorea in Huntington's disease: an acute randomized, controlled study](https://pubmed.ncbi.nlm.nih.gov/12821751/). *Neurology*, 60(12):1995-1997. DOI: [10.1212/01.wnl.0000068165.07883.64](https://doi.org/10.1212/01.wnl.0000068165.07883.64). PMID: 12821751. Reading scope: Public abstract.

**R09. Heckmann JM, Legg P, Sklar D, et al. (2004).** [IV amantadine improves chorea in Huntington's disease: an acute randomized, controlled study](https://pubmed.ncbi.nlm.nih.gov/15304616/). *Neurology*, 63(3):597-8; author reply 597-8. DOI: [10.1212/wnl.63.3.597](https://doi.org/10.1212/wnl.63.3.597). PMID: 15304616. Reading scope: Public text of the original journal correspondence; no abstract in the PubMed record.

**R10. Rickels K, Sablosky L, Silverman H, et al. (1977).** [Prazepam in anxiety: a controlled clinical trial](https://pubmed.ncbi.nlm.nih.gov/858240/). *Comprehensive psychiatry*, 18(3):239-249. DOI: [10.1016/0010-440x(77)90018-9](https://doi.org/10.1016/0010-440x(77)90018-9). PMID: 858240. Reading scope: Bibliographic record checked; complete trial results not extracted.

**R11. Weir JH. (1978).** [Prazepam in the treatment of anxiety: a placebo-controlled multicenter evaluation](https://pubmed.ncbi.nlm.nih.gov/721788/). *The Journal of clinical psychiatry*, 39(12):841-847. PMID: 721788. Reading scope: Public abstract.

**R12. Fabre LF, McLendon DM, Mallette A. (1979).** [A double-blind comparison of prazepam with diazepam, chlorazepate dipotassium and placebo in anxious out-patients](https://pubmed.ncbi.nlm.nih.gov/39012/). *The Journal of international medical research*, 7(2):147-151. DOI: [10.1177/030006057900700206](https://doi.org/10.1177/030006057900700206). PMID: 39012. Reading scope: Public abstract.

**R13. Miglis G, Kadan J, Noe G, et al. (2026).** [Management of catatonia in Huntington disease: A scoping review](https://pubmed.ncbi.nlm.nih.gov/42155211/). *General hospital psychiatry*, 101:39-44. DOI: [10.1016/j.genhosppsych.2026.05.004](https://doi.org/10.1016/j.genhosppsych.2026.05.004). PMID: 42155211. Reading scope: Public abstract.

**R14. Merida-Puga J, Ramirez-Bermudez J, Aguilar-Venegas LC, et al. (2011).** [Westphal variant Huntington disease and refractory catatonia: a case report](https://pubmed.ncbi.nlm.nih.gov/22123587/). *Cognitive and behavioral neurology : official journal of the Society for Behavioral and Cognitive Neurology*, 24(4):204-208. DOI: [10.1097/wnn.0b013e318240080d](https://doi.org/10.1097/wnn.0b013e318240080d). PMID: 22123587. Reading scope: Public abstract; the complete case medication sequence was not independently reviewed.

**R15. Iadarola MJ, Fanelli RJ, McNamara JO, et al. (1985).** [Comparison of the effects of diphenylbarbituric acid, phenobarbital, pentobarbital and secobarbital on GABA-mediated inhibition and benzodiazepine binding](https://pubmed.ncbi.nlm.nih.gov/2981311/). *The Journal of pharmacology and experimental therapeutics*, 232(1):127-133. DOI: [10.1016/s0022-3565(25)20061-1](https://doi.org/10.1016/s0022-3565(25)20061-1). PMID: 2981311. Reading scope: Public abstract.

**R16. Peters JA, Kirkness EF, Callachan H, et al. (1988).** [Modulation of the GABAA receptor by depressant barbiturates and pregnane steroids](https://pubmed.ncbi.nlm.nih.gov/2850060/). *British journal of pharmacology*, 94(4):1257-1269. DOI: [10.1111/j.1476-5381.1988.tb11646.x](https://doi.org/10.1111/j.1476-5381.1988.tb11646.x). PMID: 2850060. Reading scope: Public abstract.

**R17. NILSEN JA. (1964).** [VALIUM THERAPY OF HUNTINGTON'S CHOREA](https://pubmed.ncbi.nlm.nih.gov/14154759/). *The American journal of psychiatry*, 120:1197-1198. DOI: [10.1176/ajp.120.12.1197](https://doi.org/10.1176/ajp.120.12.1197). PMID: 14154759. Reading scope: Bibliographic record checked; no abstract available.

**R18. Caraceni TA, Girotti F, Giovannini P, et al. (1980).** [Effects of DA agonist in Huntington disease hyperkinesia](https://pubmed.ncbi.nlm.nih.gov/6210644/). *Italian journal of neurological sciences*, 1(3):155-161. DOI: [10.1007/bf02335845](https://doi.org/10.1007/bf02335845). PMID: 6210644. Reading scope: Public abstract.

**R19. Twyman RE, Rogers CJ, Macdonald RL. (1989).** [Differential regulation of gamma-aminobutyric acid receptor channels by diazepam and phenobarbital](https://pubmed.ncbi.nlm.nih.gov/2471436/). *Annals of neurology*, 25(3):213-220. DOI: [10.1002/ana.410250302](https://doi.org/10.1002/ana.410250302). PMID: 2471436. Reading scope: Public abstract.

**R20. MacDonald RL, Rogers CJ, Twyman RE. (1989).** [Barbiturate regulation of kinetic properties of the GABAA receptor channel of mouse spinal neurones in culture](https://pubmed.ncbi.nlm.nih.gov/2482885/). *The Journal of physiology*, 417:483-500. DOI: [10.1113/jphysiol.1989.sp017814](https://doi.org/10.1113/jphysiol.1989.sp017814). PMID: 2482885. Reading scope: Public abstract.

**R21. Rho JM, Donevan SD, Rogawski MA. (1996).** [Direct activation of GABAA receptors by barbiturates in cultured rat hippocampal neurons](https://pubmed.ncbi.nlm.nih.gov/8961191/). *The Journal of physiology*, 497 ( Pt 2):509-522. DOI: [10.1113/jphysiol.1996.sp021784](https://doi.org/10.1113/jphysiol.1996.sp021784). PMID: 8961191. Reading scope: Public abstract.

**R22. Peiris JB, Boralessa H, Lionel ND. (1976).** [Clonazepam in the treatment of choreiform activity](https://pubmed.ncbi.nlm.nih.gov/131236/). *The Medical journal of Australia*, 1(8):225-227. PMID: 131236. Reading scope: Public abstract.

**R23. Stewart JT. (1988).** [Treatment of Huntington's disease with clonazepam](https://pubmed.ncbi.nlm.nih.gov/2962291/). *Southern medical journal*, 81(1):102. DOI: [10.1097/00007611-198801000-00029](https://doi.org/10.1097/00007611-198801000-00029). PMID: 2962291. Reading scope: Bibliographic record checked; no abstract available.

**R24. Aoba S, Komiyama A, Yamada H, et al. (1992).** [Action myoclonus in adult Huntington's disease](https://pubmed.ncbi.nlm.nih.gov/1291167/). *Rinsho shinkeigaku = Clinical neurology*, 32(7):739-742. PMID: 1291167. Reading scope: Public English abstract; original article in Japanese.

**R25. Ishii M, Nagasawa K. (2011).** [A case of juvenile Huntington's disease treated by phenobarbital high dose therapy for four years](https://www.jstage.jst.go.jp/article/jsmid/36/3/36_491/_article/-char/en). *Japanese Journal of Severe Motor and Intellectual Disabilities*, 36(3):491–494. DOI: [10.24635/jsmid.36.3_491](https://doi.org/10.24635/jsmid.36.3_491). Reading scope: English abstract of the original J-STAGE paper.

**R26. Cloud LJ, Rosenblatt A, Margolis RL, et al. (2012).** [Seizures in juvenile Huntington's disease: frequency and characterization in a multicenter cohort](https://pubmed.ncbi.nlm.nih.gov/23124580/). *Movement disorders : official journal of the Movement Disorder Society*, 27(14):1797-1800. DOI: [10.1002/mds.25237](https://doi.org/10.1002/mds.25237). PMID: 23124580. Reading scope: Public abstract.

**R27. Leucht S, Hartung B. (2005).** [Benperidol for schizophrenia](https://pubmed.ncbi.nlm.nih.gov/15846648/). *The Cochrane database of systematic reviews*, (2):CD003083. DOI: [10.1002/14651858.cd003083.pub2](https://doi.org/10.1002/14651858.cd003083.pub2). PMID: 15846648. Reading scope: Public abstract and relevant Cochrane/PMC content.

**R28. Pubill D, Verdaguer E, Canudas AM, et al. (2001).** [Orphenadrine prevents 3-nitropropionic acid-induced neurotoxicity in vitro and in vivo](https://pubmed.ncbi.nlm.nih.gov/11159722/). *British journal of pharmacology*, 132(3):693-702. DOI: [10.1038/sj.bjp.0703869](https://doi.org/10.1038/sj.bjp.0703869). PMID: 11159722. Reading scope: Public abstract and PMC-indexed Results passages.

**R29. Sureda FX, Gabriel C, Pallàs M, et al. (1999).** [In vitro and in vivo protective effect of orphenadrine on glutamate neurotoxicity](https://pubmed.ncbi.nlm.nih.gov/10340304/). *Neuropharmacology*, 38(5):671-677. DOI: [10.1016/s0028-3908(98)00228-7](https://doi.org/10.1016/s0028-3908(98)00228-7). PMID: 10340304. Reading scope: Public abstract.

**R30. Kornhuber J, Parsons CG, Hartmann S, et al. (1995).** [Orphenadrine is an uncompetitive N-methyl-D-aspartate (NMDA) receptor antagonist: binding and patch clamp studies](https://pubmed.ncbi.nlm.nih.gov/8788072/). *Journal of neural transmission. General section*, 102(3):237-246. DOI: [10.1007/bf01281158](https://doi.org/10.1007/bf01281158). PMID: 8788072. Reading scope: Public abstract.

**R31. Desaphy JF, Dipalma A, De Bellis M, et al. (2009).** [Involvement of voltage-gated sodium channels blockade in the analgesic effects of orphenadrine](https://pubmed.ncbi.nlm.nih.gov/19217209/). *Pain*, 142(3):225-235. DOI: [10.1016/j.pain.2009.01.010](https://doi.org/10.1016/j.pain.2009.01.010). PMID: 19217209. Reading scope: Public abstract.

**R32. Bush G, Fink M, Petrides G, et al. (1996).** [Catatonia. II. Treatment with lorazepam and electroconvulsive therapy](https://pubmed.ncbi.nlm.nih.gov/8686484/). *Acta psychiatrica Scandinavica*, 93(2):137-143. DOI: [10.1111/j.1600-0447.1996.tb09815.x](https://doi.org/10.1111/j.1600-0447.1996.tb09815.x). PMID: 8686484. Reading scope: Public abstract.
