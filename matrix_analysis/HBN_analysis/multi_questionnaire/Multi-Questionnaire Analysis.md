- [Meta-factor model: My God, it's full of Factors!](#meta-factor-model-my-god-its-full-of-factors)
  - [Introduction](#introduction)
  - [Historical background and motivation](#historical-background-and-motivation)
  - [What does this paper do?](#what-does-this-paper-do)
    - [Interpretable factorization of questionnaires](#interpretable-factorization-of-questionnaires)
    - [Meta-factorization of questionnaires](#meta-factorization-of-questionnaires)
    - [Micro-Meta-Factorization of questionnaires [need a better name]](#micro-meta-factorization-of-questionnaires-need-a-better-name)
    - [Prediction on Structural Brain Imaging](#prediction-on-structural-brain-imaging)

# Meta-factor model: My God, it's full of Factors!

## Introduction

## Historical background and motivation

- many questionnaires developed at different points in time, to get at different aspects of psychopathology
- generally, a questionnaire assumes one or more latent constructs, and is designed so that the presence of these can be quantified, e.g. by
  - calculation of subscales corresponding to the constructs (e.g. by tallying positive answers)
  - (exploratory/confirmatory) factor analysis of responses, with factors representing the degree to which a factor is present in each respondent
- problem 1: factor models are not necessarily interpretable
- problem 2: redundancy across questionnaires
- problem 3: no neurobiological basis for constructs

## What does this paper do?

### Interpretable factorization of questionnaires

- We introduce a new approach for factoring questionnaires that yields more interpretable factors, and apply it to CBCL and SDQ from HBN
- qualitative assessment: show the questions associated with factors, and that some factors can cross subscales
**CBCL question factor (proposed)**
<img src="./figure/question_loadings/CBCL_propose.png" width="800"/>
  <details>
  <summary>Loadings of each CBCL question factor</summary>
  <img src="./figure/question_loadings/propose/factors_1.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_2.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_3.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_4.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_5.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_6.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_7.png" width="800"/>
  <img src="./figure/question_loadings/propose/factors_8.png" width="800"/>
  </details>
  <details>
  <summary>CBCL question factor (Factor Analysis, promax rotation)</summary>
  <img src="./figure/question_loadings/CBCL_FA.png" width="800"/>
  </details>
  <details>
  <summary>CBCL question factor (Subscale)</summary>
  <img src="./figure/question_loadings/CBCL_subscale.png" width="800"/>
  </details>

- quantitative assessment: show that the diagnostic classification performance from new factorization is indistinguishable for that using subscales or factors from factor analysis
<img src="./figure/prediction/proposed.png" width="800"/>

- quantitative assessment: apply to ABCD as well, and show that the loadings for factors are very similar between HBN and ABCD
<img src="./figure/question_loadings/ABCD_CBCL_propose.png" width="800"/>

- how are our factors and subscales related (sankey plots for CBCL and SDQ)
<img src="./figure/question_loadings/sanky_plot_CBCL.png" width="800"/>


### Meta-factorization of questionnaires

- We apply the interpretable factorization to the 21 questionnaires from the Healthy Brain Network dataset
- key point: compression does not lose information
- in the main paper, summarize main points (backed up by supplement)
  - who has filled what questionnaire
  <img src="./figure/merge_response_availability.png" width="800"/>
  <img src="./figure/merge_response_normalized.png" width="800"/>
  <details>
  <summary> Train-Validation-Test split </summary>
    <img src="./figure/merge_train.png" width="800"/>
    <img src="./figure/merge_valid.png" width="800"/>
    <img src="./figure/merge_test.png" width="800"/>
  </details>
  - compression rate table (for our method vs subscales, if available)
  <img src="./figure/compression_ratio.png" width="800"/>
  - note that "disorder-specific" questionnaires don't do much better at predicting their respective diagnosis
  - possible heatmap: diagnosis vs scales (tile so that cross-domain instruments come first, then diagnostic-specific); ribbon of cross-diagnosis results do well
  - test per questionnaire (ours vs raw vs scale)
<img src="./figure/prediction/subscale.png" width="800"/>
- in the supplement
  - repeat the qualitative and quantitative assessments of CBCL and SDQ, with their respective subscales (if available)
  - heatmaps for FA
  <img src="./figure/prediction/FA_promax.png" width="800"/>

- We introduce a second level interpretable factorization approach -- meta-factors (factors of factors) -- and apply it to the factorizations of all the HBN questionnaires
- This yields a model with 15 meta-factors
  - level 1 (factors concatenated)
  <img src="./figure/metafactor_loadings/train_metafactor_availability.png" width="800"/>
  <img src="./figure/metafactor_loadings/train_metafactor.png" width="800"/>

  <details>
  <summary> Concatenated factors for validation and test set </summary>
  Validation set
  <img src="./figure/metafactor_loadings/valid_metafactor_availability.png" width="800"/>
  <img src="./figure/metafactor_loadings/valid_metafactor.png" width="800"/>
  Test set
  <img src="./figure/metafactor_loadings/test_metafactor_availability.png" width="800"/>
  <img src="./figure/metafactor_loadings/test_metafactor.png" width="800"/>
  </details>

  - level 2 (factors of factors)
  <img src="./figure/metafactor_loadings/metafactor.png" width="800"/>
- Evaluation
  - qualitative assessment: show that meta-factors can be questionnaire specific, or cross-questionnaire (group together questions across questionnaires meaningfully)
  <img src="./figure/metafactor_loadings/metafactor_corr.png" width="800"/>

  - this is done using the diagram for the meta-factor structure, but then also the top questions per meta-factor (showing questionnaire provenance)
  <img src="./figure/metafactor_loadings/aggregrated_loadings_centroid.png" width="800"/>
  - quantitative assessment: show that the diagnostic classification performance from meta-factorization is indistinguishable for that of the best factorization for each questionnaire
  <img src="./figure/prediction/proposed.png" width="800"/>


### Micro-Meta-Factorization of questionnaires [need a better name]

Finally, it did not escape our notice that the process to generate the 15 meta-factors immediately suggests a possible approach to reducing the original thousands of questions to a much smaller, equivalently informative subset.

- We introduce an approach to identify a reduced set of informative questions:
  - learn factor models for each questionnaire
  - learn meta-factor model
  - identify top X questions in each meta-factor
  - redo the entire process using only those questions

  <img src="./figure/metafactor_loadings/cluster_centroid.png" width="800"/>

- Evaluation (as we reduce the number of questions)
  - the diagnostic performance decays very slowly

  <img src="./figure/variable_reduction/ADHD_trend.png" width="800"/>

  <img src="./figure/variable_reduction/Depression_trend.png" width="800"/>

  <img src="./figure/variable_reduction/GenAnxiety_trend.png" width="800"/>

  <details>
  <summary>Other diagnostic prediction</summary>
  <img src="./figure/variable_reduction/BPD_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Eating_disorder_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Encopresis_Enuresis_trend.png" width="800"/>
  <img src="./figure/variable_reduction/OCD_trend.png" width="800"/>
  <img src="./figure/variable_reduction/ODD_ConductDis_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Panic_Agoraphobia_SeparationAnx_SocialAnx_trend.png" width="800"/>
  <img src="./figure/variable_reduction/PTSD_Trauma_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Schizophrenia_Psychosis_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Sleep_Probs_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Specific_Phobia_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Substance_Issue_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Suicidal_SelfHarm_Homicidal_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Suspected_ASD_trend.png" width="800"/>
  <img src="./figure/variable_reduction/Tic_trend.png" width="800"/>
  </details>

  - can we reconstruct individual questionnaires?


  - correlation between meta-factors extracted using different numbers of questions decays very slowly (meta-factors are very robust)
  Question sorted according to R-squared of 'full' matrix reconstruction
  <img src="./figure/imputation/R2_full_indep_sort.png" width="800"/>
  Question sorted according to R-squared per experiment
  <img src="./figure/imputation/R2_full_consistent_sort.png" width="800"/>
- Based on our results, we believe that meta-factors can be acceptably recovered from as few as TODO questions per metafactor.
Answer: Top 10 per metafactor?
- "How low can you go?.." (LIMBO?)


- compare with CBCL and SDQ (roughly the same \# of questions)
<img src="./figure/imputation/comparison.png" width="600"/>

- are there SDQ questions in every factor? (yes, in all but communication/ASD and anxiety factors)

<details>
<summary> Top 10 questions in each metafactor </summary>
<img src="./figure/metafactor_loadings/factors/train_average/factors_1.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_2.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_3.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_4.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_5.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_6.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_7.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_8.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_9.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_10.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_11.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_12.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_13.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_14.png" width="600"/>
<img src="./figure/metafactor_loadings/factors/train_average/factors_15.png" width="600"/>
</details>


### Prediction on Structural Brain Imaging

[Brain Test](/figure/brain/brain_test.html)