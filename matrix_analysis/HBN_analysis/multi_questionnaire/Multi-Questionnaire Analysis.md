# Multi-Questionnaire Analysis


- [Multi-Questionnaire Analysis](#multi-questionnaire-analysis)
  - [Data Preparation](#data-preparation)
  - [Individual questionnaire analysis](#individual-questionnaire-analysis)
    - [Factors of each questionnaire](#factors-of-each-questionnaire)
  - [Matrix Imputation and Factorization](#matrix-imputation-and-factorization)
    - [Factor Level Imputation and Factorization](#factor-level-imputation-and-factorization)
      - [Factorization on Concatenated Factors](#factorization-on-concatenated-factors)
      - [Subject embedding](#subject-embedding)
      - [Loadings of each Factor](#loadings-of-each-factor)
    - [Prediction performance](#prediction-performance)
      - [Prediction performance of individual questionnaires](#prediction-performance-of-individual-questionnaires)
      - [Prediction performance for harmonized data](#prediction-performance-for-harmonized-data)
  - [Achieved results](#achieved-results)

## Data Preparation

With multiple questionnaires, we extract subject and question embedding by factorizing concatenated, normalized questionnaires.

The visualization of questionnaires availability is shown below. Notice that questionnaires have varying scales (upper $x$-axis in the figure).

> **[Update]**
> 
> - [2022-05] In previous version, variables such as `Total` or `subscale total` were included. We remove these variables in the current version.
> - [2022-05] `SympChck` is added into the current version. The requirement for availability is also reduced to $20\%$.

![](./figure/2022-05-02-13-42-52.png)

> **Remark**
> 
> - As recognized by Bridget, reversing scores for odd-numbered questions is required for `PANAS` questionnaire.
> - `ESWAN` questionnaire has 4 subscales: `MDD`, `DMDD`, `Panic` and `SocAnx`. These groupings were ignored during all experiments.
> - We only consider the 5 questions in `CSSRS`, ignoring variables such as `_ideationtype`, `_freq`, `_duration`, `_controllability`, etc.
> - In `SymChck`, we only consider variable in `current` time point. We ignore variables measured in the `Past`.
> - All sub-scores / total scores in each questionnaires are removed. This aligns with the assumption that each question (variable) is equally important from different questionnaires.

> **Remark on SWAN and ESWAN**
> - The scores range from $[-3, 3]$ and possibly typo appears on the description of scales: 
> 
> > -3= Far <span style="color:red">above</span> average
> > -2= <span style="color:red">Above</span> average
> > -1= Slightly <span style="color:red">above</span> average
> > 0= Average
> > 1= Slightly <span style="color:blue">below</span> average
> > 2= <span style="color:blue">Above</span>  average
> > 3= Far <span style="color:blue">above</span>  average
>
> - To align with the motivation of improving interpretation, we split each question on these questionnaires into **<span style="color:blue">Positive [P]</span>** and **<span style="color:red">Negative [N]</span>** variable sets. Therefore, the number of varaibles were doubled in `ESWAN` and `SWAN`.
>


> **Remark**
> 
> Normalizing with respect to maximum norm is the simplest approach for normalization, under the assumption that responses from different questionnaires have equivalent importance. However, it could be problematic when sparsity of questionnaires varies. For instance, the `Barratt` responses is related to occupation types and educational level. Therefore most of the responses are non-zeros (due to the way we quantify occupation types and education level). Yet, most entries from `CBCL` are zeros.
> 
> One possible way to bypass the challenge is to consider a two-level factorization. The first level is conducted on each individual questionnaire, while the second level is conducted on the concatenated factors from the first level. Results and details will be reported in later section.

After maximum norm normalization, the matrix representing the concatenated questionnaires is shown below:

![](2022-05-17-07-22-05.png)

Most of the subjects were not participated in all surveys, therefore we filtered some questionnaires with very few subjects before matrix factorization was performed.

We concatenate all questionnaires based on subject ID `EID` to obtain a complete participant list in the `HBN` dataset. We only includes questionnaires with more than `20%` subjects. In particular, the list of questionnaires we considered is shown below:
<p align="center">
<img src="./figure/2022-05-13-17-12-59.png" width="300">
</p>

## Individual questionnaire analysis

For each questionnaire, we perform analysis similar to CBCL dataset and obtain the intrinsic dimension through blockwise cross validation.
<p align="center">
<img src="./figure/2022-05-13-17-14-28.png" width="400">
</p>

> **[Update]**
> 
> - [2022-05] The intrinsic dimension of `CBCL` by cross validation is updated to 11. The difference is due to a finer scale of hyperparameter search and a better optimization strategy used in the updated algorithm.

### Factors of each questionnaire

The factors corresponding to the intrinsic dimension for each questionnaire can be computed using the constrained matrix factorization. Top $\min(10,\text{ total number of question})$ questions in each factor are visualized using barplot.

(Click to enlarge)
<details>
  <summary>ARI_P</summary>
  <p>
    <img src="./figure/ARI_P_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>ARI_S</summary>
  <p>
    <img src="./figure/ARI_S_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>ASSQ</summary>
  <p>
    <img src="./figure/ASSQ_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>AUDIT</summary>
  <p>
    <img src="./figure/AUDIT_factors.png" width="600/">
  </p>
</details>

<details>
  <summary>Barratt</summary>
  <p>
    <img src="./figure/Barratt_P1_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>C3SR</summary>
  <p>
    <img src="./figure/C3SR_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>CBCL</summary>
  <p>
    <img src="./figure/CBCL_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>ESWAN</summary>
  <p>
    <img src="./figure/ESWAN_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>ICU_P</summary>
  <p>
    <img src="./figure/ICU_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>ICU_SR</summary>
  <p>
    <img src="./figure/ICU_SR_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>MFQ_P</summary>
  <p>
    <img src="./figure/MFQ_P_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>MFQ_SR</summary>
  <p>
    <img src="./figure/MFQ_SR_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>PANAS</summary>
  <p>
    <img src="./figure/PANAS_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>RBS</summary>
  <p>
    <img src="./figure/RBS_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SCARED_P</summary>
  <p>
    <img src="./figure/SCARED_P_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SCARED_SR</summary>
  <p>
    <img src="./figure/SCARED_SR_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SCQ factors</summary>
  <p>
    <img src="./figure/SCQ_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SDQ factors</summary>
  <p>
    <img src="./figure/SDQ_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SRS factors</summary>
  <p>
    <img src="./figure/SRS_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SWAN factors</summary>
  <p>
    <img src="./figure/SWAN_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>SympChck factors</summary>
  <p>
    <img src="./figure/CSC_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>TRF factors</summary>
  <p>
    <img src="./figure/TRF_factors.png" width="600"/>
  </p>
</details>

<details>
  <summary>YSR factors</summary>
  <p>
    <img src="./figure/YSR_factors.png" width="600"/>
  </p>
</details>

## Matrix Imputation and Factorization

We consider two version of matrix imputation
- **Question Level** : Direct imputation by concatenating questions from all questionnaire. This form a matrix with dimension $3578 \times 1077$ ( Number of subjects $\times$ number of questions). Imputation performs directly on each missing entry in each questions.
- **Factor Level** : Perform matrix factorization for each questionnaire, followed by concatenating subject embedding factors from each questionnaire. Then perform imputation on the factor level (matrix with dimension $3578 \times 148$).

<!-- ### Question Level Imputation and Factorization

Performing matrix factorization directly on the matrix representing the concatenated questionnaire, we come up with the following imputed result

![](2022-05-17-07-23-50.png) -->

### Factor Level Imputation and Factorization

![](2022-05-17-07-29-13.png)

In the factor level, the concatenated factors $F$ from questionnaires are shown above. Recall that we have constrained the factors to have range $[0, 1]$.

The availability of factor loadings is shown below:

![](2022-05-17-07-46-39.png)

By treating $F$ as the input matrix, we perform imputation to obtain the following

![](2022-05-17-07-47-03.png)

> **Remark**
> Denote $M_i$ to be the matrix data for questionnaire $i$. On individual questionnaire, we have
> $$M_i \approx F_i \cdot Q_i^T$$
> On the factor level, the concatednated factors $F$ is:
> $$ F = \begin{bmatrix} F_1 \big| F_2 \big| & \cdots & \big| F_k \end{bmatrix} $$
> which is further factorized to
> $$ F \approx W \cdot P^T =W \cdot \begin{bmatrix} P_1 \big| P_2 \big| & \cdots & \big| P_k \end{bmatrix}^T$$
> This gives
> $$ M_i \approx W \cdot P_i^T \cdot Q_i^T $$

#### Factorization on Concatenated Factors

The factorization results on the factor levels reveal the relevant relationship between factors from individual questionnaires. For instance, the third factor from `SCARED_P` has a strong relationship with its 7-th factor, 4-th factor in `SympChck` and also 3-rd factor in `SCARED_SR`. 

![](2022-05-17-17-28-51.png)

The corresponding correlation matrix is

![](2022-05-17-17-47-29.png)

> **Remark**
> - There is a observable correlation between factors from self-reported questionnaires: `YSR`, `scared_sr`, `MFQ_SR` and `C3SR`
> - `CBCL` factors are correlated with multiple questionnaires. This implies `CBCL` factors cover subspaces spanned by multiple questionnaires' factors. This is consistent to our understanding that `CBCL` is comprehensive.
> - `TRF` is relatively independent and factors within `TRF` are correlated. It might indicates that questions in `TRF` are excessive.

#### Subject embedding

Clustering of subjects can be performed based on the obtained subject embedding $W$. However, more investigation has to be done to find connection between clusters and phenotypes.
![](2022-05-18-09-16-20.png)

As an exploration, we concatenate the clustered subject embedding with diagnosis labels. Some obvious consistent patterns between factors and `Suspectred ASD`, `GenAnxiety` could be recognized, which aligns with the high prediction accuracy on those labels.

<details>
  <summary>ADHD</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_ADHD.png" width=100%/>
  </p>
</details>

<details>
  <summary>BPD</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_BPD.png" width=100%/>
  </p>
</details>

<details>
  <summary>GenAnxiety</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_GenAnxiety.png" width=100%/>
  </p>
</details>

<details>
  <summary>Depression</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_Depression.png" width=100%/>
  </p>
</details>

<details>
  <summary>Eating_Disorder</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_Eating_Disorder.png" width=100%/>
  </p>
</details>

<details>
  <summary>Suspected_ASD</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_Suspected_ASD.png" width=100%/>
  </p>
</details>


<details>
  <summary>Sleep_Probs</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_Sleep_Probs.png" width=100%/>
  </p>
</details>

<details>
  <summary>Specific_Phobia</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_Specific_Phobia.png" width=100%/>
  </p>
</details>

<details>
  <summary>Panic_Agoraphobia_SeparationAnx_SocialAnx</summary>
  <p>
    <img src="./MF_impute_15/subjfactors_Panic_Agoraphobia_SeparationAnx_SocialAnx.png" width=100%/>
  </p>
</details>

#### Loadings of each Factor

Under the two-level factorization, the importance of each question in every factor is disclosed by the magnitude of $P_i. \cdot Q_i$. Top 20 questions (in terms of magnitude) for each factor is shown below:
<details>
  <summary>15 factors</summary>
  <p>
    <img src="./MF_impute_15/MF_impute_15_factors.png" width=100%/>
  </p>
</details>
<br>

> **Remark**
> Using the two-level scheme, we bypass the needs of normalization for each questionnaire during individual factorization because we enforce the subject embeddings obtained via the first level ranging from [0, 1].
> However, if we would like to study the contribution of each question in the factors, we still need to make an assumption that every question is as importance as others. Otherwise, misinterpretation may occur.

### Prediction performance

#### Prediction performance of individual questionnaires

<details>
  <summary>ADHD</summary>
  <p>
    <img src="./MF_perQ/prediction_ADHD.png" width=100%/>
  </p>
</details>

<details>
  <summary>BPD</summary>
  <p>
    <img src="./MF_perQ/prediction_BPD.png" width=100%/>
  </p>
</details>

<details>
  <summary>GenAnxiety</summary>
  <p>
    <img src="./MF_perQ/prediction_GenAnxiety.png" width=100%/>
  </p>
</details>

<details>
  <summary>Depression</summary>
  <p>
    <img src="./MF_perQ/prediction_Depression.png" width=100%/>
  </p>
</details>

<details>
  <summary>Eating_Disorder</summary>
  <p>
    <img src="./MF_perQ/prediction_Eating_Disorder.png" width=100%/>
  </p>
</details>

<details>
  <summary>Suspected_ASD</summary>
  <p>
    <img src="./MF_perQ/prediction_Suspected_ASD.png" width=100%/>
  </p>
</details>


<details>
  <summary>Sleep_Probs</summary>
  <p>
    <img src="./MF_perQ/prediction_Sleep_Probs.png" width=100%/>
  </p>
</details>

<details>
  <summary>Specific_Phobia</summary>
  <p>
    <img src="./MF_perQ/prediction_Specific_Phobia.png" width=100%/>
  </p>
</details>

<details>
  <summary>Panic_Agoraphobia_SeparationAnx_SocialAnx</summary>
  <p>
    <img src="./MF_perQ/prediction_Panic_Agoraphobia_SeparationAnx_SocialAnx.png" width=100%/>
  </p>
</details>

#### Prediction performance for harmonized data

We also compare prediction performance with different setting. The list of models are:
- `MF_direct_80` : After normalizing each questionnaire, concatenate all questions and perform matrix factorization directly. By cross-validation, the intrinsic dimension detect is 80.
- `FA_55_KNNimpute` : We use k-NN to impute missing entries. Then we perform factor analysis with varimax rotation on the matrix as in `MF_direct_80`. Cross validation result reveals the intrinsic dimension under Factor Analysis is 55.
- `impute_15` : We perform matrix factorization on individual questionnaires, the factors representing subject embeddings are then concatenated (i.e., the factor level factorization we mentioned above). After imputation, the reconstructed matrix is used as input feature for the prediction model (148 dimension/factors). Dimension 15 is chosen by cross-validation.
- `impute_15Factor` : Instead of using the reconstructed matrix as in `impute_15`, we use the 15-D embeddings of factors as input feature for the prediction model.

![](./figure/prediction_summary.png)
<!-- ![](./figure/prediction_f1.png) -->

Though there is some observable difference in the AUC, the difference is **not** statistically significant. The significance test is done by

- Choose any pairs of models $(A, B)$.
- Randomly shuffle subjects and perform stratified train-validate-test split. Repeat 1000 times.
- Train and fine tune model using train and validate dataset. Perform inference on test set.
- Compute AUCs on test set for both model $A$ and $B$.
- Perform Delong Test to compute p-value on the difference of AUCs between $A$ and $B$. This gives 1000 p-values.
- Count the number of significance (p-values $< 0.05$) out of 1000 trials.

## Achieved results

<details>
<summary>Direct embedding </summary>

**Direct Embedding**

Using the boxed-constrained non-negative matrix factorization with spartiy control, we factorize the normalized matrix to obtain subject and question embedding.

![](./figure/2022-04-07-16-28-34.png)

![](./figure/2022-04-07-16-29-55.png)

Through the question embedding, there are some clear relationship between questionnaires. For instance,

- the 1-th factor (1-st row) shows the relationship between questions in `SWAN` and `Attention` sub-scale in `CBCL`
- the 3-rd factor shows the relationship between questions in `SWAN` and `PANAS`
- the 6-th factor shows the relationship between questions in `C3SR` and `SRS`
- the 9-th factor shows the relationship between questions in `SCQ`, `C3SR`, `YSR`

**Top components in each factor**

By normalizing the question loadings in each questionnaire, we can also list the top 5 questionnaires in each factors (Apologize for a shift in factor indexing):

<p align="center">
<img src="./figure/2022-04-07-16-35-46.png" width="400">
<img src="./figure/2022-04-07-16-36-11.png" width="373">
</p>

Particularly, we could also extract the top 15 questions in each factor for more detailed understanding. We expected that most factors are unique to particular questionnaire (other it does not make sense to perform multiple surveys). Interestingly, some factors captured "reasonable" relationship across questionnaires. (Factor 6, 8, 10?)

```
=============== Factor 1 ===============
1: [SWAN_07]--7. Keeps track of things necessary for activities (doesn't lose them)
2: [SWAN_01]--1. Gives close attention to detail and avoids careless mistakes
3: [SWAN_02]--2. Sustains attention on tasks or play activities
4: [SWAN_06]--6. Engages in tasks that require sustained mental effort
5: [SWAN_04]--4. Follows through on instructions and finishes school work and chores
6: [SWAN_05]--5. Organizes tasks and activities
7: [SWAN_08]--8. Ignores extraneous stimuli
8: [SWAN_09]--9. Remembers daily activities
9: [SWAN_03]--3. Listens when spoken to directly
10: [SWAN_16]--16. Reflects on questions (controls blurting out answers)
11: [SWAN_11]--11. Stays seated (when required by class rules or social conventions)
12: [SWAN_10]--10. Sits still (controls movement of hands or feet or controls squirming)
13: [SWAN_14]--14. Settles down and rests (controls excessive talking)
14: [SWAN_18]--18. Enters into conversation and games without interrupting or intruding
15: [SWAN_15]--15. Modulates verbal activity (controls excessive talking)
```

```
=============== Factor 2 ===============
1: [YSR_93]--93. I talk too much
2: [YSR_09]--9. I can't get my mind off certain thoughts (describe):
3: [YSR_78]--78. I am inattentive or easily distracted
4: [YSR_112]--112. I worry a lot
5: [YSR_46]--46. Parts of my body twitch or make me nervous (describe):
6: [YSR_80]--80. I stand up for my rights
7: [YSR_17]--17. I daydream a lot
8: [YSR_98]--98. I like to help others
9: [YSR_71]--71. I am self-conscious or easily embarrassed
10: [YSR_106]--106. I like to be fair to others
11: [C3SR_31]--31. I talk too much.
12: [YSR_109]--109. I like to help other people when I can
13: [YSR_92]--92. I like to make others laugh
14: [YSR_73]--73. I can work well with my hands
15: [YSR_10]--10. I have trouble sitting still
```

```
=============== Factor 3 ===============
1: [PANAS_07]--7. Scared
2: [PANAS_14]--14. Inspired
3: [PANAS_12]--12. Alert
4: [PANAS_15]--15. Nervous
5: [PANAS_17]--17. Attentive
6: [PANAS_08]--8. Hostile
7: [RBS_43]--Fascination, preoccupation with movement / things that move (e.g., fans, clocks)
8: [PANAS_03]--3. Excited
9: [PANAS_01]--1. Interested
10: [PANAS_10]--10. Proud
11: [PANAS_16]--16. Determined
12: [Barratt_P1_Edu]--Parent 1 level of education
13: [Barratt_P2_Edu]--Parent 2 level of education
14: [PANAS_13]--13. Ashamed
15: [TRF_112]--112. Worries
```

```
=============== Factor 4 ===============
1: [TRF_67]--67. Disrupts class discipline
2: [TRF_24]--24. Disturbs other pupils
3: [TRF_23]--23. Disobedient at school
4: [TRF_41]--41. Impulsive or acts without thinking
5: [TRF_19]--19. Demands a lot of attention
6: [TRF_53]--53. Talks out of turn
7: [TRF_06]--6. Defiant, talks back to staff
8: [TRF_73]--73. Behaves irresponsibly (describe):
9: [TRF_28]--28. Breaks school rules
10: [TRF_10]--10. Can't sit still, restless or hyperactive
11: [TRF_03]--3. Argues a lot
12: [TRF_100]--100. Fails to carry out assigned tasks
13: [TRF_22]--22. Difficulty following directions
14: [TRF_93]--93. Talks too much
15: [TRF_02]--2. Hums or makes other odd noises in class
```

```
=============== Factor 5 ===============
1: [C3SR_36]--36. I learn more slowly than other kids my age.
2: [C3SR_18]--18. I have trouble finishing things
3: [C3SR_33]--33. I have trouble with reading.
4: [C3SR_13]--13. I have trouble with spelling.
5: [C3SR_03]--3. It is hard for me to pay attention to details.
6: [C3SR_14]--14. I lose track of what I am supposed to do
7: [C3SR_09]--9. I have trouble understanding what I read.
8: [C3SR_02]--2. I struggle to complete hard tasks
9: [C3SR_15]--15. I have trouble playing or doing things quietly. 
10: [C3SR_27]--27. I have trouble concentrating.
11: [C3SR_29]--29. I am restless.
12: [C3SR_16]--16 I get distracted by things that are going on around me
13: [C3SR_05]--5. I can’t pay attention for long.
14: [C3SR_04]--4. It is hard for me to sit still
15: [C3SR_38]--38. I have trouble with math.
```

```
=============== Factor 6 ===============
1: [SCQ_15]--15. Does she/he ever have any mannerisms or odd ways of moving her/his hands or fingers, such as flapping or moving her/his fingers in front or her/his eyes?
2: [SCQ_12]--12. Does she/he ever seem to be more interested in parts of a toy or an object (e.g., spinning the wheels of a car), rather than in using the object as it was intended?
3: [SCQ_11]--11. Does she/he ever have any interests that preoccupy her/him and might seem off to other people (e.g., traffic lights, drainpipes, or timetables?)
4: [SCQ_07]--7. Does she/he ever say the same thing over and over again?
5: [SCQ_08]--8. Does she/he ever have things that she/he seems to have to do in a very particular way or order or rituals that she/he insists that you go though?
6: [ASSQ_11]--uses language freely but fails to make adjustments to fit social contexts or the needs of different listeners
7: [ASSQ_03]--lives somewhat in a world of his/her own with restricted idiosyncratic intellectual interests
8: [SCQ_16]--16. Does she/he ever have any complicated movements of her/his whole body, such as spinning or repeatedly bouncing up and down?
9: [ASSQ_04]--accumulates facts on certain subjects (good rote memory) but does not really understand the meaning
10: [ASSQ_24]--shows idiosyncratic attachment to objects (i.e. may get strangely attached to objects as if they were people)
11: [SRS_16]--16. Avoids eye contact or has unusual eye contact.
12: [ASSQ_22]--has difficulties in completing simple daily activities because of compulsory repetition of certain actions or thoughts (i.e. any habits that s/he just has to do?)
13: [ASSQ_16]--can be with other children but only on his/her terms
14: [ASSQ_23]--has special routines; insists on no change (i.e. may need to have exactly the same change; troubles with even the slightest change in his/her environment, or routines or activities)
15: [SRS_50]--50. Has repetitive, odd behaviors such as hand flapping or rocking.
```

```
=============== Factor 7 ===============
1: [CBCL_08]--Can't concentrate, can't pay attention for long
2: [PANAS_14]--14. Inspired
3: [PANAS_07]--7. Scared
4: [PANAS_15]--15. Nervous
5: [PANAS_16]--16. Determined
6: [PANAS_12]--12. Alert
7: [PANAS_10]--10. Proud
8: [RBS_43]--Fascination, preoccupation with movement / things that move (e.g., fans, clocks)
9: [CBCL_10]--Can't sit still, restless, or hyperactive
10: [CBCL_16]--Cruelty, bullying, or meanness to others
11: [CBCL_69]--Secretive, keeps things to self
12: [CBCL_01]--Acts too young for his/her age
13: [CBCL_14]--Cries a lot
14: [CBCL_11]--Clings to adults or too dependent
15: [PANAS_17]--17. Attentive
```

```
=============== Factor 8 ===============
1: [SCQ_35]--35. Does she/he play any pretend or make-believe games?
2: [SCQ_39]--39. Does she/he ever play imaginative games with another child in such a way that you can tell that each child understands what the other is pretending?
3: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
4: [SCQ_36]--36. Does she/he seem interested in other children of approximately the same age whom she/he does not know?
5: [YSR_106]--106. I like to be fair to others
6: [YSR_109]--109. I like to help other people when I can
7: [YSR_88]--88. I enjoy being with people
8: [C3SR_21]--21. People like being around me.
9: [SCQ_32]--32. If she/he wants something or wants help, does she/he look at you and use gestures with sounds or words to get your attention?
10: [SCQ_22]--22. Does she/he ever spontaneously point at things around her/him just to show you things (not because she/he wants them)?
11: [YSR_60]--60. I like to try new things
12: [YSR_98]--98. I like to help others
13: [YSR_59]--59. I can be pretty friendly
14: [SCQ_29]--29. Does she/he ever offer to share things other than food with you?
15: [PANAS_08]--8. Hostile
```

```
=============== Factor 9 ===============
1: [SCQ_32]--32. If she/he wants something or wants help, does she/he look at you and use gestures with sounds or words to get your attention?
2: [SCQ_39]--39. Does she/he ever play imaginative games with another child in such a way that you can tell that each child understands what the other is pretending?
3: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
4: [SCQ_35]--35. Does she/he play any pretend or make-believe games?
5: [SCQ_25]--25. Does she/he shake her/his head to indicate no?
6: [SCQ_24]--24. Does she/he nod her/his head to indicate yes?
7: [SCQ_36]--36. Does she/he seem interested in other children of approximately the same age whom she/he does not know?
8: [SCQ_29]--29. Does she/he ever offer to share things other than food with you?
9: [SCQ_20]--20. Does she/he ever talk with you just to be friendly (rather than to get something)?
10: [SRS_40]--40. Is imaginative, good at pretending (without losing touch with reality).
11: [SCQ_31]--31. Does she/he ever try to comfort you if you are sad or hurt?
12: [SCQ_22]--22. Does she/he ever spontaneously point at things around her/him just to show you things (not because she/he wants them)?
13: [SRS_26]--26. Offers comfort to others when they are sad.
14: [SCQ_09]--9. Does her/his facial expression usually seem appropriate to the particular situation, as far as you can tell?
15: [YSR_103]--103. I am unhappy, sad, or depressed
```

```
=============== Factor 10 ===============
1: [TRF_13]--13. Confused or seems to be in a fog
2: [TRF_80]--80. Stares blankly
3: [TRF_17]--17. Daydreams or gets lost in his/her thoughts
4: [TRF_04]--4. Fails to finish things he/she starts
5: [CBCL_107]--Wets self during the day
6: [TRF_49]--49. Has difficulty learning
7: [TRF_08]--8. Can't concentrate, can't pay attention for long
8: [TRF_61]--61. Poor school work
9: [TRF_100]--100. Fails to carry out assigned tasks
10: [TRF_78]--78. Inattentive or easily distracted
11: [TRF_22]--22. Difficulty following directions
12: [CBCL_35]--Feels worthless or inferior
13: [CBCL_25]--Doesn't get along with other kids
14: [TRF_102]--102. Underactive, slow moving, or lacks energy
15: [CBCL_27]--Easily jealous
```

```
=============== Factor 11 ===============
1: [Barratt_P2_Edu]--Parent 2 level of education
2: [CBCL_21]--Destroys things belonging to his/her family or others
3: [SDQ_26]--Overall, do you think that your child has difficulties in one or more of the following areas: emotions, concentration, behavior or being able to get on with other people?
4: [SDQ_06]--Rather solitary, prefers to play alone (for 11-17 year olds: Would rather be alone than with other youth)
5: [Barratt_P1_Edu]--Parent 1 level of education
6: [SRS_61]--61. Is inflexible, has a hard time changing his or her mind.
7: [SDQ_29_c]--Do the difficulties interfere with your child's everyday life in the following areas? CLASSROOM LEARNING
8: [SRS_24]--24. Has more difficulty than other children with changes in his or her routine.
9: [SRS_30]--30. Becomes upset in a situation with lots of things going on.
10: [SRS_04]--4. When under stress, he or she shows rigid or inflexible patterns of behavior that seem odd.
11: [SRS_42]--42. Seems overly sensitive to sounds, textures, or smells.
12: [SRS_31]--31. Can’t get his or her mind off something once he or she starts thinking about it.
13: [CBCL_29]--Fears certain animals, situations, or places, other than school (describe)
14: [TRF_45]--45. Nervous, high-strung; or tense
15: [CBCL_25]--Doesn't get along with other kids
```

```
=============== Factor 12 ===============
1: [SCQ_24]--24. Does she/he nod her/his head to indicate yes?
2: [SCQ_25]--25. Does she/he shake her/his head to indicate no?
3: [SCQ_03]--3. Does she/he ever use odd phrases or say the same thing over and over in almost exactly the same way (either phrases that she/he hears other people use or ones that she/he makes up?
4: [SCQ_32]--32. If she/he wants something or wants help, does she/he look at you and use gestures with sounds or words to get your attention?
5: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
6: [ASSQ_04]--accumulates facts on certain subjects (good rote memory) but does not really understand the meaning
7: [ASSQ_10]--is surprisingly good at some things and surprisingly poor at others
8: [SCQ_22]--22. Does she/he ever spontaneously point at things around her/him just to show you things (not because she/he wants them)?
9: [SCQ_08]--8. Does she/he ever have things that she/he seems to have to do in a very particular way or order or rituals that she/he insists that you go though?
10: [ASSQ_05]--has a literal understanding of ambiguous and metaphoric language (i.e. takes things literally; troubles understanding expressions or metaphors)
11: [ASSQ_01]--is old-fashioned or precocious
12: [SCQ_07]--7. Does she/he ever say the same thing over and over again?
13: [SCQ_06]--6. Does she/he ever use words that she/he seems to have invented or made up her/himself; put things in odd, indirect ways; or use metaphorical ways of saying things (e.g., saying hot rain for steam)?
14: [SCQ_14]--14. Does she/he ever seem to be unusually interested in the sight, feel, sound, taske, or smell of things or people?
15: [SCQ_13]--13. Does she/he ever have any special interests that are unusual in their intensity but otherwise appropriate for her/his age and peer group (e.g., trains or dinosaurs)?
```

```
=============== Factor 13 ===============
1: [SRS_43]--43. Separates easily from caregivers.
2: [SCQ_03]--3. Does she/he ever use odd phrases or say the same thing over and over in almost exactly the same way (either phrases that she/he hears other people use or ones that she/he makes up?
3: [SCQ_05]--5. Does she/he ever get her/his pronouns mixed up (e.g., saying you or she/he for I)?
4: [SCQ_07]--7. Does she/he ever say the same thing over and over again?
5: [SRS_21]--21. Is able to imitate others' actions.
6: [SCQ_06]--6. Does she/he ever use words that she/he seems to have invented or made up her/himself; put things in odd, indirect ways; or use metaphorical ways of saying things (e.g., saying hot rain for steam)?
7: [SRS_17]--17. Recognizes when something is unfair.
8: [SRS_15]--15. Is able to understand the meaning of other people's tone of voice and facial expressions.
9: [SRS_40]--40. Is imaginative, good at pretending (without losing touch with reality).
10: [SCQ_08]--8. Does she/he ever have things that she/he seems to have to do in a very particular way or order or rituals that she/he insists that you go though?
11: [SRS_55]--55. Knows when he or she is talking too loud or making too much noise.
12: [SRS_07]--7. Is aware of what others are thinking or feeling.
13: [SRS_45]--45. Focuses his or her attention to where others are looking or listening.
14: [SRS_09]--9. Clings to adults, seems too dependent on them.
15: [SRS_48]--48. Has a sense of humor, understands jokes.
```

```
=============== Factor 14 ===============
1: [ASSQ_15]--wishes to be sociable but fails to make relationships with peers
2: [ASSQ_17]--lacks best friend
3: [SRS_18]--18. Has difficulty making friends, even when trying his or her best.
4: [SRS_29]--29. Is regarded by other children as odd or weird.
5: [SRS_37]--37. Has difficulty relating to peers.
6: [SCQ_19]--19. Does she/he have any particular friends or a best friend?
7: [ASSQ_25]--is bullied by other children
8: [SRS_33]--33. Is socially awkward, even when he or she is trying to be polite.
9: [SRS_57]--57. Gets teased a lot.
10: [SRS_22]--22. Plays appropriately with children his or her age.
11: [CBCL_73]--Sexual problems (describe)
12: [CBCL_72]--Sets fires
13: [CBCL_30]--Fears going to school
14: [SRS_13]--13. Is awkward is turn-taking interactions with peers (e.g., doesn't seem to understand the give-and-take of conversations)
15: [SRS_05]--5. Doesn't recognize when others are trying to take advantage of him or her.
```

```
=============== Factor 15 ===============
1: [SCQ_22]--22. Does she/he ever spontaneously point at things around her/him just to show you things (not because she/he wants them)?
2: [SCQ_23]--23. Does she/he ever use gestures, other than pointing or pulling your hand, to let you know what she/he wants?
3: [SCQ_25]--25. Does she/he shake her/his head to indicate no?
4: [SCQ_24]--24. Does she/he nod her/his head to indicate yes?
5: [SCQ_21]--21. Does she/he ever spontaneously copy you (or other people) or what you are doing (such as vacuuming, gardening, or mending things)?
6: [SRS_31]--31. Can’t get his or her mind off something once he or she starts thinking about it.
7: [SRS_28]--28. Thinks or talks about the same thing over and over.
8: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
9: [SRS_42]--42. Seems overly sensitive to sounds, textures, or smells.
10: [SRS_24]--24. Has more difficulty than other children with changes in his or her routine.
11: [SRS_04]--4. When under stress, he or she shows rigid or inflexible patterns of behavior that seem odd.
12: [SRS_30]--30. Becomes upset in a situation with lots of things going on.
13: [SRS_61]--61. Is inflexible, has a hard time changing his or her mind.
14: [SRS_20]--20. Shows unusual sensory interests (e.g., mouthing or spinning objects) or strange ways of playing with toys
15: [SRS_01]--1. Seems much more fidgety in social situations than when alone.
```

```
=============== Factor 16 ===============
1: [SRS_26]--26. Offers comfort to others when they are sad.
2: [TRF_80]--80. Stares blankly
3: [TRF_17]--17. Daydreams or gets lost in his/her thoughts
4: [TRF_13]--13. Confused or seems to be in a fog
5: [TRF_102]--102. Underactive, slow moving, or lacks energy
6: [SCQ_31]--31. Does she/he ever try to comfort you if you are sad or hurt?
7: [ASSQ_12]--lacks empathy (i.e. tends to see things only from his/her own perspective, and has troubles seeing things from other's perspective)
8: [SRS_38]--38. Responds appropriately to mood changes in others (e.g., when a friend's or playmate's mood changes from happy to sad).
9: [SRS_15]--15. Is able to understand the meaning of other people's tone of voice and facial expressions.
10: [TRF_60]--60. Apathetic or unmotivated
11: [SCQ_13]--13. Does she/he ever have any special interests that are unusual in their intensity but otherwise appropriate for her/his age and peer group (e.g., trains or dinosaurs)?
12: [TRF_75]--75. Too shy or timid
13: [SRS_07]--7. Is aware of what others are thinking or feeling.
14: [TRF_78]--78. Inattentive or easily distracted
15: [TRF_08]--8. Can't concentrate, can't pay attention for long
```

```
=============== Factor 17 ===============
1: [SCQ_24]--24. Does she/he nod her/his head to indicate yes?
2: [SCQ_25]--25. Does she/he shake her/his head to indicate no?
3: [ASSQ_25]--is bullied by other children
4: [SRS_57]--57. Gets teased a lot.
5: [SDQ_17]--Kind to younger children
6: [SCQ_19]--19. Does she/he have any particular friends or a best friend?
7: [CBCL_72]--Sets fires
8: [SRS_05]--5. Doesn't recognize when others are trying to take advantage of him or her.
9: [SRS_18]--18. Has difficulty making friends, even when trying his or her best.
10: [SCQ_05]--5. Does she/he ever get her/his pronouns mixed up (e.g., saying you or she/he for I)?
11: [SDQ_28]--Do the difficulties upset or distress your child?
12: [C3SR_28]--28. I am perfect in every way.
13: [SRS_58]--58. Concentrates too much on parts of things rather than seeing the whole picture. For example, if asked to describe what happened in a story, he or she may talk only about the kind of clothes the characters were wearing.
14: [TRF_49]--49. Has difficulty learning
15: [SRS_62]--62. Give unusual or illogical reasons for doing things.
```

```
=============== Factor 18 ===============
1: [C3SR_26]--26. My parents are too harsh when they punish me.
2: [SRS_25]--25. Doesn't seem to mind being out of step with or "not on the same wavelength" as others.
3: [C3SR_37]--37. My parents are too strict with me.
4: [C3SR_19]--19. Punishment in my house is not fair.
5: [Barratt_P2_Edu]--Parent 2 level of education
6: [CBCL_35]--Feels worthless or inferior
7: [C3SR_22]--22. My parents expect too much from me.
8: [SCQ_05]--5. Does she/he ever get her/his pronouns mixed up (e.g., saying you or she/he for I)?
9: [C3SR_34]--34. My parents are too critical of me.
10: [YSR_23]--23. I disobey my school
11: [Barratt_P1_Edu]--Parent 1 level of education
12: [YSR_28]--28. I break rules at home, school, or elsewhere
13: [YSR_39]--39. I hang around with kids who get in trouble
14: [YSR_43]--43. I lie or cheat
15: [CBCL_37]--Gets in many fights
```

```
=============== Factor 19 ===============
1: [TRF_50]--50. Too fearful or anxious
2: [TRF_109]--109. Whining
3: [TRF_88]--88. Sulks a lot
4: [TRF_110]--110. Unclean personal appearance
5: [TRF_71]--71. Self-conscious or easily embarrassed
6: [TRF_42]--42. Would rather be alone than with others
7: [TRF_35]--35. Feels worthless or inferior
8: [TRF_45]--45. Nervous, high-strung; or tense
9: [TRF_75]--75. Too shy or timid
10: [TRF_106]--106. Overly anxious to please
11: [TRF_87]--87. Sudden changes in mood or feelings
12: [TRF_05]--5. There is very little that he/she enjoys
13: [TRF_65]--65. Refuses to talk
14: [TRF_81]--81. Feels hurt when criticized
15: [TRF_69]--69. Secretive, keeps things to self
```

```
=============== Factor 20 ===============
1: [SCQ_39]--39. Does she/he ever play imaginative games with another child in such a way that you can tell that each child understands what the other is pretending?
2: [SCQ_35]--35. Does she/he play any pretend or make-believe games?
3: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
4: [C3SR_05]--5. I can’t pay attention for long.
5: [C3SR_04]--4. It is hard for me to sit still
6: [C3SR_03]--3. It is hard for me to pay attention to details.
7: [SCQ_03]--3. Does she/he ever use odd phrases or say the same thing over and over in almost exactly the same way (either phrases that she/he hears other people use or ones that she/he makes up?
8: [C3SR_09]--9. I have trouble understanding what I read.
9: [C3SR_15]--15. I have trouble playing or doing things quietly. 
10: [TRF_110]--110. Unclean personal appearance
11: [C3SR_27]--27. I have trouble concentrating.
12: [C3SR_14]--14. I lose track of what I am supposed to do
13: [C3SR_33]--33. I have trouble with reading.
14: [TRF_50]--50. Too fearful or anxious
15: [YSR_10]--10. I have trouble sitting still
```

**Zooming into the `CBCL` dataset**

Previous section demonstrated that by concatenate all questionnaires information to perform matrix factorization, we could reveal relationship across questionnaires. But certainly it will also affect the factorization of individual questionnaires. 

As we have an extensive studies on factorizing `CBCL` questionnaires, a quick zoom-in to the `CBCL` factorization can probably inspire us on how we should evaluate the performance.

The question embedding: 

![](./figure/2022-04-07-16-48-30.png)

The loadings related to the cofounders `Young`, `Old`, `Male` and `Female`

![](./figure/2022-04-07-16-48-48.png)

The wordcloud for each factor is shown below:

<p float="left">
<img src="./simple_model-20/Factor1.png" width="150">
<img src="./simple_model-20/Factor2.png" width="150">
<img src="./simple_model-20/Factor3.png" width="150">
<img src="./simple_model-20/Factor4.png" width="150">
<img src="./simple_model-20/Factor5.png" width="150">
<img src="./simple_model-20/Factor6.png" width="150">
<img src="./simple_model-20/Factor7.png" width="150">
<img src="./simple_model-20/Factor8.png" width="150">
<img src="./simple_model-20/Factor9.png" width="150">
<img src="./simple_model-20/Factor10.png" width="150">
<img src="./simple_model-20/Factor11.png" width="150">
<img src="./simple_model-20/Factor12.png" width="150">
<img src="./simple_model-20/Factor13.png" width="150">
<img src="./simple_model-20/Factor14.png" width="150">
<img src="./simple_model-20/Factor15.png" width="150">
<img src="./simple_model-20/Factor16.png" width="150">
<img src="./simple_model-20/Factor17.png" width="150">
<img src="./simple_model-20/Factor18.png" width="150">
<img src="./simple_model-20/Factor19.png" width="150">
<img src="./simple_model-20/Factor20.png" width="150">
</p>

> **Remark 1**
> 
> Since we concatenate all questionnaires into one large matrix, we implicitly assume that the subject embedding is universal across all questionnaires. In particular, the dimension of subject embedding is influenced by the complexity of all questionnaires. Using blockwise cross-validation on imputating matrix entries, we found that the intrinsic dimension of the concatenated matrix is k=20. However, if we only consider `CBCL` dataset, the intrinsic dimension was k=7, which is much smaller.

> **Remark 2**
> 
> The assumption that there is a universal subject embedding for all questionnaires might or might not deteriorate the factorization result. If this is not a valid assumption, an effect called `negative transfer` may happen. As the name suggested, instead of concatenating useful information from different questionnaires, a lost in information happens to compensate the invalid assumption. This undersiable effect had been observed in the community developing `Collective matrix factorization (CMF)`. The problem setting of CMF is similar but more restrictive than ours: each subject had completed all questionnaires and `CMF` is equivalent to a multi-view matrix factorization.

</details>

<details>
<summary>Partial embedding </summary>

**Improvement for multi-questionnaires analysis**

To generalize the model to avoid making the universal subject embedding assumption, an intuitive idea is to sub-divide the subject embedding into `general` and `questionnaire specific` dimension. Specifically, there will be one universal subject embedding which depicts common pattern on subjects across all questionnaires, while a specific subject embedding will also be introduced to every single questionnaires. Mathematically, the model to factorize the concatenated matrix <!-- $M$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=M"> becomes:

<!-- $$
M \approx \overline{W} \cdot \overline{Q}^T + \sum_{i=1}^N P_{i} ( W_i Q_i^T) + C \cdot \overline{Q}_c^T
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=M%20%5Capprox%20%5Coverline%7BW%7D%20%5Ccdot%20%5Coverline%7BQ%7D%5ET%20%2B%20%5Csum_%7Bi%3D1%7D%5EN%20P_%7Bi%7D%20(%20W_i%20Q_i%5ET)%20%2B%20C%20%5Ccdot%20%5Coverline%7BQ%7D_c%5ET"></div>

where <!-- $P_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=P_i"> is a projection function for survey <!-- $i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=i"> which projects a submatrix onto the corresponding position at the concatenated matrix <!-- $M$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=M">. <!-- $\overline{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BW%7D"> is the general subject embedding and <!-- $\overline{Q}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BQ%7D"> corresponds to the general question loading. We also enforce boxed-constraint on $W_i$ and sparsity control on all learnable parameters.

> **Remark**
> 
> The main challenges of this approach is the optimization procedure and the choice of each <!-- $W_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=W_i"> and <!-- $\overline{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BW%7D"> dimension. At this early stage, we perform cross validation on each questionnaires as before to obtain the individual intrinsic dimension. The dimension of <!-- $\overline{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BW%7D"> is chosen to be half of the the direct model (which is k=10)
> Another challenge is the choice of the sparsity parameter. The result below used a universal sparsity parameter for all learnable variables.
> We also add one extra column in <!-- $C$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=C"> with all elements equal 1. This serves as a dimension to represent mean response of each question.


**Embedding**

Universal subject embedding <!-- $\overline{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BW%7D">
![](./figure/2022-04-08-09-28-36.png)

Question embedding corresponding to the universal subject embedding <!-- $\overline{Q}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BQ%7D">
![](./figure/2022-04-08-09-28-56.png)

Cofounder's loading <!-- $\overline{Q}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BQ%7D_c"> : `Young`, `Old`, `Male`, `Female`, and the mean
![](./figure/2022-04-08-09-33-50.png)

**Top components in each factor**

Based on <!-- $\overline{Q}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BQ%7D">, we could similarly order the top 5 surveys contributed to each universal factors:

<img src="./figure/2022-04-08-09-37-55.png" width="400">

For each factor, the top 15 questions are:

```
=============== Factor 1 ===============
1: [SDQ_25]--Good attention span, sees chores or homework through to the end
2: [SWAN_05]--5. Organizes tasks and activities
3: [SWAN_07]--7. Keeps track of things necessary for activities (doesn't lose them)
4: [SCQ_21]--21. Does she/he ever spontaneously copy you (or other people) or what you are doing (such as vacuuming, gardening, or mending things)?
5: [SWAN_08]--8. Ignores extraneous stimuli
6: [SWAN_01]--1. Gives close attention to detail and avoids careless mistakes
7: [SWAN_04]--4. Follows through on instructions and finishes school work and chores
8: [Barratt_P1_Edu]--Parent 1 level of education
9: [SWAN_06]--6. Engages in tasks that require sustained mental effort
10: [SWAN_02]--2. Sustains attention on tasks or play activities
11: [SWAN_10]--10. Sits still (controls movement of hands or feet or controls squirming)
12: [SDQ_29_c]--Do the difficulties interfere with your child's everyday life in the following areas? CLASSROOM LEARNING
13: [SDQ_15]--Easily distracted, concentration wanders
14: [SCQ_01]--1. Is she/he now able to talk using short phrases or sentences? If no, skip to question 8.
15: [Barratt_P2_Edu]--Parent 2 level of education
```

```
=============== Factor 2 ===============
1: [YSR_107]--107. I enjoy a good joke
2: [YSR_59]--59. I can be pretty friendly
3: [YSR_92]--92. I like to make others laugh
4: [YSR_06]--6. I like animals
5: [YSR_109]--109. I like to help other people when I can
6: [YSR_98]--98. I like to help others
7: [YSR_106]--106. I like to be fair to others
8: [YSR_88]--88. I enjoy being with people
9: [YSR_108]--108. I like to take life easy
10: [YSR_73]--73. I can work well with my hands
11: [YSR_60]--60. I like to try new things
12: [YSR_80]--80. I stand up for my rights
13: [YSR_15]--15. I am pretty honest
14: [YSR_08]--8. I have trouble concentrating or paying attention
15: [YSR_78]--78. I am inattentive or easily distracted
```

```
=============== Factor 3 ===============
1: [PANAS_08]--8. Hostile
2: [PANAS_03]--3. Excited
3: [PANAS_17]--17. Attentive
4: [RBS_43]--Fascination, preoccupation with movement / things that move (e.g., fans, clocks)
5: [PANAS_07]--7. Scared
6: [PANAS_01]--1. Interested
7: [PANAS_15]--15. Nervous
8: [PANAS_14]--14. Inspired
9: [PANAS_12]--12. Alert
10: [PANAS_10]--10. Proud
11: [PANAS_16]--16. Determined
12: [PANAS_13]--13. Ashamed
13: [PANAS_09]--9. Enthusiastic
14: [RBS_44]--Overall, if you “lump together” all of the behaviors described in this questionnaire, how much of a problem are these repetitive behaviors (both for the person with autism, as well as how they affect the people around them)? Please rate on a scale from 1 to 100, where 1 = not a problem at all, and 100 = as bad as you can imagine:
15: [PANAS_02]--2. Distressed
```

```
=============== Factor 4 ===============
1: [TRF_08]--8. Can't concentrate, can't pay attention for long
2: [TRF_78]--78. Inattentive or easily distracted
3: [TRF_04]--4. Fails to finish things he/she starts
4: [TRF_17]--17. Daydreams or gets lost in his/her thoughts
5: [TRF_49]--49. Has difficulty learning
6: [TRF_22]--22. Difficulty following directions
7: [TRF_61]--61. Poor school work
8: [TRF_15]--15. Fidgets
9: [TRF_72]--72. Messy work
10: [TRF_100]--100. Fails to carry out assigned tasks
11: [TRF_92]--92. Underachieving, not working up to potential
12: [TRF_41]--41. Impulsive or acts without thinking
13: [TRF_53]--53. Talks out of turn
14: [TRF_60]--60. Apathetic or unmotivated
15: [TRF_10]--10. Can't sit still, restless or hyperactive
```

```
=============== Factor 5 ===============
1: [C3SR_06]--6. I am good at some things
2: [C3SR_11]--11. I like it when people say good things about me.
3: [C3SR_07]--7. I make mistakes.
4: [C3SR_16]--16 I get distracted by things that are going on around me
5: [C3SR_23]--23. I enjoy myself when I do my favorite activities. 
6: [C3SR_21]--21. People like being around me.
7: [C3SR_25]--25. I am happy and cheerful.
8: [C3SR_27]--27. I have trouble concentrating.
9: [C3SR_02]--2. I struggle to complete hard tasks
10: [C3SR_05]--5. I can’t pay attention for long.
11: [C3SR_04]--4. It is hard for me to sit still
12: [C3SR_03]--3. It is hard for me to pay attention to details.
13: [C3SR_14]--14. I lose track of what I am supposed to do
14: [C3SR_18]--18. I have trouble finishing things
15: [CBCL_38]--Gets teased a lot
```

```
=============== Factor 6 ===============
1: [ASSQ_10]--is surprisingly good at some things and surprisingly poor at others
2: [ASSQ_15]--wishes to be sociable but fails to make relationships with peers
3: [C3SR_11]--11. I like it when people say good things about me.
4: [C3SR_06]--6. I am good at some things
5: [ASSQ_17]--lacks best friend
6: [C3SR_23]--23. I enjoy myself when I do my favorite activities. 
7: [C3SR_07]--7. I make mistakes.
8: [ASSQ_13]--makes naïve and embarrassing remarks
9: [ASSQ_18]--lacks common sense
10: [ASSQ_11]--uses language freely but fails to make adjustments to fit social contexts or the needs of different listeners
11: [ASSQ_25]--is bullied by other children
12: [ASSQ_05]--has a literal understanding of ambiguous and metaphoric language (i.e. takes things literally; troubles understanding expressions or metaphors)
13: [ASSQ_20]--has clumsy, ill coordinated, ungainly, awkward movements or gestures
14: [ASSQ_16]--can be with other children but only on his/her terms
15: [C3SR_16]--16 I get distracted by things that are going on around me
```

```
=============== Factor 7 ===============
1: [PANAS_17]--17. Attentive
2: [PANAS_01]--1. Interested
3: [PANAS_08]--8. Hostile
4: [PANAS_03]--3. Excited
5: [RBS_43]--Fascination, preoccupation with movement / things that move (e.g., fans, clocks)
6: [PANAS_07]--7. Scared
7: [PANAS_14]--14. Inspired
8: [PANAS_12]--12. Alert
9: [PANAS_15]--15. Nervous
10: [PANAS_10]--10. Proud
11: [PANAS_16]--16. Determined
12: [CBCL_38]--Gets teased a lot
13: [CBCL_32]--Feels he/she has to be perfect
14: [AUDIT_09]--9. Have you or someone else been injured as a result of your drinking?
15: [CBCL_31]--Fears he/she might think or do something bad
```

```
=============== Factor 8 ===============
1: [SCQ_35]--35. Does she/he play any pretend or make-believe games?
2: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
3: [SCQ_22]--22. Does she/he ever spontaneously point at things around her/him just to show you things (not because she/he wants them)?
4: [SCQ_01]--1. Is she/he now able to talk using short phrases or sentences? If no, skip to question 8.
5: [SCQ_21]--21. Does she/he ever spontaneously copy you (or other people) or what you are doing (such as vacuuming, gardening, or mending things)?
6: [SCQ_23]--23. Does she/he ever use gestures, other than pointing or pulling your hand, to let you know what she/he wants?
7: [SCQ_39]--39. Does she/he ever play imaginative games with another child in such a way that you can tell that each child understands what the other is pretending?
8: [SCQ_25]--25. Does she/he shake her/his head to indicate no?
9: [SCQ_24]--24. Does she/he nod her/his head to indicate yes?
10: [SCQ_36]--36. Does she/he seem interested in other children of approximately the same age whom she/he does not know?
11: [PANAS_17]--17. Attentive
12: [SCQ_32]--32. If she/he wants something or wants help, does she/he look at you and use gestures with sounds or words to get your attention?
13: [CBCL_27]--Easily jealous
14: [PANAS_14]--14. Inspired
15: [PANAS_01]--1. Interested
```

```
=============== Factor 9 ===============
1: [SCQ_22]--22. Does she/he ever spontaneously point at things around her/him just to show you things (not because she/he wants them)?
2: [SCQ_34]--34. Does she/he ever spontaenously join in and try to copy the actions in social games, such as The Mulberry Bush or London Bridge is Falling Down?
3: [SCQ_23]--23. Does she/he ever use gestures, other than pointing or pulling your hand, to let you know what she/he wants?
4: [SCQ_35]--35. Does she/he play any pretend or make-believe games?
5: [SCQ_21]--21. Does she/he ever spontaneously copy you (or other people) or what you are doing (such as vacuuming, gardening, or mending things)?
6: [SCQ_32]--32. If she/he wants something or wants help, does she/he look at you and use gestures with sounds or words to get your attention?
7: [SCQ_39]--39. Does she/he ever play imaginative games with another child in such a way that you can tell that each child understands what the other is pretending?
8: [SCQ_25]--25. Does she/he shake her/his head to indicate no?
9: [SCQ_24]--24. Does she/he nod her/his head to indicate yes?
10: [C3SR_35]--35. I can’t do things right.
11: [SCQ_19]--19. Does she/he have any particular friends or a best friend?
12: [SCQ_36]--36. Does she/he seem interested in other children of approximately the same age whom she/he does not know?
13: [SCQ_29]--29. Does she/he ever offer to share things other than food with you?
14: [C3SR_17]--17. I break things when I am angry or upset.
15: [SCQ_04]--4. Does she/he ever use socially inappropriate questions or statements? For example, does she/he ever regularly ask personal questions or make personal comments at awkward times?
```

```
=============== Factor 10 ===============
1: [TRF_112]--112. Worries
2: [CBCL_107]--Wets self during the day
3: [SDQ_02]--Restless, overactive, cannot stay still for long
4: [SDQ_07]--Generally well behaved, usually does what adults request
5: [CBCL_35]--Feels worthless or inferior
6: [TRF_17]--17. Daydreams or gets lost in his/her thoughts
7: [SDQ_18]--Often lies or cheats
8: [CBCL_29]--Fears certain animals, situations, or places, other than school (describe)
9: [CBCL_25]--Doesn't get along with other kids
10: [CBCL_27]--Easily jealous
11: [SDQ_15]--Easily distracted, concentration wanders
12: [TRF_13]--13. Confused or seems to be in a fog
13: [CBCL_32]--Feels he/she has to be perfect
14: [CBCL_38]--Gets teased a lot
15: [TRF_80]--80. Stares blankly
```

**Zooming into the `CBCL` dataset again**

The question embedding specific to `CBCL` <!-- $Q_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Q_i">
![](./figure/2022-04-08-09-42-10.png)

Corresponding question loadings on cofounders specific to `CBCL` <!-- $\overline{Q}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Coverline%7BQ%7D_c">

![](./figure/2022-04-08-09-41-57.png)

Similarly, we could obtain the question word clouds of each specific factors:

<p float="left">
<img src="./bimodel-10U7S-specific/factor1.png" width="150">
<img src="./bimodel-10U7S-specific/factor2.png" width="150">
<img src="./bimodel-10U7S-specific/factor3.png" width="150">
<img src="./bimodel-10U7S-specific/factor4.png" width="150">
<img src="./bimodel-10U7S-specific/factor5.png" width="150">
<img src="./bimodel-10U7S-specific/factor6.png" width="150">
<img src="./bimodel-10U7S-specific/factor7.png" width="150">
</p>

and the question word clouds of each universal factors:

<p float="left">
<img src="./bimodel-10U7S-universal/factor1.png" width="150">
<img src="./bimodel-10U7S-universal/factor2.png" width="150">
<img src="./bimodel-10U7S-universal/factor3.png" width="150">
<img src="./bimodel-10U7S-universal/factor4.png" width="150">
<img src="./bimodel-10U7S-universal/factor5.png" width="150">
<img src="./bimodel-10U7S-universal/factor6.png" width="150">
<img src="./bimodel-10U7S-universal/factor7.png" width="150">
<img src="./bimodel-10U7S-universal/factor8.png" width="150">
<img src="./bimodel-10U7S-universal/factor9.png" width="150">
<img src="./bimodel-10U7S-universal/factor10.png" width="150">
</p>

As shown in the universal factors, since it is shared with all questionnaires, the number of dimension may not be optimal for `CBCL` and we can see that there are 4 factors which shares very similar pattern (Factor `1`, `4`, `5`, `7`).

</details>