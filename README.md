# mlt-external

## Multi-questionnaire

All existing factorization methods, such as Factor Analysis, NMF, PCA cannot improve the prediction power of the factors extracted from CBCL dataset. Therefore, one may wish to concatenate information from multiple available questionnaires to enrich factors' information.

Some extensions on the previous box-constrained NMF algorithm are proposed. (Still in development)

## Supervisied NMF

Several supervised NMF model have been explored. The results are not very promising though.

## Regression model for Brain measurements

#### 2021-09-17
- First version of regression model with spatial regularization is uploaded.
- Brain measurements data are not included in this repo and `SpatialReg_algo.ipynb` is not self-contained.
- Output files (both statics and interactive) are not uploaded.

#### 2021-09-18
- `ABCD_with_Dx_20210917.csv` contains the ABCD-CBCL dataset together with the diagnostic information.
- `HBN_with_scan_20210621.csv` contains subjects in HBN dataset with brain scan.
- `HBN_with_Dx_20210917.csv` contains subjects in HBN dataset with diagnostic information.
- `load ABCD.ipynb` and `load_HBN.ipynb` can be run to generate the the csv files.
