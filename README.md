This repo contains codes and scripts for simulating Dementia by breaking neural networks.

## Folders

- ```scripts```: folder that contains all codes, scripts and jupyter notebooks
    - ```data```: cleaned data. Data can be obtained by running ```start.py```
- ```results```: folder that contains the wrapped-up results from the scripts
    - ```notebooks```: folder that contains jupyter notebook visualization
    - ```cache-original```: the original GPT-2 model evaluation result on ADDreSS dataset

## Results

### Log Lexical Frequency

#### Combination of Layers, Full ADReSS Dataset, First Zeroing

% attn heads zeroed  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 25  | 9.48  |  71 | 9.52  | 54 | 0.836  |
| 50  | 9.47  |  71 | 10.24  | 41  | 0.000 |
| 75  | 9.44  |  72 | 8.46  | 64  | 0.000  |
| 100  | 9.46  |  72 | 10.67  | 27  | 0.000  |


#### Combination of Layers, Full ADReSS Dataset, Random Zeroing

% attn heads zeroed  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 25  | 9.46  |  72 | 9.74 | 61 | 0.120  |
| 50  | 9.46  |  72 | 10.11  | 45  | 0.000 |
| 75  | 9.46  |  72 | 10.41  | 41  | 0.000  |
| 100  | 9.46  |  72 | 10.67  | 27  | 0.000  |


#### Accumulated Layers

##### 25% & First

|first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.46  |  72 | 10.02  | 50  | 0.000  |
| 2  | 9.46  |  72 | 10.24  | 46  | 0.000  |
| 3  | 9.46  |  72 | 10.26  | 43  | 0.000  |
| 4  | 9.46  |  72 | 9.64  | 49  | 0.291  |
| 5  | 9.46  |  71 | 9.42 | 55  | 0.711  |
| 6  | 9.46  |  72 | 8.46  | 71  | 0.000  |
| 7  | 9.46  |  72 | 8.45  | 70  | 0.000  |
| 8  | 9.46  |  72 | 9.53  | 54  | 0.688  |
| 9  | 9.46  |  72 | 9.1  | 63  | 0.035  |
| 10  | 9.46  |  72 | 8.62  | 70  | 0.000  |
| 11  | 9.46  |  72 | 9.16  | 64  | 0.075  |
| 12  | 9.46  |  72 | 8.99  | 64  | 0.016  |


##### 50% & first

| first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.45  |  72 | 10.29  | 47  | 0.000  |
| 2  | 9.46  |  72 | 10.21  | 43  | 0.000  |
| 3  | 9.46  |  72 | 10.13  | 43  | 0.000 |
| 4  | 9.46  |  71 | 9.87  | 45  | 0.009  |
| 5  | 9.46  |  72 | 9.93  | 48  | 0.003  |
| 6  | 9.46  |  72 | 10.21  | 40  | 0.000  |
| 7  | 9.46  |  72 | 9.86  | 44  |  0.008  |
| 8  | 9.48  |  72 | 10.07  | 43  | 0.000  |
| 9  | 9.48  |  71 | 10.35  | 33  | 0.000  |
| 10  | 9.48  |  72 | 10.29  | 38  | 0.000  |
| 11  | 9.48  |  72 | 10.5  | 32  | 0.000  |
| 12  | 9.48  |  72 | 10.53  | 30  | 0.000  |

##### 75% & first

| first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.46  |  72 | 10.4  | 47  | 0.000  |
| 2  | 9.46  |  72 | 9.62  | 56  | 0.455  |
| 3  | 9.46  |  72 | 10.14  | 58  | 0.003  |
| 4  | 9.43  |  71 | 9.75  | 67  | 0.148  |
| 5  | 9.44  |  69 | 10.37  | 53  | 0.000  |
| 6  | 9.44  |  69 | 8.93  | 68  | 0.012  |
| 7  | 9.44  |  70 | 9.6  | 64  | 0.412  |
| 8  | 9.43  |  70 | 8.88  | 63  | 0.026  |
| 9  | 9.42  |  71 | 9.47  | 74  | 0.829  |
| 10  | 9.42 |  71 | 9.63  | 68  | 0.601  |
| 11  | 9.42  |  70 | 9.16  | 43  | 0.647  |
| 12  | 9.43  |  71 | 8.89  | 31  | 0.363  |


##### 100% & first

| first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.46  |  72 | 9.93  | 56  | 0.001  |
| 2  | 9.46  |  72 | 9.82  | 59  | 0.020  |
| 3  | 9.46  |  72 | 10.28  | 50  | 0.000  |
| 4  | 9.46  |  72 | 10.59  | 47  | 0.000  |
| 5  | 9.46  |  72 | 10.81  | 28  | 0.000  |
| 6  | 9.46  |  72 | 10.53  | 36  | 0.000  |
| 7  | 9.46  |  72 | 10.53  | 25  | 0.000  |
| 8  | 9.46  |  72 | 10.42  | 29  | 0.000  |
| 9  | 9.46  |  72 | 10.03  | 28  | 0.002  |
| 10  | 9.46  |  72 | 10.18  | 27  | 0.000  |
| 11  | 9.46  |  72 | 9.67  | 26  | 0.242 |
| 12  | 9.46  |  72 | 9.3  | 27  | 0.390  |


### 5 Fold Cross Validation Results

#### CCC, first, diff

| share  | train AUC (accuracy)  | test AUC (accuracy)  |
|---|---|---|
| 25  | 0.561 (0.554)  | 0.45 (0.452)  |
| 50  | 0.569 (0.446)  | 0.438 (0.45)  |
| 75  | 0.579 (0.593)  | 0.452 (0.454)  |
| 100  | 0.491 (0.475)  | 0.406 (0.451)  |

#### CCC, first, ratio

| share  | train AUC (accuracy)  | test AUC (accuracy)  |
|---|---|---|
| 25  | 0.688 (0.647)  | 0.651 (0.606)  |
| 50  | 0.707 (0.658)  | 0.672 (0.629)  |
| 75  | **0.721 (0.676)**  | **0.729 (0.69)**  |
| 100  | 0.661 (0.602)  | 0.671 (0.639)  |

#### ADReSS, first, diff

| share  | train AUC (accuracy)  | test AUC (accuracy)  |
|---|---|---|
| 25  | 0.72 (0.7)  | 0.69 (0.61)  |
| 50  | 0.75 (0.69)  | 0.73 (0.69)  |
| 75  | 0.77 (0.67)  | 0.7 (0.67)  |
| 100  | 0.62 (0.56)  | 0.57  |

#### ADReSS, first, ratio

| share  | train AUC (accuracy)  | test AUC (accuracy)  |
|---|---|---|
| 25  | 0.79 (0.71)  | 0.72 (0.63)  |
| 50  | **0.8 (0.71)**  | **0.76 (0.65)**   |
| 75  | 0.75 (0.67)  | 0.71 (0.72)  |
| 100  | 0.74 (0.68)  | 0.65 (0.59)  |
