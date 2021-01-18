This repo contains codes and scripts for simulating Dementia by breaking neural networks.

## Folders

- ```scripts```: folder that contains all codes, scripts and jupyter notebooks
    - ```data```: cleaned data. Data can be obtained by running ```start.py```
- ```results```: folder that contains the wrapped-up results from the scripts
    - ```notebooks```: folder that contains jupyter notebook visualization
    - ```cache-original```: the original GPT-2 model evaluation result on ADDreSS dataset

## Results

### Log Lexical Frequency

#### Combination of Layers

##### 25% & first & full

| model  | log lexical frequency  |  unique token ratio %|
|---|---|---|
| control  | 9.5148  | 61  |
| dementia  | 9.0731  | 62  |

t-test p-value: 0.012383069641729295

##### 50% & first & full

| model  | log lexical frequency  |  unique token ratio %|
|---|---|---|
| control  | 9.5148  | 61  |
| dementia  | 10.0783  | 39  |

t-test p-value: 0.00013840128475640006

##### 100% & first & full

| model  | log lexical frequency  |  unique token ratio %|
|---|---|---|
| control  | 9.5148  | 61  |
| dementia  | 10.6922  | 28 |

t-test p-value: 8.412031015701787e-19