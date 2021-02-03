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

% attn heads zeroed  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 25  | 9.5148  |  61 | 9.0731  | 62 | 0.012383069641729295  |
| 50  | 9.5126  |  61 | 10.0784  | 39  | 0.00013840128475640006  |
| 75  | 9.5031  |  61 | 8.5908  | 59  | 0.0002314483334680294  |
| 100  | 9.5148  |  61 | 10.6923  | 28  | 8.412031015701787e-19  |


#### Accumulated Layers

##### 25% & first

|first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.5148  |  61 | 10.3244  | 43  | 7.935984225091039e-09  |
| 2  | 9.5148  |  61 | 9.4169  | 57  | 0.5656014071240886  |
| 3  | 9.5148  |  61 | 10.03  | 47  | 0.000407506194640425  |
| 4  | 9.5148  |  61 | 9.7643  | 51  | 0.08818432766321974  |
| 5  | 9.5148  |  61 | 9.7451  | 50  | 0.13285507172370348  |
| 6  | 9.5148  |  61 | 8.3879  | 71  | 7.378532072267548e-11  |
| 7  | 9.5148  |  61 | 8.9743  | 67  | 0.0009275481980859393  |
| 8  | 9.5148  |  61 | 9.0749  | 60  | 0.01097840770438537  |
| 9  | 9.5148  |  61 | 8.8270  | 66  | 6.091713951110375e-05  |
| 10  | 9.5148  |  61 | 9.1708  | 63  | 0.027484950186029578  |
| 11  | 9.5148  |  61 | 9.0458  | 65  | 0.006538080798498917  |
| 12  | 9.5148  |  61 | 8.9925  | 64  | 0.0036312287933417333  |


##### 50% & first

| first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.5148  |  61 | 10.1602  | 49  | 1.3702604528697868e-05  |
| 2  | 9.5148  |  61 | 10.2421  | 44  | 1.8013998772544973e-06  |
| 3  | 9.5026  |  61 | 10.1000  | 47  | 0.000296375726547212  |
| 4  | 9.5003  |  61 | 9.8732  | 43  | 0.01941866551501846  |
| 5  | 9.5125  |  61 | 10.0180  | 42  | 0.0011095012975673297  |
| 6  | 9.5148  |  61 | 10.0795  | 37  | 9.583778036380851e-05  |
| 7  | 9.5148  |  61 | 9.6236  | 52  |  0.4982298914583977  |
| 8  | 9.5125  |  61 | 10.1145  | 41  | 5.0547610277615155e-05  |
| 9  | 9.5108  |  61 | 10.2620  | 38  | 5.673001789804774e-07  |
| 10  | 9.5126  |  61 | 10.2709  | 36  | 4.4752755275184794e-07  |
| 11  | 9.5126  |  61 | 10.5739  | 35  | 2.886097136876982e-14  |
| 12  | 9.5108  |  61 | 10.6043  | 31  | 7.97659821573582e-1  |

##### 75% & first

| first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.5148  |  61 | 10.3073  | 46  | 3.254812390260577e-07  |
| 2  | 9.5191  |  61 | 9.6839  | 51  | 0.4489956984952711  |
| 3  | 9.5148  |  61 | 9.9431  | 61  | 0.04830286159370075  |
| 4  | 9.5198  |  61 | 10.4486  | 63  | 0.002541301650698428  |
| 5  | 9.5176  |  61 | 9.9750  | 51  | 0.08813836739897143  |
| 6  | 9.4997  |  61 | 9.6628  | 65  | 0.49634135253221523  |
| 7  | 9.4811  |  61 | 10.1163  | 61  | 0.000742367731103421  |
| 8  | 9.4917  |  62 | 9.3955  | 48  | 0.6529481273527634  |
| 9  | 9.4953  |  62 | 9.5536  | 62  | 0.8081199884047994  |
| 10  | 9.4925 |  62 | 8.5859  | 66  | 0.06777615330855331  |
| 11  | 9.4958  |  62 | 8.2582  | 27  | 0.04062131838634473  |
| 12  | 9.4940  |  62 | 8.6062  | 27  | 0.13817647900221122  |


##### 100% & first

| first n layer  | control model log lexical frequency  |  control model unique token ratio % | dementia model log lexical frequency  | dementia model unique token ratio % | t-test p-value  |
|---|---|---|---|---|---|
| 1  | 9.5148  |  61 | 9.8863  | 59  | 0.01177192367353311  |
| 2  | 9.5148  |  61 | 9.9897  | 59  | 0.0016494624066151196  |
| 3  | 9.5148  |  61 | 10.3320  | 53  | 1.0612986232701784e-08  |
| 4  | 9.5148  |  61 | 10.6006  | 40  | 1.0792881881716632e-14  |
| 5  | 9.5119  |  61 | 10.7362  | 31  | 2.1883442162873216e-15  |
| 6  | 9.5148  |  61 | 10.5384  | 30  | 2.7488216232700678e-12  |
| 7  | 9.5148  |  61 | 10.5057  | 25  | 1.660174097902447e-10  |
| 8  | 9.5148  |  61 | 10.4242  | 27  | 1.9761916645866402e-08  |
| 9  | 9.5148  |  61 | 10.0599  | 27  | 0.0021311593308899145  |
| 10  | 9.5148  |  61 | 10.0472  | 28  | 0.0034749620681982277  |
| 11  | 9.5148  |  61 | 9.7256  | 26  | 0.23247187968501043  |
| 12  | 9.5148  |  61 | 9.3293  | 26  | 0.3186490189122912  |


