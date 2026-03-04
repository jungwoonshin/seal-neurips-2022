# SEAL Corrective Energy Alignment — Results

## Comparison Table: Test Instance-F1 (%)

| Method                            | Bibtex          | Delicious  | Genbase | CAL500          | Eurlex-ev        | Expr_fun        | SPO_fun         |
| --------------------------------- | --------------- | ---------- | ------- | --------------- | ---------------- | --------------- | --------------- |
| **Cross-entropy (CE)**      | 42.40           | 29.89      | 47.37   | 33.58           | 42.19            | 37.50           | 27.81           |
|                                   |                 |            |         |                 |                  |                 |                 |
| **Energy network with GBI** |                 |            |         |                 |                  |                 |                 |
| SPEN                              | 42.99           | 24.20      | 32.13   | 37.24           | 41.86            | 36.74           | 27.95           |
| DVN                               | **45.95** | 24.87      | 77.92   | **47.74** | 25.49            | 31.47           | 29.35           |
| NCE ranking                       | 12.95           | 12.69      | 12.40   | 33.89           | 0.19             | 27.74           | 18.06           |
|                                   |                 |            |         |                 |                  |                 |                 |
| **SEAL-static**             |                 |            |         |                 |                  |                 |                 |
| margin                            | 43.11           | 28.08      | 57.45   | 33.91           | 42.15            | 38.13           | 28.15           |
| regression                        | 42.29           | 30.09      | 96.68   | 37.63           | 42.18            | 33.12           | 28.42           |
| NCE ranking                       | 43.03           | 30.08      | 37.82   | 37.82           | 42.11            | 37.78           | 28.29           |
|                                   |                 |            |         |                 |                  |                 |                 |
| **SEAL-dynamic**            |                 |            |         |                 |                  |                 |                 |
| margin (InfNet)                   | 42.86           | 29.75      | 96.53   | 36.69           | 41.83            | 37.81           | 28.43           |
| regression                        | 43.74           | 29.79      | 96.95   | 37.97           | 41.67            | 37.95           | 28.83           |
| regression-s                      | 44.53           | 29.87      | 97.81   | 38.95           | 42.17            | 37.95           | —              |
| NCE ranking                       | 44.76           | 34.67      | 97.32   | 41.62           | 41.62            | **38.28** | 28.83           |
|                                   |                 |            |         |                 |                  |                 |                 |
| **Ours (Corrective)**       | 44.35           | **36.05** | **98.99** | 41.35           | **47.95** | 37.32           | **31.78** |

Bold indicates the best result per dataset.

## Dataset Statistics

| Dataset   | Features | Labels | Train  | Val   | Test  |
| --------- | -------- | ------ | ------ | ----- | ----- |
| Bibtex    | 1,836    | 159    | 4,407  | 1,491 | 1,497 |
| CAL500    | 68       | 174    | 283    | 105   | 114   |
| Delicious | 500      | 983    | 9,698  | 3,210 | 3,197 |
| Eurlex-ev | 5,000    | 3,993  | 11,581 | 3,883 | 3,884 |
| Expr_fun  | 561      | 500    | 1,639  | 849   | 1,291 |
| Genbase   | 1,185    | 27     | 398    | 132   | 132   |
| SPO_fun   | 86       | 500    | 1,600  | 837   | 1,266 |

## Training Time Comparison (sec/epoch)

Baseline values from the original paper (Appendix D, Table 19), measured on TitanX GPU.
Ours measured on local hardware.

| Method                      | Bibtex | CAL500 | Delicious | Eurlex-ev | Expr_fun | Genbase | SPO_fun |
| --------------------------- | ------ | ------ | --------- | --------- | -------- | ------- | ------- |
| CE                          | 22.67  | 1.00   | 33.01     | 114.84    | 6.29     | 2.12    | 5.46    |
| SPEN                        | 28.04  | 2.53   | 37.31     | 136.10    | 10.99    | 3.57    | 11.01   |
| DVN                         | 32.12  | 1.87   | 55.24     | 129.94    | 10.95    | 2.79    | 10.02   |
| NCE (Energy Net)            | 13.97  | 2.82   | 22.03     | 89.33     | 3.71     | 3.68    | 3.83    |
| SEAL-margin                 | 212.94 | 5.78   | 44.60     | 260.10    | 27.96    | 8.73    | 35.24   |
| SEAL-regression             | 27.78  | 8.07   | 143.90    | 352.63    | 46.32    | 13.91   | 56.97   |
| SEAL-regression-s           | 179.97 | 8.48   | 45.33     | 221.33    | 42.87    | 10.79   | 77.92   |
| SEAL-NCE                    | 131.73 | 16.58  | 158.97    | 431.92    | 72.43    | 17.77   | 40.34   |
| SEAL-Ranking                | 218.37 | 7.04   | 317.63    | 408.79    | 41.24    | 10.74   | 26.83   |
| **Ours (Corrective)**       | 1.93   | 0.13   | 3.63      | 5.05      | 0.81     | 0.19    | 0.80    |

## Detailed Results: SEAL Corrective

| Dataset   | Val F1 | Test F1 | Best Epoch | Epochs | Total Time | sec/epoch |
| --------- | ------ | ------- | ---------- | ------ | ---------- | --------- |
| Bibtex    | 0.4549 | 0.4435  | 32         | 70     | 135.4s     | 1.93      |
| CAL500    | 0.4171 | 0.4135  | 59         | 70     | 8.9s       | 0.13      |
| Delicious | 0.3621 | 0.3605  | 14         | 70     | 254.3s     | 3.63      |
| Eurlex-ev | 0.4827 | 0.4795  | 171        | 300    | 1516.0s    | 5.05      |
| Expr_fun  | 0.3760 | 0.3732  | 60         | 70     | 56.9s      | 0.81      |
| Genbase   | 0.9932 | 0.9899  | 18         | 70     | 13.2s      | 0.19      |
| SPO_fun   | 0.3042 | 0.3178  | 22         | 70     | 55.9s      | 0.80      |

### Configuration

- `lr_energy = 0.001`, `lr_task = 0.001`
- `beta = 0.1`, `alpha = 1.0`, `min_margin = 0.1`
- `lambda1 = 0.01`, `lambda2 = 1.0`
- `hidden_dim = 768`, `energy_hidden = 768`
- `batch_size = 32`, `epochs = 70` (eurlex_ev: 300)
- Mode: inline (no correction interval)
- Optimizer: AdamW (weight decay 1e-4)
- Scheduler: ReduceLROnPlateau (patience 10, factor 0.5)
