# Forecast combination pilot — results summary

## Pooled performance (all frequencies)

| method                 |   mase |   smape |   rmsse |    owa |
|:-----------------------|-------:|--------:|--------:|-------:|
| naive2                 | 1.7071 |  0.142  |  1.4524 | 1      |
| autoarima              | 1.3482 |  0.1212 |  1.1416 | 0.8216 |
| autoets                | 1.3419 |  0.1206 |  1.1363 | 0.8178 |
| autotheta              | 1.3384 |  0.1187 |  1.1353 | 0.81   |
| chronos-bolt-base      | 1.3458 |  0.1191 |  1.1486 | 0.8134 |
| chronos-t5-base        | 1.3767 |  0.1246 |  1.1805 | 0.842  |
| chronos-t5-large       | 1.3629 |  0.1226 |  1.1685 | 0.8308 |
| chronos-t5-small       | 1.3829 |  0.1242 |  1.1856 | 0.8423 |
| timesfm-2.5-200m       | 1.3423 |  0.1197 |  1.1462 | 0.8145 |
| combo/best_single_val  | 1.3433 |  0.1203 |  1.1464 | 0.8172 |
| combo/equal            | 1.2857 |  0.1138 |  1.0972 | 0.7773 |
| combo/inv_error_global | 1.282  |  0.1136 |  1.0942 | 0.7755 |
| combo/inv_error_series | 1.2766 |  0.1132 |  1.09   | 0.7727 |
| combo/inv_rank_global  | 1.2791 |  0.1142 |  1.0907 | 0.7767 |
| combo/inv_rank_series  | 1.2779 |  0.1136 |  1.0916 | 0.7741 |
| combo/median           | 1.2858 |  0.1135 |  1.0999 | 0.7762 |
| combo/nnls_series      | 1.3251 |  0.1176 |  1.1296 | 0.8021 |
| combo/trim1_equal      | 1.2797 |  0.1136 |  1.0927 | 0.7747 |
| combo/oracle_best      | 0.9087 |  0.0829 |  0.8077 | 0.558  |


## Mean MASE by frequency

| method                 |   daily |   hourly |   monthly |   quarterly |   weekly |   yearly |
|:-----------------------|--------:|---------:|----------:|------------:|---------:|---------:|
| naive2                 |   3.278 |    1.193 |     1.26  |       1.627 |    2.777 |    3.938 |
| autoarima              |   3.252 |    0.909 |     0.926 |       1.196 |    2.252 |    3.431 |
| autoets                |   3.254 |    1.616 |     0.964 |       1.191 |    2.548 |    3.073 |
| autotheta              |   3.213 |    1.852 |     0.961 |       1.212 |    2.536 |    3.001 |
| chronos-bolt-base      |   3.148 |    0.788 |     0.948 |       1.242 |    2.117 |    3.201 |
| chronos-t5-base        |   3.215 |    0.688 |     0.97  |       1.267 |    1.953 |    3.31  |
| chronos-t5-large       |   3.169 |    0.687 |     0.957 |       1.254 |    1.958 |    3.304 |
| chronos-t5-small       |   3.167 |    0.734 |     0.98  |       1.274 |    2.076 |    3.31  |
| timesfm-2.5-200m       |   3.286 |    0.734 |     0.946 |       1.227 |    1.967 |    3.153 |
| combo/best_single_val  |   3.23  |    0.728 |     0.944 |       1.234 |    1.982 |    3.186 |
| combo/equal            |   3.129 |    0.808 |     0.907 |       1.18  |    1.964 |    3.002 |
| combo/inv_error_global |   3.13  |    0.722 |     0.904 |       1.174 |    1.936 |    3.002 |
| combo/inv_error_series |   3.128 |    0.683 |     0.897 |       1.168 |    1.857 |    3.012 |
| combo/inv_rank_global  |   3.174 |    0.687 |     0.907 |       1.158 |    1.85  |    2.979 |
| combo/inv_rank_series  |   3.139 |    0.705 |     0.896 |       1.171 |    1.859 |    3.014 |
| combo/median           |   3.173 |    0.67  |     0.9   |       1.17  |    1.906 |    3.059 |
| combo/nnls_series      |   3.234 |    0.743 |     0.919 |       1.209 |    2.01  |    3.22  |
| combo/trim1_equal      |   3.131 |    0.734 |     0.899 |       1.167 |    1.908 |    3.031 |
| combo/oracle_best      |   2.416 |    0.516 |     0.656 |       0.83  |    1.419 |    1.917 |


## Mean per-series MASE rank (lower is better, pooled)

| method                 |   mean_rank |
|:-----------------------|------------:|
| naive2                 |      14.003 |
| autoarima              |      10.169 |
| autoets                |      10.609 |
| autotheta              |      10.394 |
| chronos-bolt-base      |      10.776 |
| chronos-t5-base        |      11.415 |
| chronos-t5-large       |      11.178 |
| chronos-t5-small       |      11.58  |
| timesfm-2.5-200m       |      10.821 |
| combo/best_single_val  |      10.704 |
| combo/equal            |       9.687 |
| combo/inv_error_global |       9.448 |
| combo/inv_error_series |       9.267 |
| combo/inv_rank_global  |       9.518 |
| combo/inv_rank_series  |       9.391 |
| combo/median           |       9.531 |
| combo/nnls_series      |      10.161 |
| combo/trim1_equal      |       9.377 |
| combo/oracle_best      |       1.972 |