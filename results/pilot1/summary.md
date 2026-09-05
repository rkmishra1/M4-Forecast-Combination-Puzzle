# Forecast combination pilot — results summary

## Pooled performance (all frequencies)

| method                 |   mase |   smape |   rmsse |    owa |
|:-----------------------|-------:|--------:|--------:|-------:|
| naive2                 | 2.2008 |  0.1081 |  1.7439 | 1      |
| autoarima              | 1.8952 |  0.0967 |  1.4791 | 0.8777 |
| autoets                | 2.0443 |  0.1039 |  1.6096 | 0.9448 |
| autotheta              | 2.0352 |  0.1001 |  1.6119 | 0.9255 |
| chronos-t5-base        | 1.8722 |  0.0843 |  1.4647 | 0.8154 |
| chronos-t5-small       | 1.8255 |  0.0831 |  1.443  | 0.7989 |
| timesfm-2.5-200m       | 1.8293 |  0.0833 |  1.4348 | 0.8007 |
| combo/best_single_val  | 1.8224 |  0.0881 |  1.4309 | 0.8216 |
| combo/equal            | 1.8071 |  0.0842 |  1.4196 | 0.7998 |
| combo/inv_error_global | 1.7834 |  0.0822 |  1.3997 | 0.7855 |
| combo/inv_error_series | 1.7584 |  0.081  |  1.3785 | 0.7743 |
| combo/inv_rank_series  | 1.7681 |  0.0823 |  1.3877 | 0.7823 |
| combo/median           | 1.8131 |  0.084  |  1.4245 | 0.8003 |
| combo/nnls_series      | 1.8258 |  0.085  |  1.4286 | 0.8078 |
| combo/trim1_equal      | 1.7795 |  0.0829 |  1.3986 | 0.7878 |
| combo/oracle_best      | 1.3665 |  0.0623 |  1.0988 | 0.5988 |


## Mean MASE by frequency

| method                 |   daily |   hourly |   monthly |   quarterly |   weekly |   yearly |
|:-----------------------|--------:|---------:|----------:|------------:|---------:|---------:|
| naive2                 |   3.58  |    1.2   |     1.302 |       1.621 |    2.596 |    4.337 |
| autoarima              |   3.695 |    0.869 |     1.019 |       1.278 |    2.224 |    2.991 |
| autoets                |   3.715 |    1.676 |     0.94  |       1.276 |    2.324 |    2.772 |
| autotheta              |   3.499 |    1.656 |     0.966 |       1.293 |    2.38  |    3.07  |
| chronos-t5-base        |   3.787 |    0.709 |     0.952 |       1.33  |    2.079 |    3.364 |
| chronos-t5-small       |   3.565 |    0.757 |     0.976 |       1.361 |    2.086 |    2.945 |
| timesfm-2.5-200m       |   3.726 |    0.743 |     0.96  |       1.297 |    1.958 |    3.186 |
| combo/best_single_val  |   3.742 |    0.738 |     0.968 |       1.306 |    2.001 |    2.842 |
| combo/equal            |   3.562 |    0.856 |     0.926 |       1.252 |    2.077 |    2.833 |
| combo/inv_error_global |   3.562 |    0.765 |     0.922 |       1.248 |    2.059 |    2.807 |
| combo/inv_error_series |   3.59  |    0.718 |     0.914 |       1.241 |    1.953 |    2.837 |
| combo/inv_rank_series  |   3.6   |    0.745 |     0.916 |       1.251 |    1.958 |    2.828 |
| combo/median           |   3.553 |    0.751 |     0.911 |       1.253 |    2.169 |    3.056 |
| combo/nnls_series      |   3.731 |    0.747 |     0.936 |       1.272 |    1.994 |    3.136 |
| combo/trim1_equal      |   3.557 |    0.787 |     0.924 |       1.234 |    2.012 |    2.878 |
| combo/oracle_best      |   2.998 |    0.557 |     0.689 |       0.92  |    1.534 |    1.681 |


## Mean per-series MASE rank (lower is better, pooled)

| method                 |   mean_rank |
|:-----------------------|------------:|
| naive2                 |      11.44  |
| autoarima              |       9.368 |
| autoets                |      10.215 |
| autotheta              |       9.843 |
| chronos-t5-base        |       8.815 |
| chronos-t5-small       |       9.146 |
| timesfm-2.5-200m       |       8.835 |
| combo/best_single_val  |       8.702 |
| combo/equal            |       8.768 |
| combo/inv_error_global |       8.217 |
| combo/inv_error_series |       7.815 |
| combo/inv_rank_series  |       8.005 |
| combo/median           |       8.376 |
| combo/nnls_series      |       8.179 |
| combo/trim1_equal      |       8.172 |
| combo/oracle_best      |       2.103 |