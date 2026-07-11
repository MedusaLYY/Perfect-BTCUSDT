# Stage 9 Final Training Report

## Scope
- Selected final tree count using DEVELOPMENT only.
- Refit one final XGBoost model on all DEVELOPMENT rows.
- Trained final prior, momentum, and logistic reference models before FINAL_TEST access.

## Inner Final Split
| final_inner_fit_count | final_inner_purged_count | final_inner_early_stop_count | final_best_iteration | final_best_n_estimators | final_best_score | stopped_early | reached_max_estimators |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 733307 | 12 | 25908 | 56 | 57 | 0.690143 | True | False |

## Final XGBoost
| training_sample_count | training_role | purged_rows_used | final_test_rows_used | refit_used_early_stopping | boosted_rounds | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| 759227 | DEVELOPMENT | False | False | False | 57 | ccf4731fb6504a7bd92fb4a335b8cb0664642b0856e4c97943c0e76c70d258f9 |

## Reference Models
| model | training_sample_count | training_role | model_sha256 | p_up | alpha |
| --- | --- | --- | --- | --- | --- |
| prior | 759227 | DEVELOPMENT | 546b12bae938da803702e2f40240d7b1dda9ba4755d98b4b9a6817a637482aa3 | 0.509604 |  |
| momentum | 759227 | DEVELOPMENT | fa54662d4ed8660fcfe5c5ac0e3683478aec662c1508e0854a2b469bc3a396b4 |  | 1 |
| logistic | 759227 | DEVELOPMENT | 737e7ade4b708a6d000c2abb138e2b20178eb8ee1d88ce462743ed21491fd719 |  |  |

## Protocol Lock
| field | value |
| --- | --- |
| protocol_lock_sha256 | e8cb01015f46279927a913e9d16f7e16d8b2e524e017d25767485455524e295d |

