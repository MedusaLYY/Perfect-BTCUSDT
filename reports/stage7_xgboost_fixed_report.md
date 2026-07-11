# Stage 7 Fixed XGBoost Report

## Scope
- Trained exactly one fixed XGBoost configuration: xgboost_fixed_v1.
- No hyperparameter search, probability calibration, threshold optimization, feature selection, outlier clipping, missing value filling, resampling, SHAP, permutation importance, or final deployment model training was performed.
- FINAL_TEST remained sealed and was not used for fit, early stopping, selection, feature importance, prediction, or metrics.

## Inputs and Hashes
| item | value |
| --- | --- |
| dataset_sha256 | 4099f37af884ce221710fd84cf45fcbc06ca786dbb3a2b2b7eb1d4ec1a3187fa |
| split_sha256 | 6f99c99a92e8687e63156a80df3f995b44cea12c53aff9ae18efe11037b4070b |
| stage6_prediction_sha256 | 4f4f81933e099a9b03705cff006fe30facdd1c2264dde3456f0fcab4991cf235 |
| dataset_manifest_sha256 | debff85c37643883005336c4c699efe4557dadd3076a801c6f0dfaa12eda1907 |
| feature_manifest_sha256 | 0edb0cfb0df991a63da7a5462bf6735db9eea13b41cd9ab5ccac8042c435507c |
| fold_manifest_sha256 | 5ddcb7e9d88f69acfd9f63197aaf6605f5fa09242e33e80eab352b64d23e923f |
| stage6_model_manifest_sha256 | fc321c8dd66b83598d2403e1f351d433df54f9c0f6367ae17fb1eb1529cf39ea |
| config_sha256 | e8bfab712202fc60fe8925ecbc38e832146972a24262580f8395ee5f6c6b559c |
| script_sha256 | 30492f342eb2b12839bb5d4d412540ec4fd207766a7cc61abe25f19bbb1892db |
| prediction_output_path | C:\ai\BTCUSDT\data\predictions\BTCUSDT_stage7_cv_xgboost_fixed_predictions.parquet |
| prediction_sha256 | abd0737f3da7850f24513e383ce196cbd15651c44abbcb125ba2e85cc25a8052 |

## Feature Check and Fixed Parameters
| item | value |
| --- | --- |
| feature_count | 63 |
| feature_list_sha256 | c2c2c84b40520359b3d437601a72ec6a4a26f26522794c7b4d9c900bd16a4491 |
| xgboost_version | 3.3.0 |
| n_jobs | 19 |
| fixed_threshold | 0.5 |

## Outer Fold and Inner Split Audit
| fold_name | inner_fit | inner_purged | inner_early_stop | best_iteration | best_n_estimators | best_score | reached_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2020 | 208686 | 12 | 25721 | 135 | 136 | 0.683213 | False |
| fold_2021 | 313428 | 12 | 25654 | 37 | 38 | 0.687931 | False |
| fold_2022 | 417735 | 12 | 25908 | 90 | 91 | 0.690331 | False |
| fold_2023 | 522855 | 12 | 25908 | 124 | 125 | 0.685877 | False |
| fold_2024 | 627899 | 12 | 25908 | 31 | 32 | 0.690689 | False |

## Dense Metrics by Fold
| fold_name | model_name | sample_count | positive_ratio | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2020 | xgboost_fixed_v1 | 104663 | 0.51769 | 0.546468 | 0.545425 | 0.564555 | 0.576653 | 0.0909684 | 0.686862 | 0.246873 |
| fold_2021 | xgboost_fixed_v1 | 104549 | 0.506758 | 0.523008 | 0.521569 | 0.530758 | 0.530913 | 0.0441528 | 0.692401 | 0.249623 |
| fold_2022 | xgboost_fixed_v1 | 105108 | 0.499334 | 0.542775 | 0.54287 | 0.55863 | 0.546435 | 0.0866328 | 0.68825 | 0.247558 |
| fold_2023 | xgboost_fixed_v1 | 105032 | 0.507141 | 0.54154 | 0.541011 | 0.551998 | 0.545255 | 0.0822471 | 0.689721 | 0.248275 |
| fold_2024 | xgboost_fixed_v1 | 105396 | 0.510437 | 0.529574 | 0.528423 | 0.54091 | 0.545219 | 0.0571902 | 0.690511 | 0.248684 |

## Non-overlap Metrics by Fold
| fold_name | model_name | sample_count | positive_ratio | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2020 | xgboost_fixed_v1 | 8721 | 0.518289 | 0.559569 | 0.55858 | 0.584544 | 0.591801 | 0.117281 | 0.681837 | 0.244396 |
| fold_2021 | xgboost_fixed_v1 | 8712 | 0.503903 | 0.530877 | 0.530095 | 0.547494 | 0.542532 | 0.0614402 | 0.690081 | 0.24847 |
| fold_2022 | xgboost_fixed_v1 | 8759 | 0.49766 | 0.55383 | 0.554176 | 0.57154 | 0.557266 | 0.109548 | 0.686184 | 0.246533 |
| fold_2023 | xgboost_fixed_v1 | 8753 | 0.513424 | 0.558437 | 0.557465 | 0.57092 | 0.567712 | 0.115217 | 0.685386 | 0.246133 |
| fold_2024 | xgboost_fixed_v1 | 8783 | 0.511556 | 0.542753 | 0.541635 | 0.554916 | 0.557316 | 0.0836554 | 0.688856 | 0.247859 |

## Pooled OOF Metrics
| subset_name | model_name | sample_count | positive_ratio | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | xgboost_fixed_v1 | 524748 | 0.508267 | 0.536675 | 0.535684 | 0.549265 | 0.54982 | 0.0718863 | 0.689549 | 0.248203 |
| NONOVERLAP_OFFSET_00 | xgboost_fixed_v1 | 43728 | 0.508965 | 0.549099 | 0.548073 | 0.565417 | 0.564156 | 0.0967862 | 0.68647 | 0.246679 |

## Fold Macro Mean and Std
| summary_type | model_name | sample_count | accuracy | balanced_accuracy | roc_auc | average_precision | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_macro_mean | xgboost_fixed_v1 | 104950 | 0.536673 | 0.53586 | 0.54937 | 0.548895 | 0.689549 | 0.248203 |
| fold_macro_std | xgboost_fixed_v1 | 344.156 | 0.00992449 | 0.0103285 | 0.0136129 | 0.0167831 | 0.00212116 | 0.00105345 |

## Offset Stability
| subset_name | sample_count | accuracy | balanced_accuracy | roc_auc | average_precision | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFFSET_00 | 43728 | 0.549099 | 0.548073 | 0.565417 | 0.564156 | 0.68647 | 0.246679 |
| OFFSET_05 | 43727 | 0.540193 | 0.539348 | 0.556242 | 0.554936 | 0.688208 | 0.247542 |
| OFFSET_10 | 43727 | 0.541176 | 0.540161 | 0.555525 | 0.554357 | 0.68837 | 0.247619 |
| OFFSET_15 | 43727 | 0.538523 | 0.537539 | 0.553423 | 0.554207 | 0.688624 | 0.247748 |
| OFFSET_20 | 43727 | 0.537059 | 0.536007 | 0.548555 | 0.548446 | 0.689762 | 0.248307 |
| OFFSET_25 | 43726 | 0.53387 | 0.533009 | 0.54393 | 0.542242 | 0.69069 | 0.248764 |
| OFFSET_30 | 43733 | 0.534791 | 0.533496 | 0.547084 | 0.548181 | 0.689941 | 0.248393 |
| OFFSET_35 | 43732 | 0.53167 | 0.5308 | 0.541286 | 0.541808 | 0.691128 | 0.248983 |
| OFFSET_40 | 43730 | 0.532015 | 0.531041 | 0.541283 | 0.54408 | 0.690939 | 0.248892 |
| OFFSET_45 | 43731 | 0.530608 | 0.529819 | 0.541533 | 0.544978 | 0.690989 | 0.248919 |
| OFFSET_50 | 43730 | 0.533387 | 0.5323 | 0.545211 | 0.548263 | 0.690275 | 0.248563 |
| OFFSET_55 | 43730 | 0.537709 | 0.536621 | 0.551419 | 0.552966 | 0.689186 | 0.248021 |

## Boundary Diagnostics
| subset_name | sample_count | accuracy | balanced_accuracy | roc_auc | average_precision | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_MARGINS | 524748 | 0.536675 | 0.535684 | 0.549265 | 0.54982 | 0.689549 | 0.248203 |
| ABS_RETURN_GE_1BPS | 511560 | 0.53772 | 0.536698 | 0.550746 | 0.551335 | 0.689255 | 0.248058 |
| ABS_RETURN_GE_2_5BPS | 491756 | 0.538873 | 0.537785 | 0.552307 | 0.553152 | 0.688948 | 0.247905 |
| ABS_RETURN_GE_5BPS | 459576 | 0.540642 | 0.539395 | 0.554598 | 0.556199 | 0.688478 | 0.247673 |
| ABS_RETURN_GE_10BPS | 399257 | 0.542317 | 0.540791 | 0.556921 | 0.559812 | 0.687976 | 0.247425 |

## Model Comparison
- Positive ROC-AUC/AP/Accuracy/MCC deltas favor XGBoost. Negative Log Loss/Brier deltas favor XGBoost.
| fold_name | subset_name | baseline_model_name | delta_roc_auc | delta_average_precision | delta_log_loss | delta_brier | delta_accuracy | delta_mcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2020 | DENSE | logistic_regression_l2 | 0.00706092 | 0.00784726 | -0.00126421 | -0.000621322 | 0.00517853 | 0.010527 |
| fold_2020 | NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 0.00920267 | 0.00777621 | -0.00232897 | -0.00113977 | 0.0103199 | 0.0207739 |
| fold_2021 | DENSE | logistic_regression_l2 | 0.00447754 | 0.000286542 | -0.00292357 | -0.00143242 | 0.00374944 | 0.00719702 |
| fold_2021 | NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 0.00696924 | 0.000748163 | -0.00237346 | -0.00116829 | 0.0021809 | 0.00457943 |
| fold_2022 | DENSE | logistic_regression_l2 | 0.0105295 | 0.0109094 | -0.00203157 | -0.0010042 | 0.00854359 | 0.0173932 |
| fold_2022 | NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 0.0121872 | 0.0147892 | -0.00221947 | -0.00109715 | 0.0125585 | 0.0255874 |
| fold_2023 | DENSE | logistic_regression_l2 | 0.0113506 | 0.00784149 | -0.00171754 | -0.000860449 | 0.0105206 | 0.0218648 |
| fold_2023 | NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 0.0141992 | 0.00918589 | -0.00252349 | -0.0012591 | 0.0178225 | 0.0379916 |
| fold_2024 | DENSE | logistic_regression_l2 | 0.005101 | 0.00499644 | -0.00115937 | -0.000567541 | 0.00346313 | 0.00736158 |
| fold_2024 | NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 0.00449957 | 0.00431277 | -0.000255751 | -0.000129046 | 0.00478197 | 0.0102141 |
| POOLED | DENSE | logistic_regression_l2 | 0.00807985 | 0.00804767 | -0.0018182 | -0.000896673 | 0.00629254 | 0.012848 |
| POOLED | NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 0.00958614 | 0.00937146 | -0.00193842 | -0.00095779 | 0.00953622 | 0.0193933 |
| fold_2020 | DENSE | momentum_60m_baseline | 0.0276394 | 0.0391675 | -0.00317625 | -0.00157521 | 0.0827131 | 0.1648 |
| fold_2020 | NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 0.0374425 | 0.0477781 | -0.00627346 | -0.00309168 | 0.105951 | 0.211482 |
| fold_2021 | DENSE | momentum_60m_baseline | 0.0194183 | 0.0183584 | -0.00261545 | -0.00130381 | 0.0342519 | 0.0668314 |
| fold_2021 | NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 0.0333431 | 0.0313538 | -0.00462607 | -0.00230253 | 0.0449954 | 0.0897416 |
| fold_2022 | DENSE | momentum_60m_baseline | 0.0216673 | 0.0272529 | -0.0025592 | -0.00127296 | 0.079737 | 0.160558 |
| fold_2022 | NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 0.0312827 | 0.0378588 | -0.00424929 | -0.00211101 | 0.0940747 | 0.190063 |
| fold_2023 | DENSE | momentum_60m_baseline | 0.0203738 | 0.0213053 | -0.00137945 | -0.000702617 | 0.0730539 | 0.145496 |
| fold_2023 | NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 0.0293402 | 0.0317854 | -0.00402905 | -0.00200381 | 0.099623 | 0.198376 |
| fold_2024 | DENSE | momentum_60m_baseline | 0.0147869 | 0.0210443 | -0.00122615 | -0.00061065 | 0.055467 | 0.109437 |
| fold_2024 | NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 0.0220113 | 0.028234 | -0.00187876 | -0.000936071 | 0.0753729 | 0.149464 |
| POOLED | DENSE | momentum_60m_baseline | 0.0221799 | 0.021333 | -0.0021896 | -0.00109221 | 0.065056 | 0.128941 |
| POOLED | NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 0.0318463 | 0.0304376 | -0.00420783 | -0.0020873 | 0.0840194 | 0.166968 |
| fold_2020 | DENSE | prior_baseline | 0.0645554 | 0.0589634 | -0.00571101 | -0.00283993 | 0.0287781 | 0.0909684 |
| fold_2020 | NONOVERLAP_OFFSET_00 | prior_baseline | 0.0845442 | 0.0735115 | -0.0107053 | -0.00530132 | 0.0412797 | 0.117281 |
| fold_2021 | DENSE | prior_baseline | 0.0307575 | 0.0241556 | -0.000765029 | -0.00038613 | 0.0162508 | 0.0441528 |
| fold_2021 | NONOVERLAP_OFFSET_00 | prior_baseline | 0.0474938 | 0.038629 | -0.00324656 | -0.00162018 | 0.0269743 | 0.0614402 |
| fold_2022 | DENSE | prior_baseline | 0.0586301 | 0.0471009 | -0.00523879 | -0.00261249 | 0.043441 | 0.0866328 |
| fold_2022 | NONOVERLAP_OFFSET_00 | prior_baseline | 0.0715399 | 0.0596069 | -0.0073872 | -0.00367938 | 0.0561708 | 0.109548 |
| fold_2023 | DENSE | prior_baseline | 0.0519984 | 0.0381145 | -0.00333899 | -0.00168171 | 0.034399 | 0.0822471 |
| fold_2023 | NONOVERLAP_OFFSET_00 | prior_baseline | 0.0709195 | 0.0542881 | -0.00742565 | -0.00369915 | 0.0450131 | 0.115217 |
| fold_2024 | DENSE | prior_baseline | 0.0409104 | 0.0347826 | -0.00242057 | -0.00120758 | 0.0191373 | 0.0571902 |
| fold_2024 | NONOVERLAP_OFFSET_00 | prior_baseline | 0.0549158 | 0.0457596 | -0.00403269 | -0.00201164 | 0.0311966 | 0.0836554 |
| POOLED | DENSE | prior_baseline | 0.0487603 | 0.0412263 | -0.00349534 | -0.0017458 | 0.0284079 | 0.0718863 |
| POOLED | NONOVERLAP_OFFSET_00 | prior_baseline | 0.0670905 | 0.05635 | -0.00655793 | -0.00326158 | 0.0401345 | 0.0967862 |

## Probability Diagnostics
| model_name | subset_name | ece | mce |
| --- | --- | --- | --- |
| xgboost_fixed_v1 | DENSE | 0.00770627 | 0.301686 |
| xgboost_fixed_v1 | NONOVERLAP_OFFSET_00 | 0.00440408 | 0.219883 |

## Feature Importance Stability
- Tree feature importance is not causal; correlated features split importance. No feature was removed in this stage.
| feature_name | mean_gain | std_gain | mean_normalized_gain | mean_rank | median_rank | used_fold_count | top10_fold_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ema_distance_120 | 426.79 | 177.722 | 0.423474 | 1 | 1 | 5 | 5 |
| ema_spread_20_60 | 415.015 | 285.042 | 0.100629 | 2.4 | 2 | 5 | 5 |
| log_return_120m | 88.8607 | 34.356 | 0.0920215 | 2.6 | 3 | 5 | 5 |
| time_sin | 53.5831 | 16.4525 | 0.0498361 | 4.2 | 4 | 5 | 5 |
| normalized_slope_60m | 73.8435 | 8.06654 | 0.0301299 | 9.2 | 8 | 5 | 4 |
| time_cos | 41.3537 | 10.8623 | 0.0291228 | 6.8 | 6 | 5 | 5 |
| buy_pressure_std_15m | 42.0008 | 9.28899 | 0.0273353 | 9 | 7 | 5 | 3 |
| range_position_120m | 75.7413 | 23.6765 | 0.0208343 | 8.4 | 9 | 5 | 4 |
| realized_volatility_120m | 45.0286 | 9.03527 | 0.0201516 | 9.6 | 8 | 5 | 3 |
| momentum_acceleration_15_60 | 58.9508 | 19.8157 | 0.0167159 | 10.8 | 10 | 5 | 3 |
| realized_volatility_60m | 42.2601 | 6.84868 | 0.0161241 | 12.2 | 12 | 5 | 1 |
| ema_distance_60 | 129.523 | 70.1839 | 0.0159967 | 13.8 | 10 | 5 | 3 |
| ema_distance_20 | 59.8231 | 14.2024 | 0.0125285 | 15.2 | 15 | 5 | 1 |
| ema_spread_5_20 | 62.3043 | 27.7843 | 0.0119798 | 14.8 | 16 | 5 | 1 |
| efficiency_ratio_60m | 36.9192 | 9.22847 | 0.0109916 | 16.4 | 14 | 5 | 0 |
| volume_ratio_15_60 | 35.9212 | 9.91 | 0.0102532 | 16.4 | 15 | 5 | 0 |
| weekday_sin | 34.4394 | 13.1449 | 0.00934654 | 17 | 19 | 5 | 1 |
| normalized_atr_60 | 32.9784 | 6.33037 | 0.00906294 | 16 | 17 | 5 | 0 |
| upper_wick_mean_15m | 26.8287 | 16.0572 | 0.0087414 | 23.6 | 18 | 4 | 0 |
| taker_buy_quote_ratio_60m | 31.4346 | 14.4427 | 0.00758169 | 20.8 | 18 | 5 | 0 |

## FINAL_TEST Audit
| field | value |
| --- | --- |
| final_test_sample_count | 159541 |
| final_test_min_decision_time | 1735689600000 |
| final_test_max_decision_time | 1783551600000 |
| final_test_min_decision_time_utc | 2025-01-01T00:00:00+00:00 |
| final_test_max_decision_time_utc | 2026-07-08T23:00:00+00:00 |
| final_test_prediction_count | 0 |
| final_test_metric_count | 0 |
| final_test_used_for_fit | False |
| final_test_used_for_early_stopping | False |
| final_test_used_for_selection | False |
| final_test_used_for_feature_importance | False |

## Engineering Gates
| gate | value |
| --- | --- |
| all_outer_fold_integrity_checks_passed | True |
| all_inner_split_integrity_checks_passed | True |
| no_outer_validation_used_for_early_stopping | True |
| no_final_test_predictions | True |
| no_final_test_metrics | True |
| feature_manifest_match | True |
| all_features_finite | True |
| all_probabilities_finite | True |
| all_prediction_counts_match | True |
| all_models_serialized | True |
| all_models_reload_verified | True |
| all_best_iterations_valid | True |
| xgboost_parameters_verified | True |
| final_test_prediction_count | 0 |
| final_test_metric_count | 0 |
| final_test_used_for_fit | False |
| final_test_used_for_early_stopping | False |
| final_test_used_for_selection | False |
| final_test_used_for_feature_importance | False |
| stage7_engineering_gate_passed | True |

## Development Recommendation
| field | value |
| --- | --- |
| xgb_beats_logistic_dense_auc_fold_count | 5 |
| xgb_beats_logistic_nonoverlap_auc_fold_count | 5 |
| xgb_beats_logistic_dense_logloss_fold_count | 5 |
| xgb_beats_logistic_nonoverlap_logloss_fold_count | 5 |
| xgb_pooled_dense_auc_delta | 0.00807985 |
| xgb_pooled_nonoverlap_auc_delta | 0.00958614 |
| xgb_pooled_dense_logloss_delta | -0.0018182 |
| xgb_pooled_nonoverlap_logloss_delta | -0.00193842 |
| any_fold_auc_below_half | False |
| development_recommendation | PROCEED_TO_LIMITED_TUNING |

## Runtime
| metric | value |
| --- | --- |
| elapsed_seconds | 80.1543 |
| python_tracemalloc_peak_bytes | 1194944420 |
| process_rss_bytes | 1083756544 |

