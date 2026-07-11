# Stage 8 Limited XGBoost Tuning Report

## Scope
- Compared only the six predeclared XGBoost candidates.
- No Optuna, GridSearchCV, RandomizedSearchCV, Bayesian optimization, candidate mutation, probability calibration, threshold optimization, feature selection, SHAP, resampling, class weighting, FINAL_TEST prediction, or final deployment model training was performed.

## Candidate Set
| model_name | learning_rate | max_depth | min_child_weight | gamma | subsample | colsample_bytree | reg_alpha | reg_lambda | max_estimators | early_stopping_rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_fixed_v1_reference | 0.03 | 4 | 20 | 0 | 0.8 | 0.8 | 0.1 | 10 | 3000 | 100 |
| xgb_depth3_v1 | 0.03 | 3 | 20 | 0 | 0.8 | 0.8 | 0.1 | 10 | 3000 | 100 |
| xgb_depth3_regularized_v1 | 0.03 | 3 | 50 | 0.05 | 0.8 | 0.8 | 0.5 | 20 | 3000 | 100 |
| xgb_depth4_regularized_v1 | 0.03 | 4 | 50 | 0.05 | 0.8 | 0.8 | 0.5 | 20 | 3000 | 100 |
| xgb_depth5_regularized_v1 | 0.03 | 5 | 50 | 0.1 | 0.75 | 0.75 | 0.5 | 20 | 3000 | 100 |
| xgb_low_learning_rate_v1 | 0.015 | 4 | 20 | 0 | 0.8 | 0.8 | 0.1 | 10 | 5000 | 150 |

## Candidate Hash
| field | value |
| --- | --- |
| candidate_definitions_sha256 | ac080a5f9fb51e91e3fc5a9bc303695fc90d5271af29ac3c00734b136487a79e |

## Inputs and Hashes
| item | value |
| --- | --- |
| dataset_sha256 | 4099f37af884ce221710fd84cf45fcbc06ca786dbb3a2b2b7eb1d4ec1a3187fa |
| split_sha256 | 6f99c99a92e8687e63156a80df3f995b44cea12c53aff9ae18efe11037b4070b |
| stage6_prediction_sha256 | 4f4f81933e099a9b03705cff006fe30facdd1c2264dde3456f0fcab4991cf235 |
| stage7_prediction_sha256 | abd0737f3da7850f24513e383ce196cbd15651c44abbcb125ba2e85cc25a8052 |
| dataset_manifest_sha256 | debff85c37643883005336c4c699efe4557dadd3076a801c6f0dfaa12eda1907 |
| feature_manifest_sha256 | 0edb0cfb0df991a63da7a5462bf6735db9eea13b41cd9ab5ccac8042c435507c |
| fold_manifest_sha256 | 5ddcb7e9d88f69acfd9f63197aaf6605f5fa09242e33e80eab352b64d23e923f |
| stage7_model_manifest_sha256 | 6719649182b82edd759cab2719430b90ee37ce018c7c501d8272e3c6fa429a9a |
| config_sha256 | 868655495f074928225681fec3e4127c01fe01f0d50ae125d5923a4143835f8f |
| script_sha256 | eda5d21eaa69b936d4dee6b45a9a501290a1c0aafbc2552751ead2eef9948346 |
| prediction_output_path | C:\ai\BTCUSDT\data\predictions\BTCUSDT_stage8_cv_xgboost_tuning_predictions.parquet |
| prediction_sha256 | 55868b75f48836a1d101851981f0c2ee96357a2299bf3fd8504d23d8ff1dc9f6 |

## Stage7 Reference Reproduction
| field | value |
| --- | --- |
| reference_config_reproduced | True |
| max_abs_prediction_diff | 0 |
| prediction_tolerance | 1e-07 |
| metric_tolerance | 1e-10 |
| metric_max_abs_diff | 0 |
| best_n_estimators_match_stage7 | True |

## Inner Split and Best Trees
| model_name | fold_name | best_iteration | best_n_estimators | best_score | stopped_early | reached_max_estimators |
| --- | --- | --- | --- | --- | --- | --- |
| xgb_depth3_regularized_v1 | fold_2020 | 290 | 291 | 0.683363 | True | False |
| xgb_depth3_regularized_v1 | fold_2021 | 35 | 36 | 0.688211 | True | False |
| xgb_depth3_regularized_v1 | fold_2022 | 98 | 99 | 0.690266 | True | False |
| xgb_depth3_regularized_v1 | fold_2023 | 121 | 122 | 0.686187 | True | False |
| xgb_depth3_regularized_v1 | fold_2024 | 27 | 28 | 0.690865 | True | False |
| xgb_depth3_v1 | fold_2020 | 292 | 293 | 0.683304 | True | False |
| xgb_depth3_v1 | fold_2021 | 36 | 37 | 0.688217 | True | False |
| xgb_depth3_v1 | fold_2022 | 85 | 86 | 0.690241 | True | False |
| xgb_depth3_v1 | fold_2023 | 124 | 125 | 0.686222 | True | False |
| xgb_depth3_v1 | fold_2024 | 27 | 28 | 0.690863 | True | False |
| xgb_depth4_regularized_v1 | fold_2020 | 112 | 113 | 0.683118 | True | False |
| xgb_depth4_regularized_v1 | fold_2021 | 37 | 38 | 0.687925 | True | False |
| xgb_depth4_regularized_v1 | fold_2022 | 69 | 70 | 0.690385 | True | False |
| xgb_depth4_regularized_v1 | fold_2023 | 119 | 120 | 0.685741 | True | False |
| xgb_depth4_regularized_v1 | fold_2024 | 28 | 29 | 0.690696 | True | False |
| xgb_depth5_regularized_v1 | fold_2020 | 98 | 99 | 0.683072 | True | False |
| xgb_depth5_regularized_v1 | fold_2021 | 59 | 60 | 0.687653 | True | False |
| xgb_depth5_regularized_v1 | fold_2022 | 72 | 73 | 0.690449 | True | False |
| xgb_depth5_regularized_v1 | fold_2023 | 123 | 124 | 0.685731 | True | False |
| xgb_depth5_regularized_v1 | fold_2024 | 38 | 39 | 0.690545 | True | False |
| xgb_fixed_v1_reference | fold_2020 | 135 | 136 | 0.683213 | True | False |
| xgb_fixed_v1_reference | fold_2021 | 37 | 38 | 0.687931 | True | False |
| xgb_fixed_v1_reference | fold_2022 | 90 | 91 | 0.690331 | True | False |
| xgb_fixed_v1_reference | fold_2023 | 124 | 125 | 0.685877 | True | False |
| xgb_fixed_v1_reference | fold_2024 | 31 | 32 | 0.690689 | True | False |
| xgb_low_learning_rate_v1 | fold_2020 | 217 | 218 | 0.683304 | True | False |
| xgb_low_learning_rate_v1 | fold_2021 | 70 | 71 | 0.687914 | True | False |
| xgb_low_learning_rate_v1 | fold_2022 | 133 | 134 | 0.690323 | True | False |
| xgb_low_learning_rate_v1 | fold_2023 | 234 | 235 | 0.685745 | True | False |
| xgb_low_learning_rate_v1 | fold_2024 | 60 | 61 | 0.690693 | True | False |

## Candidate Summary
| model_name | pooled_dense_auc | pooled_dense_logloss | pooled_dense_brier | nonoverlap_offset00_auc | nonoverlap_offset00_logloss | offset_macro_auc | offset_auc_std | offset_macro_logloss | dense_fold_auc_mean | dense_fold_auc_std | weakest_offset | worst_year | worst_year_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_fixed_v1_reference | 0.549265 | 0.689549 | 0.248203 | 0.565417 | 0.68647 | 0.549242 | 0.00743459 | 0.689549 | 0.54937 | 0.0136129 | OFFSET_40 | fold_2021 | 0.530758 |
| xgb_depth3_v1 | 0.548197 | 0.689711 | 0.248284 | 0.564715 | 0.686637 | 0.548177 | 0.00739113 | 0.689711 | 0.54846 | 0.0138364 | OFFSET_40 | fold_2021 | 0.529558 |
| xgb_depth3_regularized_v1 | 0.548198 | 0.689699 | 0.248278 | 0.564635 | 0.686626 | 0.548177 | 0.00735674 | 0.689699 | 0.548403 | 0.0137569 | OFFSET_40 | fold_2021 | 0.52955 |
| xgb_depth4_regularized_v1 | 0.54901 | 0.689545 | 0.248202 | 0.565107 | 0.686542 | 0.548989 | 0.00735458 | 0.689545 | 0.549316 | 0.0135195 | OFFSET_40 | fold_2021 | 0.530811 |
| xgb_depth5_regularized_v1 | 0.5495 | 0.689619 | 0.248236 | 0.565609 | 0.686442 | 0.549478 | 0.00762027 | 0.689619 | 0.549947 | 0.0132332 | OFFSET_35 | fold_2021 | 0.531686 |
| xgb_low_learning_rate_v1 | 0.549188 | 0.689496 | 0.248178 | 0.56533 | 0.686512 | 0.549165 | 0.0074647 | 0.689496 | 0.5494 | 0.0135777 | OFFSET_35 | fold_2021 | 0.530822 |

## Relative to Stage7 Reference
| model_name | subset_name | delta_roc_auc | delta_average_precision | delta_log_loss | delta_brier | delta_accuracy | delta_mcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_depth3_v1 | DENSE | -0.00106783 | -0.000451228 | 0.000162911 | 8.11334e-05 | -0.000935687 | -0.00189061 |
| xgb_depth3_v1 | NONOVERLAP_OFFSET_00 | -0.000702384 | -0.000254575 | 0.000166702 | 8.28695e-05 | -0.000640322 | -0.00129206 |
| xgb_depth3_regularized_v1 | DENSE | -0.00106678 | -0.000406695 | 0.000150703 | 7.57271e-05 | -0.000981423 | -0.001974 |
| xgb_depth3_regularized_v1 | NONOVERLAP_OFFSET_00 | -0.000782102 | -0.000238548 | 0.000155181 | 7.78547e-05 | -0.000891877 | -0.00178507 |
| xgb_depth4_regularized_v1 | DENSE | -0.000254779 | -0.000540463 | -3.17103e-06 | -7.56137e-07 | 5.33589e-05 | 6.59031e-05 |
| xgb_depth4_regularized_v1 | NONOVERLAP_OFFSET_00 | -0.00031026 | -0.000623685 | 7.13776e-05 | 3.52097e-05 | 0.000571716 | 0.00111846 |
| xgb_depth5_regularized_v1 | DENSE | 0.000235549 | -0.000627921 | 7.0775e-05 | 3.36136e-05 | -0.000304908 | -0.000644364 |
| xgb_depth5_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.000191803 | -0.000944856 | -2.84517e-05 | -1.54534e-05 | -0.00034303 | -0.000712225 |
| xgb_low_learning_rate_v1 | DENSE | -7.69772e-05 | -0.000272178 | -5.26669e-05 | -2.49832e-05 | -8.38498e-05 | -0.000225906 |
| xgb_low_learning_rate_v1 | NONOVERLAP_OFFSET_00 | -8.6803e-05 | -0.000314274 | 4.19916e-05 | 2.09039e-05 | -0.000182949 | -0.0004138 |

## Relative to Logistic Regression
| model_name | subset_name | delta_roc_auc | delta_average_precision | delta_log_loss | delta_brier | delta_accuracy | delta_mcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_depth3_regularized_v1 | DENSE | 0.00701307 | 0.00764098 | -0.0016675 | -0.000820946 | 0.00531112 | 0.010874 |
| xgb_depth3_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.00880404 | 0.00913292 | -0.00178324 | -0.000879936 | 0.00864435 | 0.0176083 |
| xgb_depth3_v1 | DENSE | 0.00701202 | 0.00759644 | -0.00165529 | -0.00081554 | 0.00535686 | 0.0109574 |
| xgb_depth3_v1 | NONOVERLAP_OFFSET_00 | 0.00888376 | 0.00911689 | -0.00177172 | -0.000874921 | 0.0088959 | 0.0181013 |
| xgb_depth4_regularized_v1 | DENSE | 0.00782507 | 0.00750721 | -0.00182137 | -0.000897429 | 0.0063459 | 0.0129139 |
| xgb_depth4_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.00927588 | 0.00874778 | -0.00186704 | -0.000922581 | 0.0101079 | 0.0205118 |
| xgb_depth5_regularized_v1 | DENSE | 0.0083154 | 0.00741975 | -0.00174742 | -0.00086306 | 0.00598764 | 0.0122037 |
| xgb_depth5_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.00977795 | 0.00842661 | -0.00196687 | -0.000973244 | 0.00919319 | 0.0186811 |
| xgb_fixed_v1_reference | DENSE | 0.00807985 | 0.00804767 | -0.0018182 | -0.000896673 | 0.00629254 | 0.012848 |
| xgb_fixed_v1_reference | NONOVERLAP_OFFSET_00 | 0.00958614 | 0.00937146 | -0.00193842 | -0.00095779 | 0.00953622 | 0.0193933 |
| xgb_low_learning_rate_v1 | DENSE | 0.00800287 | 0.00777549 | -0.00187087 | -0.000921656 | 0.00620869 | 0.0126221 |
| xgb_low_learning_rate_v1 | NONOVERLAP_OFFSET_00 | 0.00949934 | 0.00905719 | -0.00189643 | -0.000936886 | 0.00935327 | 0.0189795 |

## Probability Diagnostics
| model_name | subset_name | ece | mce |
| --- | --- | --- | --- |
| xgb_depth3_regularized_v1 | DENSE | 0.00786888 | 0.332122 |
| xgb_depth3_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.00425318 | 0.21875 |
| xgb_depth3_v1 | DENSE | 0.007764 | 0.333256 |
| xgb_depth3_v1 | NONOVERLAP_OFFSET_00 | 0.00468134 | 0.155763 |
| xgb_depth4_regularized_v1 | DENSE | 0.00638836 | 0.206771 |
| xgb_depth4_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.0058477 | 0.0112557 |
| xgb_depth5_regularized_v1 | DENSE | 0.00994974 | 0.233777 |
| xgb_depth5_regularized_v1 | NONOVERLAP_OFFSET_00 | 0.00252889 | 0.00410433 |
| xgb_fixed_v1_reference | DENSE | 0.00770627 | 0.301686 |
| xgb_fixed_v1_reference | NONOVERLAP_OFFSET_00 | 0.00440408 | 0.219883 |
| xgb_low_learning_rate_v1 | DENSE | 0.00619061 | 0.144978 |
| xgb_low_learning_rate_v1 | NONOVERLAP_OFFSET_00 | 0.00523634 | 0.292748 |

## Selection Audit
| model_name | engineering_qualified | probability_quality_qualified | qualification_status | dense_bad_logloss_fold_count |
| --- | --- | --- | --- | --- |
| xgb_fixed_v1_reference | True | True | QUALIFIED | 0 |
| xgb_depth3_v1 | True | True | QUALIFIED | 0 |
| xgb_depth3_regularized_v1 | True | True | QUALIFIED | 0 |
| xgb_depth4_regularized_v1 | True | True | QUALIFIED | 0 |
| xgb_depth5_regularized_v1 | True | True | QUALIFIED | 0 |
| xgb_low_learning_rate_v1 | True | True | QUALIFIED | 0 |

| rank | model_name | offset_macro_auc | offset_macro_logloss | pooled_dense_logloss | offset_auc_std | dense_fold_auc_min | max_depth | median_best_n_estimators | within_auc_tie_tolerance_of_top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | xgb_low_learning_rate_v1 | 0.549165 | 0.689496 | 0.689496 | 0.0074647 | 0.530822 | 4 | 134 | True |
| 2 | xgb_depth4_regularized_v1 | 0.548989 | 0.689545 | 0.689545 | 0.00735458 | 0.530811 | 4 | 70 | True |
| 3 | xgb_fixed_v1_reference | 0.549242 | 0.689549 | 0.689549 | 0.00743459 | 0.530758 | 4 | 91 | True |
| 4 | xgb_depth5_regularized_v1 | 0.549478 | 0.689619 | 0.689619 | 0.00762027 | 0.531686 | 5 | 73 | True |
| 5 | xgb_depth3_regularized_v1 | 0.548177 | 0.689699 | 0.689699 | 0.00735674 | 0.52955 | 3 | 99 | False |
| 6 | xgb_depth3_v1 | 0.548177 | 0.689711 | 0.689711 | 0.00739113 | 0.529558 | 3 | 86 | False |

| field | value |
| --- | --- |
| best_ranked_candidate | xgb_low_learning_rate_v1 |
| condition_a_met | False |
| condition_b_met | False |
| offset_macro_auc_delta | -7.72361e-05 |
| offset_macro_logloss_delta | -5.26653e-05 |
| dense_auc_fold_win_count | 3 |
| dense_logloss_fold_win_count | 3 |

| field | value |
| --- | --- |
| selected_development_config | xgb_fixed_v1_reference |
| development_recommendation | KEEP_STAGE7_REFERENCE |
| improvement_not_material | True |
| selection_reason | Top candidate did not meet minimum material improvement condition. |

## Feature Importance
- Tree feature importance is diagnostic only, not causal. Correlated features split importance. No feature was removed or selected from importance.
| model_name | feature_name | mean_gain | std_gain | mean_normalized_gain | mean_rank | median_rank | used_fold_count | top10_fold_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_depth3_regularized_v1 | ema_distance_120 | 526.571 | 236.319 | 0.475216 | 1 | 1 | 5 | 5 |
| xgb_depth3_regularized_v1 | ema_spread_20_60 | 743.655 | 678.526 | 0.108909 | 2.4 | 2 | 5 | 5 |
| xgb_depth3_regularized_v1 | log_return_120m | 158.353 | 87.5827 | 0.104399 | 2.6 | 3 | 5 | 5 |
| xgb_depth3_regularized_v1 | time_sin | 68.1772 | 24.4221 | 0.0493424 | 4.2 | 4 | 5 | 5 |
| xgb_depth3_regularized_v1 | normalized_slope_60m | 138.511 | 65.1472 | 0.0331943 | 7.8 | 6 | 5 | 4 |
| xgb_depth3_regularized_v1 | range_position_120m | 91.5662 | 30.2353 | 0.0228515 | 7 | 6 | 5 | 5 |
| xgb_depth3_regularized_v1 | buy_pressure_std_15m | 54.7353 | 16.2837 | 0.021408 | 10.6 | 13 | 5 | 2 |
| xgb_depth3_regularized_v1 | realized_volatility_120m | 61.0799 | 20.8465 | 0.0173046 | 9 | 7 | 5 | 3 |
| xgb_depth3_regularized_v1 | momentum_acceleration_15_60 | 72.3268 | 27.1373 | 0.0134748 | 10 | 11 | 5 | 2 |
| xgb_depth3_regularized_v1 | ema_distance_60 | 289.047 | 251.784 | 0.013026 | 13.4 | 10 | 5 | 3 |
| xgb_depth3_regularized_v1 | time_cos | 54.3441 | 18.2946 | 0.0129683 | 12.6 | 10 | 5 | 3 |
| xgb_depth3_regularized_v1 | ema_spread_5_20 | 99.462 | 62.2773 | 0.0125062 | 12.2 | 11 | 5 | 2 |
| xgb_depth3_regularized_v1 | ema_distance_20 | 74.0047 | 23.8376 | 0.0110538 | 14.2 | 13 | 5 | 2 |
| xgb_depth3_regularized_v1 | realized_volatility_60m | 61.238 | 15.3534 | 0.0106447 | 13.4 | 12 | 5 | 1 |
| xgb_depth3_regularized_v1 | efficiency_ratio_60m | 34.1763 | 22.6572 | 0.00828245 | 18.8 | 15 | 4 | 0 |
| xgb_depth3_regularized_v1 | log_return_60m | 379.049 | 589.184 | 0.00776618 | 19.6 | 21 | 5 | 1 |
| xgb_depth3_regularized_v1 | upper_wick_mean_15m | 33.1402 | 21.0862 | 0.00726248 | 22.8 | 17 | 4 | 1 |
| xgb_depth3_regularized_v1 | volume_ratio_15_60 | 39.9252 | 10.1848 | 0.00689621 | 18.2 | 16 | 5 | 0 |
| xgb_depth3_regularized_v1 | log_return_10m | 69.6043 | 29.6057 | 0.00624373 | 18.2 | 18 | 5 | 1 |
| xgb_depth3_regularized_v1 | weekday_sin | 41.5925 | 19.5875 | 0.00591405 | 19.2 | 22 | 5 | 0 |

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
| final_test_feature_matrix_created | False |

## Engineering Gates
| gate | value |
| --- | --- |
| candidate_set_frozen_before_run | True |
| candidate_count_is_six | True |
| reference_config_reproduced | True |
| all_outer_fold_checks_passed | True |
| all_inner_split_checks_passed | True |
| no_outer_validation_used_for_early_stopping | True |
| no_final_test_predictions | True |
| no_final_test_metrics | True |
| no_final_test_feature_matrix | True |
| feature_manifest_match | True |
| all_features_finite | True |
| all_probabilities_finite | True |
| all_prediction_counts_match | True |
| all_models_serialized | True |
| all_models_reload_verified | True |
| all_best_iterations_valid | True |
| xgboost_parameters_verified | True |
| selection_rule_reproducible | True |
| final_test_prediction_count | 0 |
| final_test_metric_count | 0 |
| final_test_used_for_fit | False |
| final_test_used_for_early_stopping | False |
| final_test_used_for_selection | False |
| final_test_used_for_feature_importance | False |
| final_test_feature_matrix_created | False |
| stage8_engineering_gate_passed | True |

## Runtime
| metric | value |
| --- | --- |
| elapsed_seconds | 460.451 |
| python_tracemalloc_peak_bytes | 1345620235 |
| process_rss_bytes | 1617653760 |

