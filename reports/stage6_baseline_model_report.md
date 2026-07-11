# Stage 6 Baseline Model Report

## Scope
- Trained only the three Stage 6 baselines: prior_baseline, momentum_60m_baseline, logistic_regression_l2.
- No XGBoost, LightGBM, CatBoost, neural network, probability calibration, threshold optimization, feature selection, outlier clipping, missing value imputation, class resampling, or final deployment model training was performed.
- FINAL_TEST remained sealed: no predict, predict_proba, decision_function, metrics, model selection, or threshold decisions used FINAL_TEST rows.

## Inputs and Hashes
| item | value |
| --- | --- |
| dataset_path | data/model/BTCUSDT_60m_direction_dataset_v1.parquet |
| split_path | data/splits/BTCUSDT_60m_split_assignments_v1.parquet |
| dataset_sha256 | 4099f37af884ce221710fd84cf45fcbc06ca786dbb3a2b2b7eb1d4ec1a3187fa |
| split_sha256 | 6f99c99a92e8687e63156a80df3f995b44cea12c53aff9ae18efe11037b4070b |
| config_sha256 | e90b615b7037db1fc705f9d0838dfc0c2179dc8681c0453e4f28488c87717f2b |
| script_sha256 | 8394dc3115e306eda23d54978fe6bfc11aa8d0706bc01ce90e464093b516bbfa |

## Feature Manifest Check
| check | value |
| --- | --- |
| feature_count | 63 |
| feature_manifest_match | True |
| feature_list_sha256 | c2c2c84b40520359b3d437601a72ec6a4a26f26522794c7b4d9c900bd16a4491 |

## Fold Counts and Time Boundaries
| fold_name | train_rows | validation_rows | train_max_settlement_utc | validation_min_decision_utc | overlap_count |
| --- | --- | --- | --- | --- | --- |
| fold_2020 | 234419 | 104663 | 2019-12-31T23:55:00+00:00 | 2020-01-01T00:00:00+00:00 | 0 |
| fold_2021 | 339094 | 104549 | 2020-12-31T23:55:00+00:00 | 2021-01-01T00:00:00+00:00 | 0 |
| fold_2022 | 443655 | 105108 | 2021-12-31T23:55:00+00:00 | 2022-01-01T00:00:00+00:00 | 0 |
| fold_2023 | 548775 | 105032 | 2022-12-31T23:55:00+00:00 | 2023-01-01T00:00:00+00:00 | 0 |
| fold_2024 | 653819 | 105396 | 2023-12-31T23:55:00+00:00 | 2024-01-01T00:00:00+00:00 | 0 |

## Model Definitions
| model_name | definition |
| --- | --- |
| prior_baseline | TRAIN label_up_60m mean; hard class is 1 when train prior >= 0.5. |
| momentum_60m_baseline | log_return_60m > 0 predicts 1; probabilities are TRAIN conditional rates with Laplace alpha=1.0. |
| logistic_regression_l2 | StandardScaler fitted on TRAIN only, then LogisticRegression L2 with fixed C=1.0/lbfgs/max_iter=2000/tol=1e-6 and threshold 0.5. |

## StandardScaler Audit
| fold_name | train_only_fit | scaler_n_samples_seen | mean_diff | var_diff |
| --- | --- | --- | --- | --- |
| fold_2020 | True | 234419 | 0 | 0 |
| fold_2021 | True | 339094 | 0 | 0 |
| fold_2022 | True | 443655 | 0 | 0 |
| fold_2023 | True | 548775 | 0 | 0 |
| fold_2024 | True | 653819 | 0 | 0 |

## Dense Metrics by Fold
| fold_name | model_name | sample_count | positive_ratio | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2020 | logistic_regression_l2 | 104663 | 0.51769 | 0.54129 | 0.540156 | 0.557494 | 0.568806 | 0.0804414 | 0.688126 | 0.247494 |
| fold_2020 | momentum_60m_baseline | 104663 | 0.51769 | 0.463755 | 0.463084 | 0.536916 | 0.537486 | -0.0738318 | 0.690038 | 0.248448 |
| fold_2020 | prior_baseline | 104663 | 0.51769 | 0.51769 | 0.5 | 0.5 | 0.51769 | 0 | 0.692573 | 0.249713 |
| fold_2021 | logistic_regression_l2 | 104549 | 0.506758 | 0.519259 | 0.518286 | 0.52628 | 0.530627 | 0.0369558 | 0.695324 | 0.251056 |
| fold_2021 | momentum_60m_baseline | 104549 | 0.506758 | 0.488756 | 0.488661 | 0.511339 | 0.512555 | -0.0226786 | 0.695016 | 0.250927 |
| fold_2021 | prior_baseline | 104549 | 0.506758 | 0.506758 | 0.5 | 0.5 | 0.506758 | 0 | 0.693166 | 0.250009 |
| fold_2022 | logistic_regression_l2 | 105108 | 0.499334 | 0.534231 | 0.534319 | 0.548101 | 0.535526 | 0.0692396 | 0.690281 | 0.248562 |
| fold_2022 | momentum_60m_baseline | 105108 | 0.499334 | 0.463038 | 0.463037 | 0.536963 | 0.519182 | -0.0739255 | 0.690809 | 0.248831 |
| fold_2022 | prior_baseline | 105108 | 0.499334 | 0.499334 | 0.5 | 0.5 | 0.499334 | 0 | 0.693488 | 0.250171 |
| fold_2023 | logistic_regression_l2 | 105032 | 0.507141 | 0.531019 | 0.52946 | 0.540648 | 0.537414 | 0.0603822 | 0.691439 | 0.249135 |
| fold_2023 | momentum_60m_baseline | 105032 | 0.507141 | 0.468486 | 0.468375 | 0.531625 | 0.52395 | -0.0632493 | 0.691101 | 0.248978 |
| fold_2023 | prior_baseline | 105032 | 0.507141 | 0.507141 | 0.5 | 0.5 | 0.507141 | 0 | 0.69306 | 0.249957 |
| fold_2024 | logistic_regression_l2 | 105396 | 0.510437 | 0.526111 | 0.524681 | 0.535809 | 0.540223 | 0.0498286 | 0.69167 | 0.249252 |
| fold_2024 | momentum_60m_baseline | 105396 | 0.510437 | 0.474107 | 0.473877 | 0.526123 | 0.524175 | -0.052247 | 0.691737 | 0.249295 |
| fold_2024 | prior_baseline | 105396 | 0.510437 | 0.510437 | 0.5 | 0.5 | 0.510437 | 0 | 0.692931 | 0.249892 |

## Non-overlap Metrics by Fold
| fold_name | model_name | sample_count | positive_ratio | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2020 | logistic_regression_l2 | 8721 | 0.518289 | 0.549249 | 0.548196 | 0.575342 | 0.584024 | 0.0965072 | 0.684166 | 0.245536 |
| fold_2020 | momentum_60m_baseline | 8721 | 0.518289 | 0.453618 | 0.452898 | 0.547102 | 0.544023 | -0.0942011 | 0.688111 | 0.247488 |
| fold_2020 | prior_baseline | 8721 | 0.518289 | 0.518289 | 0.5 | 0.5 | 0.518289 | 0 | 0.692542 | 0.249698 |
| fold_2021 | logistic_regression_l2 | 8712 | 0.503903 | 0.528696 | 0.528158 | 0.540525 | 0.541784 | 0.0568607 | 0.692454 | 0.249638 |
| fold_2021 | momentum_60m_baseline | 8712 | 0.503903 | 0.485882 | 0.485849 | 0.514151 | 0.511178 | -0.0283014 | 0.694707 | 0.250772 |
| fold_2021 | prior_baseline | 8712 | 0.503903 | 0.503903 | 0.5 | 0.5 | 0.503903 | 0 | 0.693328 | 0.25009 |
| fold_2022 | logistic_regression_l2 | 8759 | 0.49766 | 0.541272 | 0.541591 | 0.559353 | 0.542477 | 0.0839609 | 0.688404 | 0.24763 |
| fold_2022 | momentum_60m_baseline | 8759 | 0.49766 | 0.459756 | 0.459743 | 0.540257 | 0.519408 | -0.0805146 | 0.690434 | 0.248644 |
| fold_2022 | prior_baseline | 8759 | 0.49766 | 0.49766 | 0.5 | 0.5 | 0.49766 | 0 | 0.693571 | 0.250212 |
| fold_2023 | logistic_regression_l2 | 8753 | 0.513424 | 0.540615 | 0.537665 | 0.55672 | 0.558526 | 0.0772255 | 0.687909 | 0.247392 |
| fold_2023 | momentum_60m_baseline | 8753 | 0.513424 | 0.458814 | 0.458421 | 0.541579 | 0.535927 | -0.0831592 | 0.689415 | 0.248137 |
| fold_2023 | prior_baseline | 8753 | 0.513424 | 0.513424 | 0.5 | 0.5 | 0.513424 | 0 | 0.692812 | 0.249832 |
| fold_2024 | logistic_regression_l2 | 8783 | 0.511556 | 0.537971 | 0.536366 | 0.550416 | 0.553003 | 0.0734413 | 0.689112 | 0.247988 |
| fold_2024 | momentum_60m_baseline | 8783 | 0.511556 | 0.46738 | 0.467095 | 0.532905 | 0.529082 | -0.065809 | 0.690735 | 0.248795 |
| fold_2024 | prior_baseline | 8783 | 0.511556 | 0.511556 | 0.5 | 0.5 | 0.511556 | 0 | 0.692889 | 0.249871 |

## Pooled OOF Metrics
| subset_name | model_name | sample_count | positive_ratio | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE | logistic_regression_l2 | 524748 | 0.508267 | 0.530382 | 0.529232 | 0.541185 | 0.541772 | 0.0590383 | 0.691367 | 0.249099 |
| NONOVERLAP_OFFSET_00 | logistic_regression_l2 | 43728 | 0.508965 | 0.539563 | 0.538324 | 0.555831 | 0.554785 | 0.0773928 | 0.688409 | 0.247637 |
| DENSE | momentum_60m_baseline | 524748 | 0.508267 | 0.471619 | 0.471473 | 0.527085 | 0.528487 | -0.0570548 | 0.691738 | 0.249295 |
| NONOVERLAP_OFFSET_00 | momentum_60m_baseline | 43728 | 0.508965 | 0.46508 | 0.464909 | 0.533571 | 0.533718 | -0.0701822 | 0.690678 | 0.248766 |
| DENSE | prior_baseline | 524748 | 0.508267 | 0.508267 | 0.5 | 0.500505 | 0.508593 | 0 | 0.693044 | 0.249948 |
| NONOVERLAP_OFFSET_00 | prior_baseline | 43728 | 0.508965 | 0.508965 | 0.5 | 0.498326 | 0.507806 | 0 | 0.693028 | 0.249941 |

## Offset Stability
| subset_name | model_name | sample_count | accuracy | balanced_accuracy | roc_auc | average_precision | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OFFSET_00 | logistic_regression_l2 | 43728 | 0.539563 | 0.538324 | 0.555831 | 0.554785 | 0.688409 | 0.247637 |
| OFFSET_05 | logistic_regression_l2 | 43727 | 0.535321 | 0.534418 | 0.547588 | 0.543472 | 0.690169 | 0.248507 |
| OFFSET_10 | logistic_regression_l2 | 43727 | 0.534704 | 0.533609 | 0.545893 | 0.54558 | 0.690438 | 0.248641 |
| OFFSET_15 | logistic_regression_l2 | 43727 | 0.533217 | 0.532121 | 0.544609 | 0.543456 | 0.690747 | 0.248792 |
| OFFSET_20 | logistic_regression_l2 | 43727 | 0.528895 | 0.52774 | 0.539542 | 0.539692 | 0.691715 | 0.249271 |
| OFFSET_25 | logistic_regression_l2 | 43726 | 0.529571 | 0.528528 | 0.53679 | 0.536416 | 0.692305 | 0.249565 |
| OFFSET_30 | logistic_regression_l2 | 43733 | 0.530423 | 0.528899 | 0.541209 | 0.54392 | 0.691158 | 0.248998 |
| OFFSET_35 | logistic_regression_l2 | 43732 | 0.524925 | 0.523865 | 0.534637 | 0.536379 | 0.692673 | 0.249749 |
| OFFSET_40 | logistic_regression_l2 | 43730 | 0.524903 | 0.523726 | 0.534162 | 0.537436 | 0.692752 | 0.249789 |
| OFFSET_45 | logistic_regression_l2 | 43731 | 0.524365 | 0.523409 | 0.535098 | 0.53741 | 0.692671 | 0.249739 |
| OFFSET_50 | logistic_regression_l2 | 43730 | 0.527555 | 0.52628 | 0.536922 | 0.539947 | 0.692156 | 0.24948 |
| OFFSET_55 | logistic_regression_l2 | 43730 | 0.531146 | 0.529861 | 0.541726 | 0.54366 | 0.691208 | 0.249023 |
| OFFSET_00 | momentum_60m_baseline | 43728 | 0.46508 | 0.464909 | 0.533571 | 0.533718 | 0.690678 | 0.248766 |
| OFFSET_05 | momentum_60m_baseline | 43727 | 0.463832 | 0.46373 | 0.533184 | 0.530618 | 0.690653 | 0.248753 |
| OFFSET_10 | momentum_60m_baseline | 43727 | 0.462186 | 0.462042 | 0.536436 | 0.534612 | 0.690263 | 0.248559 |
| OFFSET_15 | momentum_60m_baseline | 43727 | 0.466119 | 0.465984 | 0.535222 | 0.534554 | 0.690794 | 0.248824 |
| OFFSET_20 | momentum_60m_baseline | 43727 | 0.472797 | 0.47265 | 0.527028 | 0.529079 | 0.691864 | 0.249358 |
| OFFSET_25 | momentum_60m_baseline | 43726 | 0.472259 | 0.472152 | 0.526783 | 0.526991 | 0.691882 | 0.249366 |
| OFFSET_30 | momentum_60m_baseline | 43733 | 0.468365 | 0.468109 | 0.530067 | 0.533167 | 0.691076 | 0.248965 |
| OFFSET_35 | momentum_60m_baseline | 43732 | 0.477865 | 0.477747 | 0.519813 | 0.522854 | 0.692814 | 0.249831 |
| OFFSET_40 | momentum_60m_baseline | 43730 | 0.478253 | 0.478112 | 0.519763 | 0.523795 | 0.692794 | 0.249821 |
| OFFSET_45 | momentum_60m_baseline | 43731 | 0.481238 | 0.481128 | 0.516902 | 0.519757 | 0.693355 | 0.250101 |
| OFFSET_50 | momentum_60m_baseline | 43730 | 0.476789 | 0.476619 | 0.522025 | 0.526052 | 0.692499 | 0.249674 |
| OFFSET_55 | momentum_60m_baseline | 43730 | 0.47464 | 0.474455 | 0.524247 | 0.527096 | 0.692184 | 0.249517 |
| OFFSET_00 | prior_baseline | 43728 | 0.508965 | 0.5 | 0.498326 | 0.507806 | 0.693028 | 0.249941 |
| OFFSET_05 | prior_baseline | 43727 | 0.506712 | 0.5 | 0.498409 | 0.505911 | 0.693132 | 0.249992 |
| OFFSET_10 | prior_baseline | 43727 | 0.50813 | 0.5 | 0.500373 | 0.508381 | 0.693051 | 0.249952 |
| OFFSET_15 | prior_baseline | 43727 | 0.507947 | 0.5 | 0.502587 | 0.509505 | 0.693044 | 0.249948 |
| OFFSET_20 | prior_baseline | 43727 | 0.50829 | 0.5 | 0.501617 | 0.509365 | 0.693038 | 0.249945 |
| OFFSET_25 | prior_baseline | 43726 | 0.507135 | 0.5 | 0.500194 | 0.507479 | 0.693097 | 0.249975 |
| OFFSET_30 | prior_baseline | 43733 | 0.510896 | 0.5 | 0.500627 | 0.511339 | 0.692918 | 0.249885 |
| OFFSET_35 | prior_baseline | 43732 | 0.507477 | 0.5 | 0.500197 | 0.507362 | 0.693082 | 0.249968 |
| OFFSET_40 | prior_baseline | 43730 | 0.508255 | 0.5 | 0.503528 | 0.510604 | 0.693024 | 0.249939 |
| OFFSET_45 | prior_baseline | 43731 | 0.507054 | 0.5 | 0.501094 | 0.507433 | 0.693096 | 0.249974 |
| OFFSET_50 | prior_baseline | 43730 | 0.509056 | 0.5 | 0.500354 | 0.509634 | 0.693012 | 0.249932 |
| OFFSET_55 | prior_baseline | 43730 | 0.509284 | 0.5 | 0.498748 | 0.508346 | 0.693006 | 0.249929 |

## Boundary Return Diagnostics
| subset_name | model_name | sample_count | accuracy | balanced_accuracy | roc_auc | average_precision | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_MARGINS | logistic_regression_l2 | 524748 | 0.530382 | 0.529232 | 0.541185 | 0.541772 | 0.691367 | 0.249099 |
| ABS_RETURN_GE_1BPS | logistic_regression_l2 | 511560 | 0.531144 | 0.529972 | 0.542333 | 0.542891 | 0.691134 | 0.248984 |
| ABS_RETURN_GE_2_5BPS | logistic_regression_l2 | 491756 | 0.532075 | 0.530851 | 0.54358 | 0.544348 | 0.690873 | 0.248855 |
| ABS_RETURN_GE_5BPS | logistic_regression_l2 | 459576 | 0.533701 | 0.532335 | 0.545585 | 0.547006 | 0.690437 | 0.248639 |
| ABS_RETURN_GE_10BPS | logistic_regression_l2 | 399257 | 0.535517 | 0.533942 | 0.547965 | 0.550393 | 0.689918 | 0.248383 |
| ALL_MARGINS | momentum_60m_baseline | 524748 | 0.471619 | 0.471473 | 0.527085 | 0.528487 | 0.691738 | 0.249295 |
| ABS_RETURN_GE_1BPS | momentum_60m_baseline | 511560 | 0.470783 | 0.470637 | 0.527781 | 0.529144 | 0.691606 | 0.249229 |
| ABS_RETURN_GE_2_5BPS | momentum_60m_baseline | 491756 | 0.469881 | 0.46973 | 0.52856 | 0.53009 | 0.691452 | 0.249152 |
| ABS_RETURN_GE_5BPS | momentum_60m_baseline | 459576 | 0.468641 | 0.468483 | 0.529694 | 0.531875 | 0.691214 | 0.249033 |
| ABS_RETURN_GE_10BPS | momentum_60m_baseline | 399257 | 0.46689 | 0.466742 | 0.531161 | 0.534249 | 0.690891 | 0.248872 |
| ALL_MARGINS | prior_baseline | 524748 | 0.508267 | 0.5 | 0.500505 | 0.508593 | 0.693044 | 0.249948 |
| ABS_RETURN_GE_1BPS | prior_baseline | 511560 | 0.508431 | 0.5 | 0.500662 | 0.508826 | 0.693035 | 0.249944 |
| ABS_RETURN_GE_2_5BPS | prior_baseline | 491756 | 0.508811 | 0.5 | 0.500408 | 0.509046 | 0.69302 | 0.249936 |
| ABS_RETURN_GE_5BPS | prior_baseline | 459576 | 0.509809 | 0.5 | 0.50005 | 0.509886 | 0.692977 | 0.249915 |
| ABS_RETURN_GE_10BPS | prior_baseline | 399257 | 0.511175 | 0.5 | 0.499374 | 0.51075 | 0.692918 | 0.249886 |

## Probability Diagnostics
- Calibration tables are raw probability diagnostics only. ECE = sum over bins of (bin_sample_count / total_sample_count) * abs(mean_predicted_probability - actual_up_ratio).
| model_name | subset_name | ece | mce |
| --- | --- | --- | --- |
| logistic_regression_l2 | DENSE | 0.017896 | 0.453548 |
| logistic_regression_l2 | NONOVERLAP_OFFSET_00 | 0.0093023 | 0.551135 |
| momentum_60m_baseline | DENSE | 0.0108672 | 0.0148164 |
| momentum_60m_baseline | NONOVERLAP_OFFSET_00 | 0.00432414 | 0.00742926 |
| prior_baseline | DENSE | 0.00344095 | 0.00344095 |
| prior_baseline | NONOVERLAP_OFFSET_00 | 0.00274321 | 0.00274321 |

## Logistic Regression Convergence
| fold_name | converged | n_iter | fit_elapsed_seconds | process_rss_bytes |
| --- | --- | --- | --- | --- |
| fold_2020 | True | [107] | 3.63081 | 792698880 |
| fold_2021 | True | [93] | 4.19139 | 867098624 |
| fold_2022 | True | [108] | 6.0125 | 936693760 |
| fold_2023 | True | [107] | 6.74745 | 1010966528 |
| fold_2024 | True | [109] | 7.93318 | 1081569280 |

## Logistic Coefficient Stability
- Coefficients are on standardized features. They are affected by collinearity, are not causal effects, and were not used for feature selection.
| feature_name | coefficient_mean | coefficient_std | mean_absolute_coefficient | positive_fold_count | negative_fold_count | sign_consistency_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| range_position_120m | -0.174534 | 0.0284225 | 0.174534 | 0 | 5 | 1 |
| semivariance_imbalance_60m | -0.0443162 | 0.0123222 | 0.0443162 | 0 | 5 | 1 |
| time_sin | -0.0432712 | 0.0104609 | 0.0432712 | 0 | 5 | 1 |
| normalized_atr_60 | 0.0399387 | 0.00982869 | 0.0399387 | 5 | 0 | 1 |
| ema_distance_120 | -0.0378281 | 0.0167419 | 0.0378281 | 0 | 5 | 1 |
| ema_spread_20_60 | 0.0359449 | 0.00867138 | 0.0359449 | 5 | 0 | 1 |
| upper_wick_mean_15m | -0.0330475 | 0.0119026 | 0.0330475 | 0 | 5 | 1 |
| range_position_30m | -0.0293285 | 0.01156 | 0.0293285 | 0 | 5 | 1 |
| log_return_120m | -0.0290904 | 0.00433887 | 0.0290904 | 0 | 5 | 1 |
| log_quote_volume_zscore_60m | -0.0283687 | 0.00748126 | 0.0283687 | 0 | 5 | 1 |
| realized_volatility_60m | -0.0283195 | 0.017945 | 0.0283195 | 0 | 5 | 1 |
| range_position_60m | 0.027102 | 0.00816959 | 0.027102 | 5 | 0 | 1 |
| log_return_60m | 0.026044 | 0.00447299 | 0.026044 | 5 | 0 | 1 |
| ema_distance_60 | 0.0253794 | 0.00612695 | 0.0253794 | 5 | 0 | 1 |
| realized_volatility_15m | 0.0215887 | 0.011793 | 0.0215887 | 5 | 0 | 1 |
| realized_volatility_120m | -0.0191466 | 0.0131095 | 0.0200764 | 1 | 4 | 0.8 |
| log_avg_trade_quote_size_zscore_60m | 0.0183404 | 0.00507751 | 0.0183404 | 5 | 0 | 1 |
| volatility_ratio_15_120 | -0.0167488 | 0.00287014 | 0.0167488 | 0 | 5 | 1 |
| support_distance_60m | 0.0165751 | 0.00953075 | 0.0165751 | 5 | 0 | 1 |
| buy_pressure_std_15m | 0.0164046 | 0.00444109 | 0.0164046 | 5 | 0 | 1 |

## Baseline Comparison
| question | answer |
| --- | --- |
| Logistic beats prior on dense ROC-AUC folds | 5 |
| Logistic beats momentum on dense ROC-AUC folds | 5 |
| Dense/non-overlap consistency | dense beats prior=5, non-overlap beats prior=5 |
| Offset differences | See OFFSET_00 through OFFSET_55 pooled rows; no threshold or model parameter was changed from these diagnostics. |
| Logistic ROC-AUC above 0.5 folds | 5 |
| Logistic Brier/LogLoss better than prior folds | brier=4, log_loss=4 |
| Failed years | {'fold_2021': 0.5262800002978488, 'fold_2024': 0.535809374900216} |
| Near-zero return boundary effect | See ABS_RETURN_GE_* diagnostics; boundary subsets were evaluation-only. |
| Majority-class behavior | logistic predicted_positive_ratio by fold={'fold_2020': 0.5335, 'fold_2021': 0.5723, 'fold_2022': 0.5658, 'fold_2023': 0.6096, 'fold_2024': 0.569} |
| Probability confidence | See equal-width and equal-frequency ECE/MCE tables; probabilities were not calibrated. |

## FINAL_TEST Seal Audit
| field | value |
| --- | --- |
| final_test_sample_count | 159541 |
| final_test_prediction_count | 0 |
| final_test_metric_count | 0 |
| final_test_used_for_fit | False |
| final_test_used_for_selection | False |

## Engineering Gates
| gate | value |
| --- | --- |
| all_fold_integrity_checks_passed | True |
| all_probabilities_finite | True |
| no_final_test_predictions | True |
| feature_manifest_match | True |
| preprocessing_train_only_verified | True |
| all_models_serialized | True |
| all_prediction_counts_match | True |
| all_metric_outputs_complete | True |
| logistic_all_folds_converged | True |
| final_test_prediction_count | 0 |
| final_test_metric_count | 0 |
| final_test_used_for_fit | False |
| final_test_used_for_selection | False |
| stage6_engineering_gate_passed | True |

## Runtime and Memory
| metric | value |
| --- | --- |
| elapsed_seconds | 51.3446 |
| python_tracemalloc_peak_bytes | 1593204707 |
| process_rss_bytes | 867274752 |

