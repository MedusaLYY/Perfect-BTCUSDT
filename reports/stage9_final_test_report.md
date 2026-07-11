# Stage 9 Final Test Report

## Required Statements
- FINAL_TEST base XGBoost predict_proba call count: 1.
- Parameters, tree count, calibrator, threshold, and metrics were frozen before test access.
- No model parameter, calibration method, threshold, or feature was changed after FINAL_TEST.
- No threshold optimization was performed.
- No trading backtest was performed.
- The final conclusion must not be interpreted as guaranteed profitability.

## Main Metrics
| model_name | subset_name | sample_count | accuracy | balanced_accuracy | roc_auc | average_precision | mcc | log_loss | brier_score | ece_equal_width | ece_equal_frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_final | DENSE | 159541 | 0.517027 | 0.516714 | 0.527632 | 0.526929 | 0.0339235 | 0.693434 | 0.250135 | 0.0259297 | 0.0258684 |
| logistic_regression_final | NONOVERLAP_OFFSET_00 | 13296 | 0.522488 | 0.522449 | 0.537701 | 0.537723 | 0.0455897 | 0.6917 | 0.249282 | 0.0207227 | 0.0200106 |
| momentum_60m_baseline_final | DENSE | 159541 | 0.487787 | 0.48778 | 0.51222 | 0.508099 | -0.0244403 | 0.69394 | 0.250394 | 0.021825 | 0.0221025 |
| momentum_60m_baseline_final | NONOVERLAP_OFFSET_00 | 13296 | 0.484356 | 0.484356 | 0.515644 | 0.508292 | -0.031288 | 0.693542 | 0.250195 | 0.0184248 | 0.0197868 |
| prior_baseline_final | DENSE | 159541 | 0.50184 | 0.5 | 0.5 | 0.50184 | 0 | 0.693261 | 0.250057 | 0.00776419 | 0.0108049 |
| prior_baseline_final | NONOVERLAP_OFFSET_00 | 13296 | 0.500226 | 0.5 | 0.5 | 0.500226 | 0 | 0.693323 | 0.250088 | 0.00937822 | 0.00982397 |
| xgboost_final_calibrated | DENSE | 159541 | 0.518469 | 0.518257 | 0.529124 | 0.528139 | 0.0367585 | 0.692542 | 0.249697 | 0.0183341 | 0.0187374 |
| xgboost_final_calibrated | NONOVERLAP_OFFSET_00 | 13296 | 0.527076 | 0.52705 | 0.539635 | 0.538587 | 0.0544492 | 0.691108 | 0.248983 | 0.0102807 | 0.0173466 |
| xgboost_final_raw | DENSE | 159541 | 0.518469 | 0.518257 | 0.529124 | 0.528139 | 0.0367585 | 0.692542 | 0.249697 | 0.0183341 | 0.0187374 |
| xgboost_final_raw | NONOVERLAP_OFFSET_00 | 13296 | 0.527076 | 0.52705 | 0.539635 | 0.538587 | 0.0544492 | 0.691108 | 0.248983 | 0.0102807 | 0.0173466 |

## Offset Stability
| subset_name | sample_count | roc_auc | average_precision | log_loss | brier_score | mcc |
| --- | --- | --- | --- | --- | --- | --- |
| OFFSET_00 | 13296 | 0.539635 | 0.538587 | 0.691108 | 0.248983 | 0.0544492 |
| OFFSET_05 | 13295 | 0.52509 | 0.518872 | 0.69321 | 0.250028 | 0.0359407 |
| OFFSET_10 | 13295 | 0.526223 | 0.520072 | 0.693184 | 0.250016 | 0.0346587 |
| OFFSET_15 | 13295 | 0.528639 | 0.531386 | 0.692521 | 0.249687 | 0.0379135 |
| OFFSET_20 | 13295 | 0.529801 | 0.530375 | 0.692421 | 0.249637 | 0.0371486 |
| OFFSET_25 | 13295 | 0.528206 | 0.524668 | 0.692741 | 0.249796 | 0.0343049 |
| OFFSET_30 | 13295 | 0.52771 | 0.526565 | 0.692614 | 0.249731 | 0.0341271 |
| OFFSET_35 | 13295 | 0.52528 | 0.52677 | 0.693077 | 0.249963 | 0.0271222 |
| OFFSET_40 | 13295 | 0.528906 | 0.532935 | 0.692512 | 0.249683 | 0.0343855 |
| OFFSET_45 | 13295 | 0.528939 | 0.529352 | 0.692549 | 0.249701 | 0.0343619 |
| OFFSET_50 | 13295 | 0.529152 | 0.53067 | 0.692472 | 0.249662 | 0.0355078 |
| OFFSET_55 | 13295 | 0.531864 | 0.530247 | 0.692101 | 0.249476 | 0.0413541 |

## Boundary Return Diagnostics
| subset_name | sample_count | roc_auc | average_precision | log_loss | brier_score | mcc |
| --- | --- | --- | --- | --- | --- | --- |
| ALL_MARGINS | 159541 | 0.529124 | 0.528139 | 0.692542 | 0.249697 | 0.0367585 |
| ABS_RETURN_GE_1BPS | 154802 | 0.529973 | 0.528922 | 0.692421 | 0.249636 | 0.0382149 |
| ABS_RETURN_GE_2_5BPS | 147903 | 0.530679 | 0.529278 | 0.692335 | 0.249594 | 0.0395597 |
| ABS_RETURN_GE_5BPS | 136559 | 0.532086 | 0.530601 | 0.692134 | 0.249493 | 0.0420643 |
| ABS_RETURN_GE_10BPS | 115130 | 0.534684 | 0.533065 | 0.691775 | 0.249315 | 0.0461795 |

## Final Test vs Stage7 OOF
| subset_name | test_minus_oof_auc | test_minus_oof_logloss | test_minus_oof_brier | test_minus_oof_mcc |
| --- | --- | --- | --- | --- |
| DENSE | -0.0201406 | 0.00299391 | 0.00149424 | -0.0351279 |
| NONOVERLAP_OFFSET_00 | -0.0257815 | 0.0046377 | 0.00230428 | -0.042337 |

## Assessment
| field | value |
| --- | --- |
| final_model_assessment | READY_FOR_RESEARCH_DEPLOYMENT |

## Engineering Gates
| gate | value |
| --- | --- |
| selected_config_matches_stage8 | True |
| calibration_uses_oof_only | True |
| calibration_forward_validation_passed | True |
| calibration_frozen_before_test | True |
| final_tree_count_selected_without_test | True |
| final_model_fit_uses_development_only | True |
| final_model_serialized_before_test | True |
| final_model_reload_verified_before_test | True |
| logistic_fit_uses_development_only | True |
| protocol_lock_written_before_test | True |
| final_test_feature_matrix_created_after_lock | True |
| final_test_predict_proba_call_count_is_one | True |
| final_test_predictions_unique | True |
| no_parameter_change_after_test | True |
| feature_manifest_match | True |
| all_probabilities_finite | True |
| all_models_serialized | True |
| final_test_sample_count_match | True |
| stage9_engineering_gate_passed | True |

