# Stage 9 Calibration Selection Report

## Scope
- Compared only UNCALIBRATED, PLATT, and ISOTONIC.
- Calibration used Stage7 development OOF predictions only.
- FINAL_TEST was not read or predicted during calibration selection.

## Inputs and Hashes
| item | value |
| --- | --- |
| dataset_sha256 | 4099f37af884ce221710fd84cf45fcbc06ca786dbb3a2b2b7eb1d4ec1a3187fa |
| split_sha256 | 6f99c99a92e8687e63156a80df3f995b44cea12c53aff9ae18efe11037b4070b |
| dataset_manifest_sha256 | debff85c37643883005336c4c699efe4557dadd3076a801c6f0dfaa12eda1907 |
| feature_manifest_sha256 | 0edb0cfb0df991a63da7a5462bf6735db9eea13b41cd9ab5ccac8042c435507c |
| fold_manifest_sha256 | 5ddcb7e9d88f69acfd9f63197aaf6605f5fa09242e33e80eab352b64d23e923f |
| stage7_oof_prediction_sha256 | abd0737f3da7850f24513e383ce196cbd15651c44abbcb125ba2e85cc25a8052 |
| stage6_oof_prediction_sha256 | 4f4f81933e099a9b03705cff006fe30facdd1c2264dde3456f0fcab4991cf235 |
| stage7_model_manifest_sha256 | 6719649182b82edd759cab2719430b90ee37ce018c7c501d8272e3c6fa429a9a |
| stage8_selection_audit_sha256 | b70692551d1ea3960c74136d074242cf200fb65354384e82bc83289146aef699 |
| config_sha256 | 3be1d75979bc588b80a8e1618fd8e185657240f8ab4813aac889bb4478d4eec3 |
| script_sha256 | 642a2d51002feb1fa727b9a5eaaad92d9a65fe2ec806ea2587580855bca5b6a2 |

## Pooled Calibration Metrics
| method | subset_name | sample_count | roc_auc | average_precision | log_loss | brier_score | ece_equal_width | ece_equal_frequency | mce_equal_width | mce_equal_frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNCALIBRATED | DENSE | 420085 | 0.545479 | 0.540899 | 0.690218 | 0.248534 | 0.00604169 | 0.00677441 | 0.0460354 | 0.0269528 |
| UNCALIBRATED | NONOVERLAP_OFFSET_00 | 35007 | 0.560766 | 0.555334 | 0.687625 | 0.247248 | 0.00631881 | 0.00852871 | 0.0187975 | 0.0180086 |
| PLATT | DENSE | 420085 | 0.544429 | 0.539611 | 0.690271 | 0.248563 | 0.00650665 | 0.007634 | 0.0646618 | 0.0153605 |
| PLATT | NONOVERLAP_OFFSET_00 | 35007 | 0.559119 | 0.553605 | 0.688267 | 0.247567 | 0.0127498 | 0.0142964 | 0.0230071 | 0.0304924 |
| ISOTONIC | DENSE | 420085 | 0.543737 | 0.538992 | 0.690468 | 0.248655 | 0.00448358 | 0.00771131 | 0.153259 | 0.0173254 |
| ISOTONIC | NONOVERLAP_OFFSET_00 | 35007 | 0.558297 | 0.552832 | 0.68846 | 0.247659 | 0.0150297 | 0.0179083 | 0.0996806 | 0.0295821 |

## Selection
| field | value |
| --- | --- |
| selected_calibration_method | UNCALIBRATED |
| calibration_improvement_material | False |
| primary_rule_choice | UNCALIBRATED |
| selection_reason | Material improvement gate not met; keeping UNCALIBRATED. |
| engineering_qualified_methods | ['UNCALIBRATED', 'PLATT', 'ISOTONIC'] |

## Ranked Methods
| method | pooled_nonoverlap_log_loss | pooled_nonoverlap_brier | pooled_dense_log_loss | pooled_nonoverlap_ece_equal_frequency | worst_year_log_loss | complexity_rank |
| --- | --- | --- | --- | --- | --- | --- |
| UNCALIBRATED | 0.687625 | 0.247248 | 0.690218 | 0.00852871 | 0.690081 | 0 |
| PLATT | 0.688267 | 0.247567 | 0.690271 | 0.0142964 | 0.690703 | 1 |
| ISOTONIC | 0.68846 | 0.247659 | 0.690468 | 0.0179083 | 0.690588 | 2 |

