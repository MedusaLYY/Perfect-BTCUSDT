# 阶段3A：短窗口60分钟方向标签代理敏感性验证报告

- 范围限制: 仅使用2022-09-02至2022-09-04短重叠窗口；未构造完整2017-2026标签；未生成模型特征、切分或模型。
- 输出: `C:\ai\BTCUSDT\data\interim\BTCUSDT_stage3a_label_proxy_comparison.parquet`
- 输出文件大小(bytes): `311299`
- 运行耗时(seconds): `0.88`

## 样本数量
| 指标 | 值 |
| --- | --- |
| 输出行数 | 790 |
| label_only候选数 | 790 |
| model_aligned候选数 | 790 |
| primary reliable样本数 | 595 |

## VWAP窗口标签覆盖
| 窗口 | valid_count | coverage_of_label_only |
| --- | --- | --- |
| vwap_1s | 790 | 1.0 |
| vwap_2s | 790 | 1.0 |
| vwap_5s | 790 | 1.0 |
| vwap_10s | 790 | 1.0 |

## K线open vs 前5秒VWAP标签一致性
| sample_count | agreement_count | flip_count | agreement_rate | agreement_rate_ci95_low | agreement_rate_ci95_high | flip_rate | flip_rate_ci95_low | flip_rate_ci95_high | cohen_kappa | mcc | kline_up_rate | reference_up_rate | up_to_down_count | down_to_up_count | sample_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 790 | 776 | 14 | 0.9822784810126582 | 0.970474609980503 | 0.9894147880008058 | 0.017721518987341773 | 0.010585211999194124 | 0.029525390019497114 | 0.9643228108205754 | 0.9643228108205754 | 0.45949367088607596 | 0.45949367088607596 | 7 | 7 | all_comparable |
| 595 | 586 | 9 | 0.984873949579832 | 0.9715050329441535 | 0.9920221101769879 | 0.015126050420168067 | 0.00797788982301206 | 0.028494967055846625 | 0.9694783099362209 | 0.9695284882963654 | 0.45546218487394957 | 0.4504201680672269 | 6 | 3 | primary_reliable |
| 790 | 776 | 14 | 0.9822784810126582 | 0.970474609980503 | 0.9894147880008058 | 0.017721518987341773 | 0.010585211999194124 | 0.029525390019497114 | 0.9643228108205754 | 0.9643228108205754 | 0.45949367088607596 | 0.45949367088607596 | 7 | 7 | model_aligned |
| 790 | 776 | 14 | 0.9822784810126582 | 0.970474609980503 | 0.9894147880008058 | 0.017721518987341773 | 0.010585211999194124 | 0.029525390019497114 | 0.9643228108205754 | 0.9643228108205754 | 0.45949367088607596 | 0.45949367088607596 | 7 | 7 | label_only |
| 765 | 758 | 7 | 0.9908496732026144 | 0.9812337093413818 | 0.9955606497348565 | 0.009150326797385621 | 0.004439350265143443 | 0.01876629065861823 | 0.9815621395492968 | 0.98156554453294 | 0.4562091503267974 | 0.45751633986928103 | 3 | 4 | abs_return_kline_ge_1bps |
| 725 | 723 | 2 | 0.9972413793103448 | 0.9899980404719718 | 0.9992431604424464 | 0.002758620689655172 | 0.0007568395575534986 | 0.010001959528028213 | 0.9944323278245377 | 0.9944323278245377 | 0.45241379310344826 | 0.45241379310344826 | 1 | 1 | abs_return_kline_ge_2.5bps |
| 672 | 672 | 0 | 1.0 | 0.9943160355575149 | 1.0 | 0.0 | 0.0 | 0.005683964442485163 | 1.0 | 1.0 | 0.4479166666666667 | 0.4479166666666667 | 0 | 0 | abs_return_kline_ge_5bps |
| 562 | 562 | 0 | 1.0 | 0.9932110686468603 | 1.0 | 0.0 | 0.0 | 0.006788931353139716 | 1.0 | 1.0 | 0.4501779359430605 | 0.4501779359430605 | 0 | 0 | abs_return_kline_ge_10bps |

## K线open与各VWAP窗口标签一致性
| 窗口 | sample_count | agreement_count | flip_count | agreement_rate | agreement_rate_ci95_low | agreement_rate_ci95_high | cohen_kappa | mcc | up_to_down_count | down_to_up_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vwap_1s | 790 | 782 | 8 | 0.9898734177215189 | 0.9801458121436504 | 0.9948599543098049 | 0.9796130347546144 | 0.9796130347546145 | 4 | 4 |
| vwap_2s | 790 | 777 | 13 | 0.9835443037974684 | 0.9720504271951759 | 0.9903583655314908 | 0.9668643406100575 | 0.966867481423881 | 7 | 6 |
| vwap_5s | 790 | 776 | 14 | 0.9822784810126582 | 0.970474609980503 | 0.9894147880008058 | 0.9643228108205754 | 0.9643228108205754 | 7 | 7 |
| vwap_10s | 790 | 774 | 16 | 0.979746835443038 | 0.9673549935487981 | 0.9874956149417952 | 0.9592092269761258 | 0.9592216960936345 | 9 | 7 |

## 连续收益比较
| 指标 | 值 |
| --- | --- |
| Pearson | 0.9981456895632609 |
| Spearman | 0.9980032379234866 |
| 收益差均值 | -9.433021543726338e-07 |
| 收益差中位数 | 3.503195224463518e-07 |
| 收益差bps均值 | -0.009433021543726338 |
| 收益差bps p95绝对值 | 3.848901942043791 |
| 收益差bps p99绝对值 | 7.887659667998649 |
| 收益差bps最大绝对值 | 33.806409019829786 |
| entry价格差bps均值 | -0.026954995437842143 |
| settlement价格差bps均值 | -0.035834641516889254 |

## ID gap与标签翻转
| flip_count | flips_with_entry_id_gap | flips_with_settlement_id_gap | nonflips_with_entry_id_gap | nonflips_with_settlement_id_gap |
| --- | --- | --- | --- | --- |
| 14 | 12 | 13 | 580 | 581 |

## 诊断阈值
| sample_size_check_passed | overall_agreement_check_passed | non_boundary_agreement_check_passed | agreement_ge_5bps_check_passed | vwap_coverage_check_passed | return_bias_check_passed | proxy_label_recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| True | True | True | True | True | True | ACCEPT_WITH_LIMITATIONS |

## 结论
在当前短重叠窗口中，K线open生成的60分钟方向标签与前5秒VWAP标签高度一致，可作为完整历史标签的工程代理，但必须保留代理误差和局限性说明。该结论不能证明2017-2026所有年份都有相同代理误差。

## 输出Schema
| name | type |
| --- | --- |
| feature_open_time | int64 |
| decision_time | int64 |
| entry_minute_open_time | int64 |
| settlement_minute_open_time | int64 |
| is_prediction_time_5m | bool |
| is_model_candidate | bool |
| is_label_only_candidate | bool |
| is_model_aligned_candidate | bool |
| is_primary_reliable_sample | bool |
| entry_exists | bool |
| settlement_exists | bool |
| entry_is_reliable_overlap_minute | bool |
| settlement_is_reliable_overlap_minute | bool |
| entry_is_boundary_partial_minute | bool |
| settlement_is_boundary_partial_minute | bool |
| entry_has_any_id_gap | bool |
| settlement_has_any_id_gap | bool |
| entry_id_gap_event_count | int64 |
| settlement_id_gap_event_count | double |
| entry_cross_minute_id_gap_event_count | int64 |
| settlement_cross_minute_id_gap_event_count | double |
| entry_kline_base_volume | double |
| settlement_kline_base_volume | double |
| entry_and_settlement_reliable | bool |
| entry_kline_volume_quantile_bucket | large_string |
| entry_price_kline_open | double |
| settlement_price_kline_open | double |
| return_kline_open | double |
| log_return_kline_open | double |
| label_kline_open | int8 |
| entry_price_vwap_1s | double |
| settlement_price_vwap_1s | double |
| has_entry_vwap_1s | bool |
| has_settlement_vwap_1s | bool |
| is_valid_vwap_1s_label | bool |
| return_vwap_1s | double |
| log_return_vwap_1s | double |
| label_vwap_1s | int8 |
| entry_price_vwap_2s | double |
| settlement_price_vwap_2s | double |
| has_entry_vwap_2s | bool |
| has_settlement_vwap_2s | bool |
| is_valid_vwap_2s_label | bool |
| return_vwap_2s | double |
| log_return_vwap_2s | double |
| label_vwap_2s | int8 |
| entry_price_vwap_5s | double |
| settlement_price_vwap_5s | double |
| has_entry_vwap_5s | bool |
| has_settlement_vwap_5s | bool |
| is_valid_vwap_5s_label | bool |
| return_vwap_5s | double |
| log_return_vwap_5s | double |
| label_vwap_5s | int8 |
| entry_price_vwap_10s | double |
| settlement_price_vwap_10s | double |
| has_entry_vwap_10s | bool |
| has_settlement_vwap_10s | bool |
| is_valid_vwap_10s_label | bool |
| return_vwap_10s | double |
| log_return_vwap_10s | double |
| label_vwap_10s | int8 |
| label_agree_kline_vs_5s | bool |
| label_flip_kline_vs_5s | bool |
| return_difference_kline_vs_5s | double |
| absolute_return_difference_kline_vs_5s | double |
| entry_price_difference_bps_kline_vs_5s | double |
| settlement_price_difference_bps_kline_vs_5s | double |
| return_difference_bps_kline_vs_5s | double |
| absolute_return_kline_open_bps | double |
| absolute_return_vwap_5s_bps | double |
| absolute_return_kline_open_bps_bucket | large_string |
| absolute_return_vwap_5s_bps_bucket | large_string |
| decision_hour_utc | int8 |
| decision_date_utc | large_string |
| label_agree_kline_open_vs_vwap_1s | bool |
| label_flip_kline_open_vs_vwap_1s | bool |
| return_difference_bps_kline_open_vs_vwap_1s | double |
| label_agree_vwap_1s_vs_vwap_5s | bool |
| label_flip_vwap_1s_vs_vwap_5s | bool |
| return_difference_bps_vwap_1s_vs_vwap_5s | double |
| label_agree_kline_open_vs_vwap_2s | bool |
| label_flip_kline_open_vs_vwap_2s | bool |
| return_difference_bps_kline_open_vs_vwap_2s | double |
| label_agree_vwap_2s_vs_vwap_5s | bool |
| label_flip_vwap_2s_vs_vwap_5s | bool |
| return_difference_bps_vwap_2s_vs_vwap_5s | double |
| label_agree_kline_open_vs_vwap_10s | bool |
| label_flip_kline_open_vs_vwap_10s | bool |
| return_difference_bps_kline_open_vs_vwap_10s | double |
| label_agree_vwap_10s_vs_vwap_5s | bool |
| label_flip_vwap_10s_vs_vwap_5s | bool |
| return_difference_bps_vwap_10s_vs_vwap_5s | double |

