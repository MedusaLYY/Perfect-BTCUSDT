# 阶段1：agg trades独立清洗报告

- 范围限制: 只清洗agg trades，不聚合到分钟，不合并K线，不生成滚动订单流特征、标签或模型。
- 输出Parquet: `C:\ai\BTCUSDT\data\processed\BTCUSDT_spot_agg_trades_clean.parquet`
- 输出文件大小(bytes): `354478760`

## 行数与时间范围
| 指标 | 值 |
| --- | --- |
| 输入行数 | 10912000 |
| 输出行数 | 10912000 |
| 总删除行数 | 0 |
| 开始时间UTC | 2022-09-02T00:00:00+00:00 |
| 结束时间UTC | 2022-09-04T18:46:05.068000+00:00 |

## 删除统计
| 原因 | 数量 |
| --- | --- |
| missing_agg_trade_id | 0 |
| duplicate_agg_trade_id_keep_last_original_order | 0 |
| missing_trade_time | 0 |
| nonfinite_price | 0 |
| nonfinite_quantity | 0 |
| nonpositive_price | 0 |
| nonpositive_quantity | 0 |
| invalid_is_buyer_maker | 0 |
| invalid_rows_after_dedup | 0 |

## is_buyer_maker解析
| 指标 | 值 |
| --- | --- |
| 非法值计数 | {} |
| 非法值样例 | [] |

## 时间与ID连续性
| 指标 | 值 |
| --- | --- |
| 原始顺序成交时间倒退数 | 0 |
| 排序后成交时间倒退数 | 0 |
| 同一毫秒多笔成交的毫秒数 | 1733101 |
| 同一毫秒多笔成交涉及行数 | 6958464 |
| 同一毫秒最大成交笔数 | 548 |
| 超长无成交阈值(ms) | 60000 |
| 超长无成交间隔数 | 0 |
| 最大无成交间隔(ms) | 0 |
| ID gap记录数 | 5268 |
| 缺失ID总数 | 29548 |
| 最大ID gap缺失数 | 279 |
| ID倒退数 | 0 |

## 输出Parquet Schema
| name | type |
| --- | --- |
| symbol | string |
| agg_trade_id | int64 |
| price | double |
| quantity | double |
| first_trade_id | int64 |
| last_trade_id | int64 |
| trade_time | int64 |
| trade_time_utc | large_string |
| is_buyer_maker | bool |
| is_best_match | bool |
| quote_quantity | double |
| is_active_buy | bool |
| is_active_sell | bool |
| id_gap_before | int64 |

## 说明
- 未补造缺失成交ID。
- 未聚合agg trades到分钟。
- 未构造滚动订单流特征，未使用OFI命名。
- `is_active_buy = not is_buyer_maker`; `is_active_sell = is_buyer_maker`。
