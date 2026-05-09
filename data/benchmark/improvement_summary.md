# TRACE-Kin v3 improvement summary

Source: `data/benchmark/trace_kin_v3_results.csv` (14 rows)

## Headline

v3 improves over v1 by mean +0.0250 RMSE across 14 paired rows. v3 beats or ties RF on 7/14 reran rows. Catalytic gate: 2/4 kinetics pass. Ki preservation: largest v3−v1 regression = +0.0488.

## Per-task results

| dataset_folder | k_type | split_mode | embedding | v1_rmse | v3_rmse | rf_best_rmse | v3_minus_v1 | v3_minus_rf | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MPEK_kcat_ESMv1_embedding_cold_drug | kcat | cold_drug | ESMv1 | 1.1600 | 1.0879 | 1.1317 | -0.0721 | -0.0438 | ✓ beats RF |
| MPEK_kcat_ESM2_embedding_cold_protein | kcat | cold_protein | ESM2 | 1.1760 | 1.1550 | 1.1669 | -0.0210 | -0.0119 | ✓ beats RF |
| MPEK_kcat_ESMv1_embedding_random | kcat | random | ESMv1 | 0.8758 | 0.8073 | 0.8348 | -0.0685 | -0.0275 | ✓ beats RF |
| EITLEM_kkm_MUTAPLM_embedding_cold_drug | kcat_km | cold_drug | MutaPLM | 1.4715 | 1.4924 | 1.4342 | 0.0209 | 0.0582 | ✗ worse than v1 |
| EITLEM_kkm_ProteinCLIP_embedding_cold_protein | kcat_km | cold_protein | ProteinCLIP | 1.4258 | 1.4586 | 1.3457 | 0.0328 | 0.1129 | ✗ worse than v1 |
| EITLEM_kkm_ProteinCLIP_embedding_random | kcat_km | random | ProteinCLIP | 1.2249 | 1.3250 | 1.2553 | 0.1001 | 0.0698 | ✗ worse than v1 |
| inhouse_Kd_ESM2_embedding_cold_drug | kd | cold_drug | ESM2 | 1.2162 | 1.3006 | 1.1604 | 0.0844 | 0.1402 | ✗ worse than v1 |
| inhouse_Kd_ESM2_embedding_cold_protein | kd | cold_protein | ESM2 | 1.2056 | 1.2347 | 1.1589 | 0.0291 | 0.0759 | ✗ worse than v1 |
| inhouse_Kd_MUTAPLM_embedding_random | kd | random | MutaPLM | 1.0221 | 1.0068 | 0.9835 | -0.0153 | 0.0234 | ✗ worse than RF |
| catpred_ki_ProteinCLIP_embedding_cold_drug | ki | cold_drug | ProteinCLIP | 1.2875 | 1.3038 | 1.4180 | 0.0163 | -0.1142 | ✗ worse than v1 |
| catpred_ki_ProteinCLIP_embedding_cold_protein | ki | cold_protein | ProteinCLIP | 1.5863 | 1.5066 | 1.5016 | -0.0797 | 0.0050 | ~ ties RF |
| catpred_ki_ProteinCLIP_embedding_random | ki | random | ProteinCLIP | 1.2284 | 1.2772 | 1.2852 | 0.0488 | -0.0080 | ✗ worse than v1 |
| EITLEM_km_ProteinCLIP_embedding_cold_protein | km | cold_protein | ProteinCLIP | 0.9383 | 0.6983 | 0.8488 | -0.2400 | -0.1505 | ✓ beats RF |
| EITLEM_km_ESMv1_embedding_random | km | random | ESMv1 | 0.7955 | 0.6095 | 0.7690 | -0.1860 | -0.1595 | ✓ beats RF |

## Per-kinetic aggregate

| k_type | n_splits | beats_rf | mean_v1 | mean_v3 | mean_rf | mean_delta_v1 | mean_delta_rf | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kcat | 3 | 3 | 1.0706 | 1.0167 | 1.0445 | -0.0539 | -0.0277 | PASS |
| kcat_km | 3 | 0 | 1.3741 | 1.4253 | 1.3451 | 0.0513 | 0.0803 | FAIL |
| kd | 3 | 0 | 1.1480 | 1.1807 | 1.1009 | 0.0328 | 0.0798 | FAIL |
| ki | 3 | 2 | 1.3674 | 1.3625 | 1.4016 | -0.0049 | -0.0391 | PASS |
| km | 2 | 2 | 0.8669 | 0.6539 | 0.8089 | -0.2130 | -0.1550 | PASS |

## Legend

- `✓ beats RF` — v3 RMSE ≤ RF best RMSE
- `~ ties RF` — v3 within 0.005 RMSE of RF best (above)
- `✗ worse than RF` — v3 RMSE > RF best (and not within tie tolerance)
- `✗ worse than v1` — v3 RMSE > v1 RMSE (regression vs. baseline)

Per-kinetic verdict mirrors `analysis/promotion_gate.py`: PASS if ≥2/3 splits beat RF or the per-kinetic mean v3 RMSE ≤ mean RF RMSE.

