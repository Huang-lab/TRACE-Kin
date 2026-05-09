# TRACE-Kin v3 Promotion Gate Decision

Source: `data/benchmark/trace_kin_v3_results.csv` (14 v3 result rows)

## Catalytic kinetics gate
### `kcat`: PASS
- 3/3 splits beat RF; mean v3=1.0167 vs mean RF=1.0445. by_count_pass=True, by_mean_pass=True
  - MPEK_kcat_ESMv1_embedding_cold_drug: v3=1.087861665577792, RF=1.131657045, gap=-0.0438
  - MPEK_kcat_ESM2_embedding_cold_protein: v3=1.154997794629122, RF=1.166934154, gap=-0.0119
  - MPEK_kcat_ESMv1_embedding_random: v3=0.8073103175296037, RF=0.834783124, gap=-0.0275

### `km`: PASS
- 2/2 splits beat RF; mean v3=0.6539 vs mean RF=0.8089. by_count_pass=True, by_mean_pass=True
  - EITLEM_km_ProteinCLIP_embedding_cold_protein: v3=0.6982583751807458, RF=0.848788617, gap=-0.1505
  - EITLEM_km_ESMv1_embedding_random: v3=0.6095208235268159, RF=0.769015358, gap=-0.1595

### `kd`: FAIL
- 0/3 splits beat RF; mean v3=1.1807 vs mean RF=1.1009. by_count_pass=False, by_mean_pass=False
  - inhouse_Kd_ESM2_embedding_cold_drug: v3=1.3006337304565052, RF=1.160384137, gap=+0.1402
  - inhouse_Kd_ESM2_embedding_cold_protein: v3=1.2347470112010916, RF=1.158869143, gap=+0.0759
  - inhouse_Kd_MUTAPLM_embedding_random: v3=1.006802643102997, RF=0.98345077, gap=+0.0234

### `kcat_km`: FAIL
- 0/3 splits beat RF; mean v3=1.4253 vs mean RF=1.3451. by_count_pass=False, by_mean_pass=False
  - EITLEM_kkm_MUTAPLM_embedding_cold_drug: v3=1.4923771673696935, RF=1.434169498, gap=+0.0582
  - EITLEM_kkm_ProteinCLIP_embedding_cold_protein: v3=1.458554045924037, RF=1.345698365, gap=+0.1129
  - EITLEM_kkm_ProteinCLIP_embedding_random: v3=1.325045915909914, RF=1.255293405, gap=+0.0698

**Catalytic gate: 2/4 kinetics passed → FAIL**

## Ki preservation guard
Tolerance: v3 may not be more than +0.02 RMSE worse than v1 on Ki tasks.
  - catpred_ki_ProteinCLIP_embedding_cold_drug: v3=1.3038, v1=1.2875, Δ=+0.0163 (OK)
  - catpred_ki_ProteinCLIP_embedding_cold_protein: v3=1.5066, v1=1.5863, Δ=-0.0797 (OK)
  - catpred_ki_ProteinCLIP_embedding_random: v3=1.2772, v1=1.2284, Δ=+0.0488 (GUARD FAIL)
**Ki guard: FAIL**

---
## Overall verdict: **FAIL**
- Do NOT generate paper figures/tables yet.
- Catalytic gate failed: review per-kinetic breakdown above and iterate v3 design.
- Ki guard failed: ship v1 weights for Ki tasks; v3 weights for everything else if catalytic gate passed.

