# Benchmark CSV normalization log

- Input: `trace_doc/kinetic_regress_benchmark.csv` (1847 rows)
- Output: `data/benchmark/kinetic_regress_benchmark_clean.csv` (1819 rows)
- Dropped: 28 (34 required-field, 1 RMSE-NaN, 1 duplicate)

## Renames applied
### `Dataset`
- `'Test'` → `'test'`
- `'Validation'` → `'validation'`

### `dataset_name`
- `'in house '` → `'inhouse'`

### `embedding_type`
- `'     ESM2'` → `'ESM2'`
- `' MutaPLM'` → `'MutaPLM'`
- `' ProteinCLIP'` → `'ProteinCLIP'`
- `'ESM1V'` → `'ESMv1'`
- `'ESM1v'` → `'ESMv1'`
- `'MUTAPLM'` → `'MutaPLM'`

### `k_type`
- `'KCAT'` → `'kcat'`
- `'KKM'` → `'kcat_km'`
- `'Kcat'` → `'kcat'`
- `'Kd'` → `'kd'`
- `'Km'` → `'km'`
- `'kd '` → `'kd'`
- `'kkm'` → `'kcat_km'`

### `split_mode`
- `'cold drug'` → `'cold_drug'`
- `'cold drug '` → `'cold_drug'`
- `'cold protein'` → `'cold_protein'`
- `'cold protein '` → `'cold_protein'`
- `'cold protein  '` → `'cold_protein'`
- `'embedding random'` → `'random'`
- `'embedding random '` → `'random'`
- `'random '` → `'random'`

## Required-field drop counts
- `k_type`: 26 rows lacked a canonical value
- `split_mode`: 2 rows lacked a canonical value
- `embedding_type`: 2 rows lacked a canonical value
- `dataset_name`: 2 rows lacked a canonical value
- `Dataset`: 2 rows lacked a canonical value

## Distinct values after cleanup
- `k_type`: ['kcat', 'kcat_km', 'kd', 'ki', 'km']
- `split_mode`: ['cold_drug', 'cold_protein', 'random']
- `embedding_type`: ['ESM2', 'ESMv1', 'MutaPLM', 'ProteinCLIP']
- `dataset_name`: ['EITLEM', 'MPEK', 'catpred', 'inhouse']
- `Dataset`: ['test', 'validation']
- `Model`: ['Diffusion', 'GBM', 'Linear Regression', 'MLP', 'PSICHIC', 'Random Forest', 'SVR', 'XGBoost', 'catpred']
