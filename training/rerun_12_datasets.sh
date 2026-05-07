#!/bin/bash
# Submit the v2 redesign rerun for the 12 RF-gap-target datasets + 2 Ki preservation guards.
#
# **Prefer the LSF job array `training/run_v2_rerun_array.lsf` instead.** It is
# a pure `#BSUB`-driven LSF file you submit with `bsub < ...lsf`, matches the
# user's HPC submission pattern, and supports clean chaining off the smoke
# test via `-w "done(<smoke_id>)"`. This bash wrapper is kept as a fallback
# for users who want bash-loop control.
#
# Default DRY_RUN=true: print bsub commands without submitting. To actually launch:
#     DRY_RUN=false bash training/rerun_12_datasets.sh
#
# Cross-dataset pooling: for kcat and Km the v2 plan trains on pooled
# MPEK + EITLEM + CatPred data. For Kd, kcat/Km, and Ki we use the single-dataset
# training the historical benchmark used. The pool path lists are written
# relative to ENZYME_BASE; they're injected into POOL_TRAIN_CSVS.
#
# Promotion gate (analysis/promotion_gate.py): v2 must beat or tie RF on >=3 of
# {kcat, Km, Kd, kcat/Km}, while keeping Ki within +0.02 RMSE of v1.
set -euo pipefail

WORKDIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/TRACE_Kin"
RESULT_ROOT="${RESULT_ROOT:-/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/TRACE_Kin_Results_v2}"
ENZYME_BASE="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/drug_discovery/output/data/enzyme_embeddings_dataset"
CONFIG_PATH="${CONFIG_PATH:-training/config_v2.json}"
DRY_RUN="${DRY_RUN:-true}"
SEED="${SEED:-1}"
USE_SWA="${USE_SWA:-true}"

cd "$WORKDIR"

# -----------------------------------------------------------------------------
# 14 jobs: 12 RF-gap targets + 2 Ki preservation guards.
# Each row: SOURCE_DIR | DATASET_NAME | POOL_DIRS (semicolon-separated, optional)
# -----------------------------------------------------------------------------
read -r -d '' JOBS <<'EOF' || true
EITLEM_dataset|EITLEM_km_ESMv1_embedding_random|MPEK_dataset/MPEK_km_ESMv1_embedding_random;catpred_dataset/catpred_km_ESMv1_embedding_random
EITLEM_dataset|EITLEM_km_ProteinCLIP_embedding_cold_protein|MPEK_dataset/MPEK_km_ProteinCLIP_embedding_cold_protein;catpred_dataset/catpred_km_ProteinCLIP_embedding_cold_protein
MPEK_dataset|MPEK_kcat_ESMv1_embedding_random|EITLEM_dataset/EITLEM_kcat_ESMv1_embedding_random;catpred_dataset/catpred_kcat_ESMv1_embedding_random
MPEK_dataset|MPEK_kcat_ESMv1_embedding_cold_drug|EITLEM_dataset/EITLEM_kcat_ESMv1_embedding_cold_drug;catpred_dataset/catpred_kcat_ESMv1_embedding_cold_drug
MPEK_dataset|MPEK_kcat_ESM2_embedding_cold_protein|EITLEM_dataset/EITLEM_kcat_ESM2_embedding_cold_protein;catpred_dataset/catpred_kcat_ESM2_embedding_cold_protein
inhouse_dataset|inhouse_Kd_MUTAPLM_embedding_random|
inhouse_dataset|inhouse_Kd_ESM2_embedding_cold_drug|
inhouse_dataset|inhouse_Kd_ESM2_embedding_cold_protein|
EITLEM_dataset|EITLEM_kkm_ProteinCLIP_embedding_random|
EITLEM_dataset|EITLEM_kkm_MUTAPLM_embedding_cold_drug|
EITLEM_dataset|EITLEM_kkm_ProteinCLIP_embedding_cold_protein|
catpred_dataset|catpred_ki_ProteinCLIP_embedding_cold_protein|
catpred_dataset|catpred_ki_ProteinCLIP_embedding_random|
catpred_dataset|catpred_ki_ProteinCLIP_embedding_cold_drug|
EOF

JOB_NUM=0
while IFS='|' read -r SRC NAME POOLS; do
    [ -z "$SRC" ] && continue
    JOB_NUM=$((JOB_NUM + 1))

    POOL_TRAIN_CSVS=""
    if [ -n "$POOLS" ]; then
        # Convert ';'-separated relative paths into a comma-separated absolute-path list.
        IFS=';' read -ra REL_LIST <<< "$POOLS"
        ABS_LIST=()
        for rel in "${REL_LIST[@]}"; do
            ABS_LIST+=("$ENZYME_BASE/$rel")
        done
        POOL_TRAIN_CSVS=$(IFS=','; echo "${ABS_LIST[*]}")
    fi

    CMD="DATASET_SOURCE=$SRC SINGLE_DATASET=$NAME RESULT_BASE=$RESULT_ROOT SKIP_COMPLETED=false"
    CMD="$CMD CONFIG_PATH=$CONFIG_PATH MODEL_VERSION=v2 USE_SWA=$USE_SWA SEED=$SEED"
    if [ -n "$POOL_TRAIN_CSVS" ]; then
        CMD="$CMD POOL_TRAIN_CSVS=\"$POOL_TRAIN_CSVS\""
    fi
    CMD="$CMD bsub < training/run_benchmark.lsf"

    echo "[$JOB_NUM/14] $NAME"
    if [ "$DRY_RUN" = "true" ]; then
        echo "  $CMD"
    else
        eval "$CMD"
    fi
done <<< "$JOBS"

echo ""
echo "Total jobs: $JOB_NUM (DRY_RUN=$DRY_RUN). Set DRY_RUN=false to submit."
