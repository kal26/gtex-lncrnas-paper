#!/usr/bin/env bash
# Step 4 — run S-LDSC (--h2) for one trait × model job.
# Results: <output_dir>/results/<trait_id>/<trait_id>.<model_id>.results
#
# Required: --models_manifest --trait_id --model_id --output_dir --munged_dir
#           --baseline_dir --ldscores_dir
#           --weights_ld_prefix --frqfile_prefix --hm3_snplist
# Requires ``ldsc.py`` on PATH (pixi env ``ldsc-cli``).
#
# Models manifest columns:
#   model_id, baseline, annotations, primary_annotation, baseline_track
# Resolved prefixes (order preserved):
#   <baseline_dir>/baselineLD_v2.2_<baseline>/baselineLD.
#   [optional] <baseline_dir>/single_tracks/<baseline_track>/<baseline_track>.
#   for each annot in annotations: <ldscores_dir>/<annot>/<annot>.
set -o nounset -o errexit -o pipefail

models_manifest=""; trait_id=""; model_id=""; output_dir=""; munged_dir=""
baseline_dir=""; ldscores_dir=""
weights_ld_prefix=""; frqfile_prefix=""; hm3_snplist=""

arguments=$(getopt --options "" --longoptions "models_manifest:,trait_id:,model_id:,output_dir:,munged_dir:,baseline_dir:,ldscores_dir:,weights_ld_prefix:,frqfile_prefix:,hm3_snplist:" --name 04_run_ldsc -- "$@")
eval set -- "${arguments}"
while true; do
  case "${1}" in
    --) shift; break ;;
    --*) printf -v "${1#--}" '%s' "${2}"; shift 2 ;;
  esac
done
for v in models_manifest trait_id model_id output_dir munged_dir baseline_dir ldscores_dir weights_ld_prefix frqfile_prefix hm3_snplist; do
  [[ -n "${!v}" ]] || { echo "04_run_ldsc: --${v} is required" >&2; exit 1; }
done

# Limit threading to prevent "Resource temporarily unavailable" errors
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

results_dir="${output_dir}/results/${trait_id}"
mkdir -p "${results_dir}"

header="$(head -1 "${models_manifest}")"
manifest_row="$(awk -F'\t' -v m="${model_id}" 'NR>1 && $1==m {print; exit}' "${models_manifest}")"
[[ -n "${manifest_row}" ]] || { echo "Model not in manifest: ${model_id}" >&2; exit 1; }

col_index() {
  echo "${header}" | tr '\t' '\n' | grep -n "^${1}$" | cut -d: -f1
}
col_value() {
  local c
  c="$(col_index "${1}")"
  [[ -n "${c}" ]] || { echo "Missing column ${1} in ${models_manifest}" >&2; exit 1; }
  echo "${manifest_row}" | awk -F'\t' -v c="${c}" '{print $c}'
}

baseline="$(col_value baseline)"
annotations="$(col_value annotations)"
baseline_track="$(col_value baseline_track)"

prefixes=()
prefixes+=("${baseline_dir}/baselineLD_v2.2_${baseline}/baselineLD.")
if [[ -n "${baseline_track}" ]]; then
  prefixes+=("${baseline_dir}/single_tracks/${baseline_track}/${baseline_track}.")
fi
IFS=',' read -ra annot_list <<< "${annotations}"
for annot in "${annot_list[@]}"; do
  annot="$(echo "${annot}" | tr -d '[:space:]')"
  [[ -n "${annot}" ]] || continue
  prefixes+=("${ldscores_dir}/${annot}/${annot}.")
done
ref_ld_prefixes="$(IFS=,; echo "${prefixes[*]}")"

out_prefix="${results_dir}/${trait_id}.${model_id}"
sumstats_file="${munged_dir}/${trait_id}.sumstats.gz"
echo "[04_run_ldsc] start $(date -Is) trait_id=${trait_id} model_id=${model_id}"
echo "[04_run_ldsc] sumstats=${sumstats_file} out_prefix=${out_prefix}"
echo "[04_run_ldsc] ref_ld_prefixes=${ref_ld_prefixes}"

command -v ldsc.py >/dev/null 2>&1 || { echo "ldsc.py not found on PATH (activate pixi env ldsc-cli)" >&2; exit 1; }

# Regress on HapMap3 SNPs; LD score files stay unfiltered so baselineLD and custom prefixes concatenate.
ldsc.py --h2 "${sumstats_file}" \
  --ref-ld-chr "${ref_ld_prefixes}" \
  --w-ld-chr "${weights_ld_prefix}" \
  --overlap-annot \
  --frqfile-chr "${frqfile_prefix}" \
  --print-snps "${hm3_snplist}" \
  --out "${out_prefix}" \
  --print-coefficients
echo "[04_run_ldsc] done $(date -Is) trait_id=${trait_id} model_id=${model_id}"
