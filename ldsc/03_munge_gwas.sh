#!/usr/bin/env bash
# Step 3 — munge one GWAS trait for LDSC. Output: <output_dir>/munged/<trait_id>.sumstats.gz
#
# Required: --trait_id --gwas_manifest --gwas_dir --output_dir
#   gwas_manifest columns: trait_id, sample_size, extra_munge_args
#   sumstats resolved as: <gwas_dir>/imputed_<trait_id>.txt.gz
# Requires ``munge_sumstats.py`` on PATH (pixi env ``ldsc-cli``).
set -o nounset -o errexit -o pipefail

trait_id=""; gwas_manifest=""; gwas_dir=""; output_dir=""
chunksize="5000000"

arguments=$(getopt --options "" --longoptions "trait_id:,gwas_manifest:,gwas_dir:,output_dir:" --name 03_munge_gwas -- "$@")
eval set -- "${arguments}"
while true; do
  case "${1}" in
    --) shift; break ;;
    --*) printf -v "${1#--}" '%s' "${2}"; shift 2 ;;
  esac
done
for v in trait_id gwas_manifest gwas_dir output_dir; do
  [[ -n "${!v}" ]] || { echo "03_munge_gwas: --${v} is required" >&2; exit 1; }
done

out_prefix="${output_dir}/munged/${trait_id}"
mkdir -p "${output_dir}/munged"

echo "[03_munge_gwas] start $(date -Is) trait_id=${trait_id} output_prefix=${out_prefix}"

header="$(head -1 "${gwas_manifest}")"
manifest_row="$(awk -F'\t' -v t="${trait_id}" 'NR>1 && $1==t {print; exit}' "${gwas_manifest}")"
[[ -n "${manifest_row}" ]] || { echo "Trait not in manifest: ${trait_id}" >&2; exit 1; }

col_index() {
  # 1-based column index for a header name
  echo "${header}" | tr '\t' '\n' | grep -n "^${1}$" | cut -d: -f1
}
extra_col="$(col_index extra_munge_args)"
extra_args=""
if [[ -n "${extra_col}" ]]; then
  extra_args="$(echo "${manifest_row}" | awk -F'\t' -v c="${extra_col}" '{print $c}')"
fi

sumstats_path="${gwas_dir}/imputed_${trait_id}.txt.gz"
[[ -s "${sumstats_path}" ]] || { echo "GWAS file missing for ${trait_id}: ${sumstats_path}" >&2; exit 1; }

command -v munge_sumstats.py >/dev/null 2>&1 || { echo "munge_sumstats.py not found on PATH (activate pixi env ldsc-cli)" >&2; exit 1; }

munge_input="${sumstats_path}"
tmp_sumstats=""
if [[ "${sumstats_path}" == *.gz ]]; then
  tmp_sumstats="$(mktemp "${TMPDIR:-/tmp}/ldsc_${trait_id}.XXXXXX.sumstats.txt")"
  gzip -dc "${sumstats_path}" > "${tmp_sumstats}"
  munge_input="${tmp_sumstats}"
fi
trap '[[ -n "${tmp_sumstats}" && -f "${tmp_sumstats}" ]] && rm -f "${tmp_sumstats}"' EXIT

cmd=(
  munge_sumstats.py
  --sumstats "${munge_input}"
  --out "${out_prefix}"
  --chunksize "${chunksize}"
  --snp "variant_id"
  --N-col "sample_size"
  --a1 "effect_allele"
  --a2 "non_effect_allele"
  --p "pvalue"
  --signed-sumstats "zscore,0"
)
# extra_args may carry per-trait column mappings from the manifest.
"${cmd[@]}" ${extra_args:-}

echo "[03_munge_gwas] done $(date -Is) trait_id=${trait_id}"
