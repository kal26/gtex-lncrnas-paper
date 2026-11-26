#!/usr/bin/env bash
# Step 2 — compute LD scores for one annotation (chromosomes serially).
# Expects <ldscores_dir>/<annotation_name>/<annotation_name>.<chr>.annot.gz (from 01_build_annotations.py).
#
# Required: --annotation_name --ldscores_dir --bfile_prefix
#           --baseline_snplist (SNP list extracted from baselineLD; not HM3)
# Always uses chromosomes 1–22 and --ld-wind-cm 1.
# Requires ``ldsc.py`` on PATH (pixi env ``ldsc-cli``).
set -o nounset -o errexit -o pipefail

annotation_name=""; ldscores_dir=""; bfile_prefix=""; baseline_snplist=""
chromosomes="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
ld_wind_cm="1"

arguments=$(getopt --options "" --longoptions "annotation_name:,ldscores_dir:,bfile_prefix:,baseline_snplist:" --name 02_compute_ldscores -- "$@")
eval set -- "${arguments}"
while true; do
  case "${1}" in
    --) shift; break ;;
    --*) printf -v "${1#--}" '%s' "${2}"; shift 2 ;;
  esac
done
for v in annotation_name ldscores_dir bfile_prefix baseline_snplist; do
  [[ -n "${!v}" ]] || { echo "02_compute_ldscores: --${v} is required" >&2; exit 1; }
done
[[ -s "${baseline_snplist}" ]] || { echo "baseline_snplist missing/empty: ${baseline_snplist}" >&2; exit 1; }

out_dir="${ldscores_dir}/${annotation_name}"
mkdir -p "${out_dir}"
echo "[02_compute_ldscores] start $(date -Is) annotation_name=${annotation_name} out_dir=${out_dir}"

command -v ldsc.py >/dev/null 2>&1 || { echo "ldsc.py not found on PATH (activate pixi env ldsc-cli)" >&2; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

for chr in ${chromosomes}; do
  annot_file="${out_dir}/${annotation_name}.${chr}.annot.gz"
  out_prefix="${out_dir}/${annotation_name}.${chr}"
  echo "[02_compute_ldscores] chr=${chr} annot=${annot_file}"
  ldsc.py --l2 --bfile "${bfile_prefix}${chr}" --ld-wind-cm "${ld_wind_cm}" --annot "${annot_file}" --print-snps "${baseline_snplist}" --out "${out_prefix}"
done
echo "[02_compute_ldscores] done $(date -Is) annotation_name=${annotation_name}"
