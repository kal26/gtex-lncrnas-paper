#!/usr/bin/env bash
# RNA-SeQC one sample against a matched intergenic-null GTF.
#
# Requires ``rnaseqc`` on PATH (repo pixi env ``intergenic-null``: RNA-SeQC 2.4.2,
# same pin as gtex quantifybam). Same invocation as gtex-pipeline run_rnaseqc.py:
#   rnaseqc <gtf> <bam> <outdir> -s <sample_id> -vv [--stranded rf|fr]
# then gzip the three GCT outputs.
set -o xtrace -o nounset -o pipefail -o errexit

check_for_directory() {
    argument_name="${1}"
    directory_path="${2}"
    if [[ ! -d ${directory_path} ]]; then
        echo "Error: directory ${directory_path} passed with ${argument_name} does not exist."
        exit 1
    fi
}

check_for_file() {
    argument_name="${1}"
    file_path="${2}"
    if [[ ! -f ${file_path} ]]; then
        echo "Error: file ${file_path} passed with ${argument_name} does not exist."
        exit 1
    fi
}

options_array=(
    output_dir
    sample_id
    bam_dir
    genes_gtf
    bam_file_end
    stranded
)

longoptions=$(echo "${options_array[@]}" | sed -e 's/ /:,/g' | sed -e 's/$/:/')

arguments=$(getopt --options a --longoptions "${longoptions}" --name '01_quantify_rnaseqc_null' -- "$@")
eval set -- "${arguments}"

stranded=${stranded-}

while true; do
    case "${1}" in
        --output_dir )
            output_dir="${2}"; check_for_directory "${1}" "${2}"; shift 2 ;;
        --sample_id )
            sample_id="${2}"; shift 2 ;;
        --bam_dir )
            bam_dir="${2}"; check_for_directory "${1}" "${2}"; shift 2 ;;
        --genes_gtf )
            genes_gtf="${2}"; check_for_file "${1}" "${2}"; shift 2 ;;
        --bam_file_end )
            bam_file_end="${2}"; shift 2 ;;
        --stranded )
            stranded="${2}"; shift 2 ;;
        --)
            shift; break;;
        * )
            echo "Invalid argument ${1} ${2}" >&2
            exit 1
    esac
done

: "${output_dir:?need --output_dir}"
: "${sample_id:?need --sample_id}"
: "${bam_dir:?need --bam_dir}"
: "${genes_gtf:?need --genes_gtf}"
: "${bam_file_end:?need --bam_file_end}"

bam_file="${bam_dir}/${sample_id}.${bam_file_end}"
check_for_file "--bam" "${bam_file}"

out_dir="${output_dir}/rnaseqc_null"
mkdir -p "${out_dir}"

rnaseqc_cmd=(
    rnaseqc
    "${genes_gtf}"
    "${bam_file}"
    "${out_dir}"
    -s "${sample_id}"
    -vv
)
if [[ -n "${stranded}" ]]; then
    rnaseqc_cmd+=(--stranded "${stranded}")
fi

"${rnaseqc_cmd[@]}"

gzip -f \
    "${out_dir}/${sample_id}.exon_reads.gct" \
    "${out_dir}/${sample_id}.gene_tpm.gct" \
    "${out_dir}/${sample_id}.gene_reads.gct"
