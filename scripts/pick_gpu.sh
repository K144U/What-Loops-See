#!/bin/bash
# Select the freest GPU and export CUDA_VISIBLE_DEVICES.
#
# PBS on the target cluster does not track ngpus as a consumable resource,
# so CUDA_VISIBLE_DEVICES is never set for us and two jobs will happily
# land on the same card and fight over memory. Every job sources this.
#
# Source it, do not execute it. It must export into the calling shell:
#
#   source scripts/pick_gpu.sh
#
# The Python side re-checks the selected device and fails loudly if it
# already has more than 2 GB in use. This script picking a card is not the
# same as that card being free by the time the process starts, and only the
# second check catches the race.

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "pick_gpu: nvidia-smi not found, leaving CUDA_VISIBLE_DEVICES unset" >&2
    return 0 2>/dev/null || exit 0
fi

# memory.used first, utilisation as the tiebreak. A card with low memory
# but pinned utilisation is running someone's job and we should not join it.
_best=$(nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | sort -t, -k2,2n -k3,3n \
    | head -n 1)

if [ -z "$_best" ]; then
    echo "pick_gpu: nvidia-smi returned nothing, leaving CUDA_VISIBLE_DEVICES unset" >&2
    return 0 2>/dev/null || exit 0
fi

_idx=$(echo "$_best"  | cut -d, -f1 | tr -d ' ')
_mem=$(echo "$_best"  | cut -d, -f2 | tr -d ' ')
_util=$(echo "$_best" | cut -d, -f3 | tr -d ' ')

export CUDA_VISIBLE_DEVICES="$_idx"
echo "pick_gpu: selected GPU $_idx (${_mem} MiB used, ${_util}% util)"

if [ "$_mem" -gt 2048 ]; then
    echo "pick_gpu: WARNING every GPU has more than 2 GB in use." >&2
    echo "pick_gpu: the freest was $_idx at ${_mem} MiB. The training process" >&2
    echo "pick_gpu: will refuse to start rather than sharing a card." >&2
fi

unset _best _idx _mem _util
