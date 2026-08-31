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

# Selection is on memory FREE, not memory used, with utilisation as the
# tiebreak. A card with low memory but pinned utilisation is running
# someone's job and we should not join it.
#
# The threshold is free memory, not used memory, and that is a deliberate
# change from IMPLEMENTATION.md Section 3, which said to fail if the
# selected device has more than 2 GB in use. On this cluster the cards are
# 80 GB A100s shared by roughly forty concurrent jobs, and at the time of
# writing every single card had more than 2 GB in use while six of the
# eight had over 70 GB free. A used-memory threshold would have refused to
# start on a completely usable machine. See docs/decisions.md D-013.
#
# Override for a larger model: LOOPVISION_MIN_FREE_MIB=40960 source ...
_min_free="${LOOPVISION_MIN_FREE_MIB:-20480}"

_best=$(nvidia-smi \
    --query-gpu=index,memory.free,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | sort -t, -k2,2nr -k4,4n \
    | head -n 1)

if [ -z "$_best" ]; then
    echo "pick_gpu: nvidia-smi returned nothing, leaving CUDA_VISIBLE_DEVICES unset" >&2
    return 0 2>/dev/null || exit 0
fi

_idx=$(echo "$_best"  | cut -d, -f1 | tr -d ' ')
_free=$(echo "$_best" | cut -d, -f2 | tr -d ' ')
_used=$(echo "$_best" | cut -d, -f3 | tr -d ' ')
_util=$(echo "$_best" | cut -d, -f4 | tr -d ' ')

export CUDA_VISIBLE_DEVICES="$_idx"
export LOOPVISION_GPU_FREE_MIB="$_free"
echo "pick_gpu: selected GPU $_idx (${_free} MiB free, ${_used} MiB used, ${_util}% util)"

if [ "$_free" -lt "$_min_free" ]; then
    echo "pick_gpu: FATAL the freest GPU ($_idx) has only ${_free} MiB free," >&2
    echo "pick_gpu: below the ${_min_free} MiB floor. Every card is loaded." >&2
    echo "pick_gpu: Refusing rather than starting a run that will die on OOM" >&2
    echo "pick_gpu: hours later, or quietly evict someone else's job." >&2
    unset _best _idx _free _used _util _min_free
    return 1 2>/dev/null || exit 1
fi

unset _best _idx _free _used _util _min_free
