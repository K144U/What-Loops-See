#!/bin/bash
# Select a GPU and export CUDA_VISIBLE_DEVICES.
#
# PBS on this cluster does not track ngpus as a consumable resource, so
# CUDA_VISIBLE_DEVICES is never set for us and jobs will otherwise pile
# onto whichever card they happen to choose. Every job sources this.
#
# Source it, do not execute it. It must export into the calling shell:
#
#   source scripts/pick_gpu.sh
#
# SELECTION POLICY, and why it changed.
#
# The first version ranked by most free memory. That is the wrong
# objective and it cost real time: on 2026-09-03 both 1M-step runs were
# found sharing GPU 2 at 100 percent utilisation with three other
# processes, while GPUs 1, 5 and 7 sat at 0 to 8 percent. Free memory is a
# poor proxy for contention, because an 80 GB A100 can have 60 GB free and
# still be compute saturated.
#
# Memory is a hard constraint, so it stays as a floor. Utilisation is the
# thing to actually minimise. Among cards that clear the floor we pick the
# least busy, breaking near-ties at random so that two jobs starting at the
# same moment do not both choose the same card, which is exactly what
# happened before. See docs/decisions.md D-024.

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "pick_gpu: nvidia-smi not found, leaving CUDA_VISIBLE_DEVICES unset" >&2
    return 0 2>/dev/null || exit 0
fi

_min_free="${LOOPVISION_MIN_FREE_MIB:-20480}"
# Cards within this many utilisation points of the best are treated as
# equally good, and one is chosen at random to spread concurrent jobs.
_tol="${LOOPVISION_UTIL_TOLERANCE:-15}"

_rows=$(nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)

if [ -z "$_rows" ]; then
    echo "pick_gpu: nvidia-smi returned nothing, leaving CUDA_VISIBLE_DEVICES unset" >&2
    return 0 2>/dev/null || exit 0
fi

# Keep only cards with enough free memory, then sort by utilisation.
_eligible=$(echo "$_rows" | awk -F', *' -v min="$_min_free" '$2 >= min' | sort -t, -k4,4n)

if [ -z "$_eligible" ]; then
    _best_free=$(echo "$_rows" | sort -t, -k2,2nr | head -1 | cut -d, -f2 | tr -d ' ')
    echo "pick_gpu: FATAL no GPU has ${_min_free} MiB free. Freest has ${_best_free} MiB." >&2
    echo "pick_gpu: Refusing rather than starting a run that will die on OOM" >&2
    echo "pick_gpu: hours later, or quietly evict someone else's job." >&2
    unset _rows _eligible _min_free _tol
    return 1 2>/dev/null || exit 1
fi

_best_util=$(echo "$_eligible" | head -1 | cut -d, -f4 | tr -d ' ')
_pool=$(echo "$_eligible" | awk -F', *' -v b="$_best_util" -v t="$_tol" '$4 <= b + t')
_n=$(echo "$_pool" | wc -l)
_pick=$(( (RANDOM % _n) + 1 ))
_row=$(echo "$_pool" | sed -n "${_pick}p")

_idx=$(echo "$_row"  | cut -d, -f1 | tr -d ' ')
_free=$(echo "$_row" | cut -d, -f2 | tr -d ' ')
_used=$(echo "$_row" | cut -d, -f3 | tr -d ' ')
_util=$(echo "$_row" | cut -d, -f4 | tr -d ' ')

export CUDA_VISIBLE_DEVICES="$_idx"
export LOOPVISION_GPU_FREE_MIB="$_free"
echo "pick_gpu: selected GPU $_idx (${_util}% util, ${_free} MiB free, ${_used} MiB used)"
echo "pick_gpu: chose from $_n card(s) within ${_tol} points of the least busy (${_best_util}%)"

unset _rows _eligible _min_free _tol _best_util _pool _n _pick _row _idx _free _used _util
