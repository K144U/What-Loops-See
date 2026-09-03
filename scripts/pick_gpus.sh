#!/bin/bash
# Choose N distinct GPUs, least busy first, and export them as a list.
#
#   N_GPUS=4 source scripts/pick_gpus.sh
#   echo "$LOOPVISION_GPUS"      # e.g. "2 5 1 7"
#
# The plural sibling of pick_gpu.sh, for multirun.pbs. Same policy: free
# memory is a hard floor, utilisation is what we minimise, because an 80 GB
# A100 can have 60 GB free and still be compute saturated.
#
# Why this exists at all. On 2026-09-03 the GPU node had 94 of 96 cores
# allocated with GPUs 2 and 5 sitting at 0 percent utilisation. The cards
# were idle because there were no cores left to feed them, not because
# nobody wanted them. Submitting more PBS jobs cannot fix that; running
# more training processes inside a core allocation we already hold can.
# See docs/decisions.md D-025.

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "pick_gpus: nvidia-smi not found" >&2
    return 1 2>/dev/null || exit 1
fi

_n="${N_GPUS:-2}"
_min_free="${LOOPVISION_MIN_FREE_MIB:-10240}"

_eligible=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader,nounits \
            | awk -F', *' -v min="$_min_free" '$2 >= min' \
            | sort -t, -k3,3n)

_avail=$(echo "$_eligible" | grep -c . )
if [ "$_avail" -lt "$_n" ]; then
    echo "pick_gpus: only $_avail card(s) have ${_min_free} MiB free, need $_n" >&2
    if [ "$_avail" -eq 0 ]; then
        return 1 2>/dev/null || exit 1
    fi
    _n="$_avail"
fi

export LOOPVISION_GPUS=$(echo "$_eligible" | head -n "$_n" | cut -d, -f1 | tr -d ' ' | tr '\n' ' ' | sed 's/ $//')
export LOOPVISION_NGPUS="$_n"
echo "pick_gpus: selected $_n card(s): $LOOPVISION_GPUS"
echo "$_eligible" | head -n "$_n" | while IFS=, read -r i f u; do
    echo "  GPU$(echo $i|tr -d ' '): $(echo $u|tr -d ' ')% util, $(echo $f|tr -d ' ') MiB free"
done

unset _n _min_free _eligible _avail
