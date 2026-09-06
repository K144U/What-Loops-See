#!/bin/bash
# Choose N distinct GPUs, least busy first, claim them, and export the list.
#
#   N_GPUS=4 source scripts/pick_gpus.sh
#   echo "$LOOPVISION_GPUS"      # e.g. "2 5 1 7"
#   release_gpu_claims           # call from a trap when the job exits
#
# The plural sibling of pick_gpu.sh, for multirun.pbs. Free memory is a
# hard floor, utilisation is what we minimise, because an 80 GB A100 can
# have 60 GB free and still be compute saturated.
#
# Why this exists at all. On 2026-09-03 the GPU node had 94 of 96 cores
# allocated with GPUs 2 and 5 sitting at 0 percent utilisation. The cards
# were idle because there were no cores left to feed them, not because
# nobody wanted them. See docs/decisions.md D-025.
#
# WHY THE CLAIM FILES. Ranking by utilisation is right and is not enough.
# Two jobs submitted seconds apart each run this independently, see the
# same snapshot, and pick the same "freest" card. Worse, a job that has
# just started still reads 0 percent until it ramps, so the second job sees
# the first one's card as idle. Measured on 2026-09-04: jobs 5180 and 5181
# both reported 58286 MiB free, the same card, while GPUs 2, 4 and 6 sat at
# 0 percent untouched.
#
# A claim is an atomic mkdir, which is the one primitive that cannot race.
# The job id goes inside so a stale claim can be recognised later.

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "pick_gpus: nvidia-smi not found" >&2
    return 1 2>/dev/null || exit 1
fi

_claims="${LOOPVISION_CLAIM_DIR:-$HOME/.loopvision-gpu-claims}"
_me="${PBS_JOBID:-shell$$}"
mkdir -p "$_claims" 2>/dev/null

# Sweep stale claims before taking any.
#
# This is the part that has to be right. A claim wrongly kept blacklists a
# good card forever, which is worse than the collision this fixes. A claim
# wrongly removed lets two jobs share a card, which is only as bad as the
# old behaviour. So the sweep errs toward removing: either qstat saying the
# owner is gone, or an age past the longest possible walltime, is enough.
for _c in "$_claims"/*; do
    [ -d "$_c" ] || continue
    _owner=$(cat "$_c/jobid" 2>/dev/null)
    _stale=0
    if [ -z "$_owner" ]; then
        _stale=1
    elif command -v qstat >/dev/null 2>&1; then
        qstat "$_owner" >/dev/null 2>&1 || _stale=1
    fi
    # Queue walltime maxes at 24 hours, so anything older cannot be live,
    # and this is the fallback when qstat itself is unreachable.
    if [ -n "$(find "$_c" -maxdepth 0 -mmin +1500 2>/dev/null)" ]; then
        _stale=1
    fi
    if [ "$_stale" = "1" ]; then
        rm -rf "$_c" 2>/dev/null
        echo "pick_gpus: cleared stale claim on GPU$(basename "$_c") (owner ${_owner:-unknown})"
    fi
done

_n="${N_GPUS:-2}"
_min_free="${LOOPVISION_MIN_FREE_MIB:-10240}"

_eligible=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader,nounits \
            | awk -F', *' -v min="$_min_free" '$2 >= min' \
            | sort -t, -k3,3n)

if [ -z "$(echo "$_eligible" | grep -c . )" ] || [ "$(echo "$_eligible" | grep -c . )" = "0" ]; then
    echo "pick_gpus: no card has ${_min_free} MiB free" >&2
    return 1 2>/dev/null || exit 1
fi

# First pass: take only unclaimed cards, least busy first.
_picked=""
_count=0
while IFS=, read -r _i _f _u; do
    [ "$_count" -ge "$_n" ] && break
    _i=$(echo "$_i" | tr -d ' ')
    if mkdir "$_claims/$_i" 2>/dev/null; then
        echo "$_me" > "$_claims/$_i/jobid"
        _picked="$_picked $_i"
        _count=$((_count + 1))
        echo "  GPU$_i: $(echo $_u|tr -d ' ')% util, $(echo $_f|tr -d ' ') MiB free, claimed"
    fi
done <<EOF
$_eligible
EOF

# Second pass, only if the first could not fill the request. Running on a
# shared card beats not running, so a shortage degrades rather than fails,
# and says so.
if [ "$_count" -lt "$_n" ]; then
    echo "pick_gpus: only $_count unclaimed card(s) for $_n run(s), sharing the rest" >&2
    while IFS=, read -r _i _f _u; do
        [ "$_count" -ge "$_n" ] && break
        _i=$(echo "$_i" | tr -d ' ')
        case " $_picked " in *" $_i "*) continue;; esac
        _picked="$_picked $_i"
        _count=$((_count + 1))
        echo "  GPU$_i: $(echo $_u|tr -d ' ')% util, shared, not claimed"
    done <<EOF
$_eligible
EOF
fi

export LOOPVISION_GPUS=$(echo "$_picked" | sed 's/^ *//;s/ *$//')
export LOOPVISION_NGPUS="$_count"
export LOOPVISION_CLAIM_DIR="$_claims"
# Captured now rather than re-read later. release_gpu_claims runs from a
# trap, and a trap can fire in a context where PBS_JOBID is no longer set,
# which would silently release nothing and leave every claim behind.
export LOOPVISION_CLAIM_OWNER="$_me"
echo "pick_gpus: selected $_count card(s): $LOOPVISION_GPUS"

release_gpu_claims() {
    # Called from a trap. Only removes claims this job owns, so a crash
    # here can never free somebody else's card.
    local d owner
    [ -n "$LOOPVISION_CLAIM_OWNER" ] || return 0
    for d in "${LOOPVISION_CLAIM_DIR:-$HOME/.loopvision-gpu-claims}"/*; do
        [ -d "$d" ] || continue
        owner=$(cat "$d/jobid" 2>/dev/null)
        if [ "$owner" = "$LOOPVISION_CLAIM_OWNER" ]; then
            rm -rf "$d" 2>/dev/null
        fi
    done
}

unset _n _min_free _eligible _picked _count _i _f _u _c _owner _stale _claims _me
