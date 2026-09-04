#!/usr/bin/env bash
# Report newly finished cluster runs, once each.
#
# Runs on the LOCAL machine, not the cluster, because the cluster is only
# reachable through the VPN on that machine. A cloud scheduled agent cannot
# see <cluster-ip> at all.
#
# Called by the scheduled task "loopvision-run-watch". Each scheduled run
# starts a fresh session with no memory, so "have I already reported this"
# lives in a state file rather than in context. Without that the task would
# re-announce every finished run every time it fired.
#
# Prints nothing when nothing has changed. That is the normal case and it
# is what lets the task be quiet most of the time.
#
#   scripts/local_run_watch.sh [--reset] [--all]
#
#     --reset  forget what has been reported, then report current state
#     --all    print the full status table regardless of changes

set +e

STATE="${LOOPVISION_WATCH_STATE:-$HOME/.claude/loopvision-watch-state.txt}"
RUNS="famB_curve111_s1 famB_curve111_s2
      famA_d2_d4only_s0 famA_d2_d4only_s1
      famA_d2_s3only_s0 famA_d2_s3only_s1
      famC_breadth111_s0 famC_breadth111_s1 famC_breadth111_s2"

case " $* " in *" --reset "*) rm -f "$STATE";; esac
mkdir -p "$(dirname "$STATE")" 2>/dev/null
touch "$STATE" 2>/dev/null

remote=$(ssh -o BatchMode=yes -o ConnectTimeout=30 <login-host> "cd ~/loopvision && for r in $RUNS; do
    if [ -f runs/\$r/DONE ]; then echo \"\$r DONE\";
    elif [ -f runs/\$r/metrics.parquet ]; then echo \"\$r live\";
    else echo \"\$r pending\"; fi
  done
  echo \"QUEUE \$(qstat -u <cluster-user> 2>/dev/null | awk '/^[0-9]/ {split(\$1,a,\".\"); printf \"%s \", a[1]}')\"" 2>/dev/null)

if [ -z "$remote" ]; then
  # Unreachable is not the same as nothing happening, but it is also not
  # worth waking anyone for: the VPN and the login node have both dropped
  # repeatedly and the jobs are unaffected either way.
  echo "UNREACHABLE"
  exit 0
fi

changed=0
for r in $RUNS; do
  line=$(echo "$remote" | grep "^$r ")
  state=${line#"$r "}
  key="$r=$state"
  case "$state" in
    DONE|live) ;;
    *) continue;;
  esac
  if ! grep -qxF "$key" "$STATE" 2>/dev/null; then
    # Only announce DONE and the first sighting of a run starting.
    if [ "$state" = "DONE" ]; then
      echo "FINISHED: $r has completed all 300k steps"
      changed=1
    elif ! grep -q "^$r=" "$STATE" 2>/dev/null; then
      echo "STARTED: $r is now training"
      changed=1
    fi
    grep -v "^$r=" "$STATE" > "$STATE.tmp" 2>/dev/null
    mv "$STATE.tmp" "$STATE" 2>/dev/null
    echo "$key" >> "$STATE"
  fi
done

case " $* " in
  *" --all "*)
    echo "$remote" | grep -v '^QUEUE'
    echo "$remote" | grep '^QUEUE'
    ;;
esac

if [ "$changed" = "0" ]; then
  echo "NOCHANGE"
fi
