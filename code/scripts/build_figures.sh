#!/bin/bash
# Every figure the draft cites, drawn from inputs this stores first. D-074.
#
#   scripts/build_figures.sh runs/_figures_20261022
#
# WHY THIS EXISTS. Each plot script reads a summary that a tested module
# writes, which keeps statistics out of the plots, but nothing kept the
# summary. On 2026-10-22 F2 no longer matched the runs: it had been drawn
# from a pooled summary of 58 runs that existed nowhere, and the pool had
# grown to 85. A figure is checkable only if what it was drawn from is on
# disk beside it, so this writes every input into DIR before it draws, with
# the commit that made it, and refuses to write over a DIR that exists.
#
# Light work: no training, no checkpoint loaded, about two minutes on the
# login node. Run it from the repository root with the venv active.
#
# F5 and F10 are not drawn. They belong to the H4 work the draft leaves to
# concurrent work in submission.

set -euo pipefail

DIR=${1:?usage: scripts/build_figures.sh runs/_figures_YYYYMMDD}
if [ -e "$DIR" ]; then
    echo "refusing: $DIR exists, and a stored input is never overwritten"
    exit 2
fi
if [ -n "$(git status --porcelain -- src scripts configs)" ]; then
    echo "refusing: src, scripts or configs differ from the commit, so the"
    echo "commit stored beside the inputs would not be the code that made them"
    exit 2
fi
mkdir -p "$DIR"
git rev-parse HEAD > "$DIR/cluster_commit.txt"

# The H1 arms, by the config files that trained them. D-074.
H1_CONFIGS=(--config configs/m3/famB_curve_111.yaml
            --config configs/m3/famC_breadth_111.yaml)
# The halting summary holds family B and family C; F6 draws family B.
F6_TASK='B d(1, 2, 3, 4, 5, 6) b(12, 13, 14, 15, 16)'
# The coda lens of the depth 6 trajectories, findings.md "models do not walk
# the chain hop by hop".
CODA_RUN=runs/famB_curve111_s0

python -m loopvision.analysis.escape_rates --json "$DIR/rates.json" \
    > "$DIR/escape_rates.txt"
python -m loopvision.analysis.m1_curves --no-fit "${H1_CONFIGS[@]}" \
    --summary "$DIR/m1_summary.json" --out "$DIR/m1_kmin.parquet" \
    > "$DIR/m1_curves.txt"
python -m loopvision.analysis.m5_h2 --summary "$DIR/m5_summary.json" \
    > "$DIR/m5_h2.txt"
cp "$CODA_RUN/coda_lens.json" "$DIR/coda_lens.json"

# Appendix B's patching numbers, which until D-076 were quoted from a run of
# 2026-09-10 that nothing kept. The module exits 1 on a split or falsified
# verdict and Python exits 1 on a crash, so the verdict line is what counts.
set +e
python -m loopvision.analysis.h3_contrast > "$DIR/h3_contrast.txt" 2>&1
set -e
if ! grep -qE "^H3 (SUPPORTED|FALSIFIED|SPLIT)" "$DIR/h3_contrast.txt"; then
    echo "h3_contrast printed no verdict: see $DIR/h3_contrast.txt"
    exit 1
fi

python -m loopvision.analysis.plots.f1_tasks --out figures/f1_tasks.png
python -m loopvision.analysis.plots.f2_curves --summary "$DIR/m1_summary.json"
python -m loopvision.analysis.plots.f3_patching
python -m loopvision.analysis.plots.f4_coda --lens "$DIR/coda_lens.json" \
    --out figures/f4_coda.png
python -m loopvision.analysis.plots.f6_halting --summary "$DIR/m5_summary.json" \
    --task "$F6_TASK"
python -m loopvision.analysis.plots.f8_absent
python -m loopvision.analysis.plots.f9_binding --rates "$DIR/rates.json"
# Added 2026-11-12 (D-079): every seed's training curve for four arms, and a
# schematic that reads no data. The trap band is read from its registration.
python -m loopvision.analysis.plots.f11_training --runs runs --rates "$DIR/rates.json" \
    --registered runs/_revision_verdicts_20261021/registered.txt \
    --out figures/f11_training
python -m loopvision.analysis.plots.f12_schematic --out figures/f12_schematic
# Added 2026-11-12 (D-080): the illustrations. F13 and F16 read only the
# generator and the group's arithmetic; F14 and F15 read the stored outputs
# named here, which the draft's tables already quote.
python -m loopvision.analysis.plots.f13_anatomy --out figures/f13_anatomy
python -m loopvision.analysis.plots.f14_renderings \
    --salience runs/_tile_distinguishability_20261022_d076/d070.json \
    --out figures/f14_renderings
python -m loopvision.analysis.plots.f15_rescue \
    --timing runs/_revision_verdicts_20261021/timing.json \
    --out figures/f15_rescue
python -m loopvision.analysis.plots.f16_trap --out figures/f16_trap

sha256sum figures/f1_tasks.png figures/f2_curves.png figures/f3_patching.png \
    figures/f4_coda.png figures/f6_halting.png figures/f8_absent.png \
    figures/f9_binding.png figures/f11_training.png figures/f12_schematic.png \
    figures/f13_anatomy.png figures/f14_renderings.png figures/f15_rescue.png \
    figures/f16_trap.png \
    > "$DIR/figures.sha256"
echo "stored inputs and figure hashes in $DIR"
