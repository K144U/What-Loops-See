# Gap Analysis and Related Work Survey

**Date:** 2026-08-31
**Method:** direct retrieval of arXiv abstract pages, plus keyword search and one curated community list. Every paper is tagged with how it was checked.
**Purpose:** establish what is already published, which of our four hypotheses survive contact with it, and where the defensible novelty actually sits.

**Confidence tags used throughout:**

- **VERIFIED** means I fetched the arXiv abstract page and read title, authors, date and abstract.
- **LISTED** means the paper appeared in a curated list or a search snippet and I have not fetched the abstract. Do not cite a LISTED paper without verifying it first.

**Headline:** the field moved a long way between the spec's reference list and today. Twenty one directly relevant papers exist that the spec does not cite. Two of our four hypotheses are substantially weakened, one is largely intact, and the framing of the paper's most practical contribution needs to change. The core novelty survives, but it is narrower and more specific than the spec assumes, and it now has to be argued for rather than assumed.

---

## 1. Audit of the spec's reference list

Every reference I checked from Section 15 of the implementation spec exists and is described accurately, with one framing problem. This is worth stating plainly because the arXiv IDs dated 2602, 2604 and 2607 fall past my training data, and a plausible worry was that some were wrong.

| Cited as | Actual title | Authors | Date | Status |
|---|---|---|---|---|
| ELT, 2604.09168 | Elastic Looped Transformers for Visual Generation | Goyal, Agrawal, Anil, Jain, Paul, Kusupati | 2026-04-10, rev 2026-07-23 | VERIFIED, **framing problem, see below** |
| HIVE, 2602.05359 | Multimodal Latent Reasoning via Hierarchical Visual Cues Injection | Y. Zhang, Yan, Jiang, Han | 2026-02-05, rev 2026-05-08 | VERIFIED |
| RecursiveVLM, 2602.09080 | Looping Back to Move Forward: Recursive Transformers for Efficient and Flexible Large Multimodal Models | Xu, Gao, Wang, Li, Chen, Guo, Yang, S. Zhang | 2026-02-09 | VERIFIED |
| Lee and Park, 2607.00774 | Soft Mixture-of-Recursions: Going Deeper with Recursive Vision Transformers | Sang In Lee, Jihun Park | 2026-07-01 | VERIFIED |
| Ouro, 2510.25741 | Scaling Latent Reasoning via Looped Language Models | Rui-Jie Zhu and 32 others | 2025-10-29, rev 2026-07-01 | VERIFIED |
| Zhang et al., 2607.20594 | When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers | T. Zhang, Hu, Peng, Xie | 2026-07-22 | VERIFIED, **highest threat, see Section 4** |
| Pappone et al., 2509.23314 | Two-Scale Latent Dynamics for Recurrent-Depth Transformers | Pappone, Crisostomi, Rodola | 2025-09-27, rev 2025-11-13 | VERIFIED, **threat, see Section 4** |
| Lu et al., 2507.02199 | Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer | Lu, Yang, Lee, Li, Liu | 2025-07-02, rev 2025-09-28 | VERIFIED |

**The ELT framing problem.** The spec cites ELT under "looped vision" and treats it as part of the looped visual recognition line. It is a **visual generation** paper: FID 2.0 on ImageNet 256x256, FVD 72.8 on UCF-101. The spec separately declares the generation arm out of scope. Both things can be true, and the Intra-Loop Self Distillation borrowing for the stability fallback is legitimate, but the related work section must not imply ELT is doing recognition. A reviewer who knows ELT will notice.

**Remaining spec citations, verified 2026-08-31:**

| Cited as | Actual title | Authors | Date | Status |
|---|---|---|---|---|
| Huginn, 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach | Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein | 2025-02-07 | VERIFIED. 3.5B, 800B tokens, checkpoint `huginn-0125`. Matches the spec. |
| Mixture-of-Recursions, 2507.10524 | Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation | Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Yun | 2025-07-14 | VERIFIED |
| Xu and Sato, 2410.01405 | On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding | Kevin Xu, Issei Sato | 2024-10-02, rev 2025-06-05 | VERIFIED. The spec's "time-modulated looped transformers" is a fair description, not the title. Cite by the real title. |
| Fixed point diffusion, 2401.08741 | Fixed Point Diffusion Models | Xingjian Bai, Luke Melas-Kyriazi | 2024-01-16 | VERIFIED |
| Svete and Sabharwal, 2510.13117, "ICLR 2026" | **On the Reasoning Abilities of Masked Diffusion Language Models** | Anej Svete, Ashish Sabharwal | 2025-10-15, rev 2026-04-26 | VERIFIED but **twice misdescribed, see below** |

**The Svete and Sabharwal problem, resolved 2026-08-31.** The spec filed this under "Equivalences and theory" as looped-transformer work. It is a paper about masked diffusion language models, whose relevant result is an equivalence between masked diffusion models and polynomially-padded looped transformers. Citing it for that equivalence is legitimate. Citing it as a looped-transformer theory paper is not, and a reviewer in that subfield will notice. **Fixed in the spec at line 500**, which now names the paper and states which result to cite it for.

**The ICLR 2026 attribution is correct.** It flagged only because the arXiv page carries no venue field across all three versions (v1 2025-10-15, v2 2026-03-02, v3 2026-04-26). A domain-restricted search on iclr.cc confirms it as an ICLR 2026 oral. The OpenReview forum (id `BVnIsh4Nz1`) exists but sits behind a bot challenge, so the acceptance is confirmed by search rather than by the authoritative record. The presentation type, oral, is the least certain part and is worth one direct check before it appears in a bibliography.

**General lesson for this bibliography.** An absent arXiv venue field is not evidence of non-acceptance. Several papers in this survey are likely published without saying so on arXiv. Check the venue separately rather than inferring it from the abstract page, and do not silently downgrade a cited venue to "preprint" on arXiv's silence alone.

**Not re-checked**, being established, pre-cutoff and non-arXiv: Universal Transformers (Dehghani et al. ICLR 2019), Giannou et al. ICML 2023, Saunshi et al. ICLR 2025, deep equilibrium models (Bai, Kolter, Koltun NeurIPS 2019), multiscale DEQ (NeurIPS 2020), RAFT (Teed and Deng ECCV 2020). These are safe to cite from memory of the venue, but confirm page numbers and exact titles at bibliography time.

---

## 2. What the spec's reference list is missing

Twenty one papers, all directly relevant, none cited. Grouped by what they threaten.

### 2.1 Looped vision, the line we claim to be entering

| ID | Title | Date | Why it matters | Check |
|---|---|---|---|---|
| 2602.02156 | LoopViT: Scaling Visual ARC with Looped Transformers | 2026-02-02 | **Our nearest vision neighbour.** 18M parameters, weight-tied recurrence, ARC-AGI-1 at 65.8 percent beating 73M ensembles. Has a parameter-free Dynamic Exit on predictive entropy where the model halts when its state "crystallizes into a low-uncertainty attractor". Looped vision at our exact scale, with a halting rule, using attractor language. | VERIFIED |
| 2605.10661 | bViT: Investigating Single-Block Recurrence in Vision Transformers for Image Recognition | 2026-05-11 | 12-step bViT-B matches ViT-B on ImageNet-1K with an order of magnitude fewer parameters. Reports "implicit depth multiplexing", and mechanistic analyses showing the shared block changes effective behaviour across steps rather than repeating. Overlaps our M3 and M4 territory. | VERIFIED |
| 2608.04879 | Training Crossroads for Recurrent Vision Transformers: Recurrence, Neural ODEs, and Deep Supervision | 2026-08-05 | **Three weeks old.** Matched-FLOPs and matched-memory comparison of recurrent ViTs against standard ViTs on CIFAR-100. Finds standard ViTs preferable when FLOPs are the constraint. Directly relevant to gate G2, see Section 5. | VERIFIED |
| 2605.30215 | Deja View: Looping Transformers for Multi-View 3D Reconstruction | 2026-05 | Looped block applied K times to per-view features, K as an inference-time compute knob. Explicit looping beats independent per-step parameters. | VERIFIED |
| 2607.09061 | On Locality and Length Generalization in Visual Reasoning | 2026-07-10 | Vision models exploit global shortcuts and fail to generalise over task length or complexity. Recurrent vision policies with strictly local perception mitigate it. Closest published work to our shortcut concern. | VERIFIED |
| 2602.07845 | Recurrent-Depth VLA | 2026-02 | Tasks failing at 0 percent with one iteration exceed 90 percent with four. Adaptive stopping on latent convergence. **Complexity is defined operationally by success rate, not by ground truth.** See Section 3. | VERIFIED |
| 2605.09948 | LoopVLA: Learning Sufficiency in Recurrent Refinement for VLA Models | 2026-05 | Dual head with an explicit sufficiency head, estimating sufficiency from evolving representations "rather than heuristic early-exit criteria". | VERIFIED |
| 2607.28989 | SILVA Networks, structured implicit layers and vector attractors | 2026-07 | Implicit recurrent dynamics including vision. | VERIFIED |

### 2.2 Mechanism and interpretability of looped models

| ID | Title | Date | Why it matters | Check |
|---|---|---|---|---|
| 2512.19941 | Block-Recurrent Dynamics in Vision Transformers | 2025-12 | **Biggest surprise of this survey.** Finds "directional convergence into class-dependent angular basins with self-correcting trajectories", token-specific dynamics where cls reorients late while patch tokens cohere, and "collapse to low rank updates in late depth, consistent with convergence to low-dimensional attractors". Our rays-not-fixed-points secondary claim has a vision precedent. See Section 4. | VERIFIED |
| 2604.11791 | A Mechanistic Analysis of Looped Reasoning Language Models | 2026-04-13 | Blayney, Arroyo, Obando-Ceron, Castro, Courville, Bronstein, Dong. Each layer in the cycle converges to a distinct fixed point, giving a consistent cyclic trajectory. Studies how block size, **input injection** and normalisation affect fixed point emergence. Overlaps our injection ablation. Strong author list, well resourced. | VERIFIED |
| 2605.17811 | One Model, Two Roles: Emergent Specialization in a Shared Recurrent Transformer | 2026-05 | Asymmetric state injection induces distinct functional roles in recurrent blocks. | VERIFIED |
| 2604.09870 | Relational Preference Encoding in Looped Transformer Internal States | 2026-04-10, v2 2026-07-16 | Jan Kirin. **Do not cite without the erratum.** v2 carries a prepended erratum: a post-publication audit found the three headline results inflated by two independent evaluation errors, a canonical-ordering artifact and train-test leakage. The 95.2 percent pairwise evaluator accuracy becomes 63.9 percent antisymmetrised on the full 8,552-pair test set, and the central effect drops to 2.3 percentage points. The author notes the two errors were "mutually invisible", each hiding the other. | VERIFIED |

### 2.3 Halting, early exit and adaptive depth

This cluster is dense and it is where H2 lives.

| ID | Title | Date | Why it matters | Check |
|---|---|---|---|---|
| 2607.20519 | Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts | 2026-07-08 | Popescu, Saez de Ocariz Borde, Lio. Finds simple post-hoc confidence readouts often match or beat learned gates, and that the problem is the trajectory induced by joint gate training rather than gate expressivity. Evaluated on modular arithmetic, binary parity, and Ouro 1.4B and 2.6B. **Direct competitor to H2's methodology.** | VERIFIED |
| 2606.29983 | Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping | 2026-06-29 | Kuo, Chayti, Reizinger, Brendel, Jaggi. "Introducing stochasticity into the number of loops during training sharply reduces OOD variance." Traces instability to a spurious correlation between sequence length and loop count. **This is H4, already published, in text.** | VERIFIED |
| 2602.08864 | Understanding Dynamic Compute Allocation in Recurrent Transformers | 2026-02 | ANIRA framework. Compute allocation aligns with task difficulty without supervision, yet models still fail to extrapolate to unseen input sizes despite allocating more computation. Distinguishes early structural cues from online halting. | VERIFIED |
| 2606.18206 | Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers | 2026-06 | Fixed-point convergence as an end-to-end halting signal. | VERIFIED |
| 2608.18222 | Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth | 2026-08-18 | Viakhirev, Borodin, Almutairi, Barannikov, Abramov, Mkrtchian. **More relevant than the list entry suggested.** The dynamical regime of the trained operator, settling, marginal or drifting, determines whether extra test-time iterations improve, hold or degrade performance. That is a mechanism-level predictor of extrapolation, adjacent to H2. | VERIFIED |
| 2608.26556 | Dynamical phase selection controls compute scaling in looped transformers | 2026-08-27 | Gunn Kim. **Four days old.** Identical architectures trained to identical accuracy land in different dynamical phases depending on initialisation, and the phase sets how inference cost scales. Bears on our initialisation choices, in particular the `1/sqrt(2 * core_depth * k_mean)` output projection scaling and the randn versus zeros state init ablation. | VERIFIED |
| 2604.21999 | Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning | 2026-04 | Learned memory tokens necessary for nontrivial adaptive-depth reasoning. | VERIFIED |
| 2607.10110 | Looped State-Space Language Models with Adaptive Exit-State Selection | 2026-07 | Adaptive inference by exit-state selection. | VERIFIED |
| 2608.24136 | Steering Recurrent Reasoners at Inference Time with Readout Feedback | 2026-08 | Intermediate predictions as steering forces. | VERIFIED |

### 2.4 Scaling and theory

| ID | Title | Date | Why it matters | Check |
|---|---|---|---|---|
| 2604.07822 | Loop, Think, and Generalize: Implicit Reasoning in Recurrent-Depth Transformers | 2026-04 | Kohli, Parthasarathy, Sun, Yao. Generalisation beyond training depth unlocked by scaling inference-time recurrence, models trained on 5-hop extrapolating to 10-hop. Identifies "overthinking": excessive recurrence degrades predictions. **Closest published work to H1, in text.** | VERIFIED |
| 2604.21106 | How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models | 2026-04 | Schwethelm, Rueckert, Kaissis. Recurrence-equivalence exponent phi = 0.46, so one loop is worth well under one layer. A 410M looped model matches a 580M non-looped model but costs the training compute of a 1B one. | VERIFIED |
| 2605.20670 | LT2: Linear-Time Looped Transformers | 2026-05 | | VERIFIED |
| 2605.23872 | Training-Free Looped Transformers | 2026-05 | | VERIFIED |
| 2606.18208 | Looped World Models | 2026-06 | | VERIFIED |
| 2605.30523 | Revisiting Padded Transformer Expressivity: Which Architectural Choices Matter and Which Don't | 2026-05-28 | Svete, Merrill, Cotterell, Sabharwal. Padded transformers keep the same circuit-complexity equivalences across architectural variations, with numeric precision and depth the primary determinants. Same group as 2510.13117. | VERIFIED |

---

## 3. The one thing nobody has

Across every paper above that relates loop count to task difficulty, difficulty is **operationally defined, never known**.

- Recurrent-Depth VLA (2602.07845) defines complexity by which tasks succeed at one iteration versus four. Complexity is read off the success rate it is supposed to explain.
- Loop, Think and Generalize (2604.07822) uses hop count in text, which is genuine ground truth, but only along one axis and only in language.
- When Does Recurrence Become an Algorithm (2607.20594) uses group word problem length n, again genuine, again text, again one axis.
- LoopViT (2602.02156) uses ARC-AGI, where nobody knows the required composition depth of any individual puzzle.
- HIVE, RecursiveVLM and SoftMoR report benchmark accuracy and do not ask the question at all.

**This is the gap, and it is a real one.** In pixels, with programmatic generation, we know the composition depth exactly and we can vary it independently of scene breadth. Nobody currently can. Every claim about what loops buy in vision is presently made against a proxy for difficulty rather than against difficulty itself.

That single sentence is the paper. It is narrower than "we discover loops track depth", and it is more defensible, because it is a claim about measurement rather than about a phenomenon that three other groups are also finding.

---

## 4. Threat assessment, hypothesis by hypothesis

### H1, loops track depth and are flat in breadth: **WEAKENED, still viable**

The depth half is close to known. 2604.07822 shows inference-time recurrence unlocks deeper reasoning and 2607.20594 gives a quantitative budget law, v ~ n_train/T_train with exponent 0.98 plus or minus 0.04 and R squared 0.99, plus a derived halting rule T* = ceil(n / v_hat). In text, "more loops for longer problems" is established with a scaling law attached.

**What survives:** the breadth half. Nobody has crossed composition depth against scene breadth with both known exactly, and nobody has done either in pixels. The dissociation is the contribution, not the depth trend.

**Consequence for the paper.** H1 cannot be presented as discovering that loops track depth. It has to be presented as a dissociation: loops track depth *and not breadth*, in a setting where the two are independently controlled for the first time. Family C stops being merely the null arm and becomes the load-bearing arm, since the breadth axis is the genuinely unexplored one. Any temptation to cut Family C should now be treated as fatal rather than merely costly.

### H2, convergence diagnostics do not predict extrapolation: **SEVERELY THREATENED as stated**

This is the bad news, and it is worth being blunt about it.

- Pappone et al. (2509.23314), September 2025, already built an early-exit rule on second-order differences in step size and showed it **outperforms KL-divergence exit strategies**. The claim that KL exit is suboptimal is published.
- 2607.20594 states that standard instruments "provably saturate at the fixed points trained loops converge to", which is our argument, and then goes further by introducing convergence-time scaling and showing that in-distribution head measurements predict out-of-distribution fate "where tail metrics do not". They have both the negative result and a replacement.
- 2607.20519 evaluates learned halting gates against post-hoc confidence readouts and locates the failure in the trajectory rather than the gate.
- 2602.08864 reports extrapolation failure despite correct-looking compute allocation.

H2 as written in the spec, that update-norm and KL diagnostics fail to predict extrapolation success, is at serious risk of being a replication in a new modality rather than a finding.

**What survives, and it is not nothing.** Every one of those papers evaluates a diagnostic against either downstream accuracy or a search over k. None of them can evaluate a diagnostic against **the number of loops the task actually required**, because none of them knows it. We can. The reframed claim is:

> Given ground-truth required depth, we can measure how far each halting criterion lands from where the computation actually finishes, and in which direction it errs. This is an oracle comparison that the text literature cannot construct.

That is a measurement contribution rather than a discovery contribution, and it is honest. It also makes contribution 4 in `paper.md` depend on the benchmark, which strengthens contribution 1 at the same time.

**Action required:** rewrite the H2 statement in `paper.md` Section 4 and in the pre-registration before it is frozen. Do not freeze the current wording.

### H3, patching separates by task family: **LARGELY INTACT, now the strongest hypothesis**

Closest prior work is the damage cones of 2607.20594, which is a causal instrument whose slope reproduces their v, and 2604.11791, which finds per-layer fixed points and cyclic trajectories in looped language models. Both are text.

Nobody has run activation patching over a (loop iteration, spatial patch position) grid in a looped vision model. The spatial axis has no text analogue, which is exactly why the instrument is new. The prediction that depth tasks are loop-localised and spatially distributed while breadth tasks are the reverse is untested by anyone.

**H3 is now our best hypothesis and F3 is our most defensible figure.** Consider promoting it in the contribution ordering. The spec ranks the loop-count curves first and patching third. The survey suggests the reverse ranking is more defensible.

### H4, loop-count jitter reduces wrong-attractor landings: **SCOOPED as a phenomenon**

Kuo et al. (2606.29983), June 2026, introduce stochasticity into the number of training loops and report that it sharply reduces out-of-distribution variance, tracing the instability to a spurious correlation between sequence length and loop count. That is the same intervention with the same direction of effect.

They do not use attractor framing or count wrong-attractor landings, and they are in text on algorithmic tasks. So a vision replication with the attractor analysis is still reportable, but it is a paragraph, not a contribution.

**Fortunate accident:** the spec already labels H4 exploratory. Keep it there, cite Kuo et al. as prior work rather than presenting jitter as our idea, and keep the width-zero arm because the comparison is still worth having.

### Secondary claim, rays not fixed points: **PARTIALLY PRE-EMPTED in vision**

The spec calls this the interesting secondary claim that comes free from M4, and states that if the direction-versus-norm geometry replicates in vision then every update-norm early exit is halting at the wrong place.

Block-Recurrent Dynamics in Vision Transformers (2512.19941) already reports directional convergence into class-dependent angular basins, token-specific dynamics, and low-rank collapse consistent with convergence to low-dimensional attractors, in vision transformers. Pappone et al. already report shrinking, increasingly orthogonal steps in text.

**What survives:** 2512.19941 studies standard ViTs re-expressed as block-recurrent, not models trained as looped from the start, and it does not make the halting argument. It also does not appear to isolate raw norm growth from direction-normalised quantities, which is the specific comparison our M4 is built around. So there is room, but the claim must be positioned as extending a known geometry to trained looped models and drawing the halting consequence, not as discovering the geometry.

---

## 5. A new risk to gate G2

Training Crossroads for Recurrent Vision Transformers (2608.04879, three weeks old) compares recurrent ViTs against standard ViTs at matched FLOPs and matched parameter memory on CIFAR-100, and concludes that **standard ViTs remain preferable when FLOPs are the primary constraint**, with recurrence winning only on the accuracy-per-parameter trade-off under memory constraints.

Gate G2 requires our looped model to beat a matched-compute feedforward baseline on depth 3. On classification, a published result says that comparison goes the other way.

Our tasks are compositional rather than classification, which is precisely the regime where looping should win, so this is not a prediction that G2 fails. It is a warning that G2 is a genuinely uncertain gate rather than a formality, and that if it fails we should not assume the implementation is broken. The iso-depth scaling result (2604.21106, phi = 0.46) points the same way: a loop buys well under a layer.

**Action:** add this to the risk register in `progress.md`, and make sure the G2 report distinguishes "looped loses at matched FLOPs" from "looped model is broken", because the responses are completely different.

---

## 6. What is genuinely open, ranked

1. **Ground-truth composition depth in pixels.** Nobody has it. Everything else in the paper is downstream of this.
2. **The depth-versus-breadth dissociation.** Untouched by anyone, in any modality. CLEVR's original 2016 design does distinguish chain from tree topology and defines an "effective question size", so the conceptual distinction has precedent, but nobody has related either axis to loop requirements.
3. **Activation patching over (loop, spatial position).** The spatial axis has no text analogue. Strongest instrument novelty.
4. **Halting criteria measured against known required depth**, rather than against accuracy or a search over k.
5. **Vision and text on the same axes.** Our own prior text work makes this cheap for us and expensive for everyone else. Underexploited in the current plan.

---

## 7. Recommended changes to the plan

Ordered by urgency. Each becomes a decisions.md entry if adopted.

1. **Rewrite H2 before the pre-registration freeze.** From "diagnostics do not predict extrapolation" to an oracle-comparison claim. As currently worded it risks being a replication of 2509.23314 in a new modality. This is the single most important change and it must happen before milestone 5.
2. **Reorder the contributions.** H3 and the patching instrument are now more defensible than the loop-count curves. Consider leading with mechanism.
3. **Protect Family C absolutely.** The breadth axis is the unexplored one. D-006 in decisions.md asks whether family A or B has priority. The survey says the answer must not come at family C's expense.
4. **Demote H4 to a cited replication.** Cite Kuo et al. (2606.29983) as prior work.
5. **Reposition the geometry claim** against 2512.19941 rather than presenting it as new.
6. **Add roughly twenty papers to related work**, and split the section into looped vision, mechanism, and halting, because all three are now crowded enough to need their own paragraphs.
7. **Position explicitly against 2607.20594 in the introduction.** It is the nearest neighbour, it is three weeks older than our project start, and it uses group word problems, which is our Family A in text. Reviewers who know it will ask. The answer is pixels, the breadth axis, and the oracle comparison, and that answer is strong enough to state up front rather than defend in rebuttal.
8. **Recheck this survey at milestone 9 and again before submission.** The field is producing directly relevant work at roughly two papers a month. Between February and August 2026 the looped vision literature roughly tripled.

---

## 7b. Who is at Google, verified 2026-09-03

Checked because "Google is competing with us" came up as a concern. It is
partly true and the framing matters, so the affiliations were verified
individually rather than inferred from author lists.

| Paper | Google authors | Domain | Competes with us? |
|---|---|---|---|
| ELT, arXiv 2604.09168 | **Prateek Jain** and **Sujoy Paul** (Google Research India), **Aditya Kusupati** (Google DeepMind, senior author). Three of six, including the advising author | looped **vision**, generation | **No.** Parameter efficiency and elastic test-time compute for image and video synthesis. Does not ask what determines loop count, and has no ground-truth composition depth |
| Mixture-of-Recursions, arXiv 2507.10524 | **Tal Schuster**, **Adam Fisch** (Google DeepMind). NeurIPS 2025 | language | No. Token-level adaptive recursion depth |
| Recirculation, arXiv 2608.17981 | **Michael C. Mozer**, Siddiqui, Sawyer, Sanyal, **Rosanne Liu** (Google DeepMind), 2026-08-18 | language | No. An inference-time technique that **explicitly distinguishes itself from depth-recurrence looping**. New to this survey, added here |

**Assessment.** Google is genuinely active in looped models and holds one
paper in looped vision, so the concern is not baseless. But their line is
efficiency and flexibility: fewer parameters, an adjustable compute knob.
Ours is mechanism and measurement. None of the three can ask our question,
because none has access to the true reasoning depth of an example. We also
depend on ELT rather than merely competing with it: Intra-Loop Self
Distillation is the milestone 3 stability fallback in IMPLEMENTATION.md.

**The real competitor remains arXiv 2607.20594** (Zhang, Hu, Peng, Xie;
Peking University and Fudan University), which uses group word problems in
text, our family A in language. That is the paper the introduction must
address directly, per D-007.

---

## 8. Watch list

Groups producing work that could scoop parts of this, worth checking monthly:

- **Tong Zhang, Junhao Hu, Yun Peng, Tao Xie** (2607.20594). Closest to us scientifically. If they move to vision, our depth axis is largely gone and only the breadth dissociation and the spatial patching remain.
- **Rui-Jie Zhu**, author on both Ouro (2510.25741) and LoopViT (2602.02156). Spans looped language and looped vision, and LoopViT already has a halting rule with attractor framing.
- **Gruszczynski, Olszowiec, Byra, Stefanski, Presta**, authors on both bViT (2605.10661) and Training Crossroads (2608.04879). Active in recurrent ViTs, publishing fast, and the August paper lands directly on our G2 comparison.
- **Blayney, Arroyo, Obando-Ceron, Castro, Courville, Bronstein, Dong** (2604.11791). Mechanistic analysis of looped models with substantial resources behind it.
- **Pappone, Crisostomi, Rodola** (2509.23314). Owns the two-scale geometry and the step-size early exit.

---

## 9. Verification status

**Cleared 2026-08-31.** All nineteen outstanding items were verified by fetching their arXiv abstract pages. Every arXiv ID in this document now resolves to a real paper whose title, authors and date are confirmed. No entry in this survey is now marked LISTED.

Two came back with problems, both recorded above: the Svete and Sabharwal misdescription and unconfirmed venue (Section 1), and the Kirin erratum (Section 2.2).

One further paper was verified and added while clearing the debt:

| ID | Title | Date | Note | Check |
|---|---|---|---|---|
| 2606.18023 | LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling | 2026-06-16 | Yang et al. Finds two loops optimal and **additional loops degrading performance due to positional mismatch**. A concrete overthinking mechanism, and a caution for our k up to 64 extrapolation arm. | VERIFIED |

**Standing requirement.** This status is true as of 2026-08-31 and decays. Re-run verification on any citation added after this date, and re-check the whole list before the bibliography freezes at milestone 12. Version numbers matter: 2604.09870 was clean at v1 and carries an erratum at v2, so confirm the version being cited, not only the identifier.

---

## 10. Update, 2026-09-04. Three papers that change where the novelty sits

All three VERIFIED by fetching the arXiv abstract page.

### 10.1 Our coda lens result is a replication, not a discovery

**Lu, Yang, Lee, Li, Liu, "Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer", arXiv 2507.02199, v1 2025-07-02, v2 2025-09-28.**

They probe Huginn-3.5B on arithmetic **using the Logit Lens and the Coda Lens**, the same two instruments under the same names we use, and report "limited evidence of interpretable latent CoT" from rank trajectories of intermediate result tokens. They also report "significant probing inconsistencies across recurrent blocks", where interpretability depends on layer index and decoding method, and that increasing recurrence depth yields only marginal gains.

**Consequences, stated plainly.**

1. **Our finding that the models do not walk the chain replicates theirs in another modality.** It is convergent evidence, not a discovery, and the entry in `findings.md` must cite them. Presenting it as novel is an error a reviewer in this area would catch immediately.
2. **What is ours is the ground truth.** They track intermediate result tokens in arithmetic, a proxy for the intermediate state. We know the answer after every hop by construction, so we score the lens against the real intermediate rather than a stand-in for it.
3. **They leave open exactly what our state probe answers.** Their probing inconsistency across blocks means they cannot separate absent information from undecodable information. Our trained probe with the D4 positive control does separate them. That is a methodological contribution built on top of their result, and it is the strongest kind: it answers a question their paper raises and does not settle.

### 10.2 The binding problem is where the S3 result belongs

**Huang, Li, Salehi, Chang, Soni, Kording, "Formalizing the Binding Problem", arXiv 2606.03976, v1 2026-06-02.**

An information-theoretic formalisation of binding plus a probing method for measuring binding information in representations. Tested on **Vision Transformers only**, all feedforward. The abstract does not separate binding of spatial features from binding of appearance features.

This reframes the S3 finding. The standing explanation for feedforward binding failure is that such models must resolve every concept simultaneously in one pass, where biological vision uses serial attention over time. **A looped model is the obvious remedy, being serial by construction.**

Our result says the remedy does not work, and says it with a sharp asymmetry: the same looped model composes spatial rearrangements perfectly and never even extracts the colour permutation. That is a claim about the binding problem, made on the architecture the binding literature has not tested, separating two feature types their formalisation does not separate.

**This is now the strongest thing in the project, ahead of H1.** H1 is a measurement contribution that several groups are circling. This is a negative result about the architecture everyone assumes is the fix, inside a problem that acquired a formal definition three months ago.

### 10.3 Gate G2 is riskier, and more interesting, than recorded

**Gao, Chen, Xiao, Yang, Tao, Zhou, Dai, "Loop the Loopies!", arXiv 2607.16051, v1 2026-07-17, v2 2026-07-20.**

They state the standing expectation directly: given an N times increase in pre-training compute, increasing parameter count by N usually beats looping N times. They then overturn it at scale with a mixture-of-experts design, 20B and 6B, against a 30B baseline.

So the compute-matched question is live, was recently believed settled against looping, and turns on design rather than on looping as such. If G2 fails at d=384 with a simple tied core, the correct reading is not "loops do not help" but "loops do not help at this design and scale", and the honest framing cites both this result and the expectation it overturned.

### 10.4 What this does to the plan

| item | before | after |
|---|---|---|
| S3 asymmetry | a curiosity inside family A | **the headline**, positioned in the binding problem |
| coda lens result | novel finding | replication of 2507.02199, cite it |
| state probe | supporting instrument | **the methodological contribution**, answers what 2507.02199 leaves open |
| Huginn graft | confirm in the wild | bring ground truth to a model already probed without it |
| H1 | the paper | the measurement contribution the rest rests on |
| G2 | a formality | a live question with recent literature on both sides |
