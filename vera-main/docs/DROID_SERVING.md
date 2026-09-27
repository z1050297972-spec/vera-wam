# DROID Policy Serving

## 1. What you get

This walkthrough serves the **two-stage DROID policy** on real hardware (Franka FR3 on a
DROID rig): a fine-tuned **WAN 2.1 I2V 14B video planner** (`wan-droid-14b`) generates a
language-conditioned multi-view future on a 128×576 three-camera canvas, and a
VGGT-based **Jacobian IDM** (`idm-droid`) inverts that dream into executable actions.
The server exposes the whole stack over a websocket; the client sends camera context +
proprioception, gets back a chunk of 10 SE(3)-delta actions (`(tx,ty,tz,rx,ry,rz,gripper)`),
and plays them at 15 Hz through the robot's impedance controller. Everything up to and
including §4 runs **without any robot** — the replay backend feeds recorded camera clips
through the identical wire path, so you can validate the full stack server-side first.

> **Research code — no safety certification.** VERA is an experimental research policy.
> It has not been tested for functional safety, carries no certification (ISO 10218 /
> ISO/TS 15066 or otherwise), and can command arbitrary, unpredictable motions at any
> time — including after many successful chunks. It must never be run near people, on
> hardware without a hardware e-stop, or in any setting where an unexpected full-speed
> motion could cause injury or damage. You are solely responsible for safe operation of
> your robot. Read §6 (Safety) in full before connecting any hardware.

## 2. Prerequisites

### Checkpoints

All VERA checkpoints are hosted at
[huggingface.co/sizhe-lester-li/VERA](https://huggingface.co/sizhe-lester-li/VERA):

```bash
pip install -U huggingface_hub   # 1.x REQUIRED — hub <=0.36 truncates this repo's listing ("Fetching 0 files")
hf download sizhe-lester-li/VERA --include "wan-droid-14b/*"     --local-dir vera-ckpts   # ~31 GB: DiT-only bf16 + algo_config.yaml
hf download sizhe-lester-li/VERA --include "idm-droid/*"         --local-dir vera-ckpts   # ~5.1 GB: model.ckpt + config.yaml
hf download sizhe-lester-li/VERA --include "droid-demo-clips/*"  --local-dir vera-ckpts   # ~100 MB: 3-camera test clips
```

Plus the **frozen Wan2.1 base components** (UMT5 text encoder, WAN VAE, CLIP — pulled
from Wan-AI, not re-hosted here):

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /path/to/Wan2.1-I2V-14B-480P
```

License caveat: `idm-droid/` bundles VGGT-1B backbone weights and is therefore
**CC-BY-NC-4.0 (non-commercial)**; everything else is Apache-2.0 weights / MIT code.

### Environment variables

The hosted `algo_config.yaml` resolves all base-model paths through two env vars
(`${oc.env:...}` interpolation is built into the yaml — no path editing needed):

```bash
export VERA_DROID_CKPT_DIR=./vera-ckpts/wan-droid-14b          # tuned DiT + algo_config.yaml
export VERA_WAN14B_CKPT_ROOT=/path/to/Wan2.1-I2V-14B-480P      # frozen text encoder / VAE / CLIP
```

Optional: `VERA_DROID_DYNAMICS_CKPT` (path to the IDM `model.ckpt`; defaults to
`<repo>/vera-ckpts/idm-droid/model.ckpt`, which the download above already satisfies —
see known-gap #1 below), `VERA_DEBUG_DUMP_DIR` (fallback per-chunk dump dir, see §3),
`VERA_ALLTRACKER_ROOT` (path to a local **clone of the
[AllTracker](https://github.com/aharley/alltracker) repo** — it is not
pip-installable; DROID uses the `alltracker` tracker backend by default, so clone it
once with `git clone https://github.com/aharley/alltracker` and point this var at the
clone. Also honored: a sibling `third_party/alltracker` checkout next to the repo),
`XDG_CACHE_HOME` (fallback checkpoint cache when `<repo>/outputs/downloaded` is
unwritable).

### GPU

One ~80 GB card (A100-80GB or H200) holds the 16.39B-param stack in bf16. Build + load
takes ~3 min. Your environment **must have `flash_attn`** — the WAN attention silently
falls back to SDPA without it, which is ~3× slower (~47 s vs ~15 s per generation).
Verify it imports before blaming anything else for slowness.

### Known gaps in the released code (read before filing issues)

1. **IDM checkpoint resolution: local-first, wandb fallback.**
   `vera/server/start_server_droid.py` auto-loads the local checkpoint at
   `<repo>/vera-ckpts/idm-droid/model.ckpt` (with the `config.yaml` sidecar next to
   it — exactly the layout the §2 download produces) whenever that file exists; no
   patching or flags needed. Override the path with the `VERA_DROID_DYNAMICS_CKPT`
   env var. Only when no local checkpoint is found does the builder fall back to
   wandb resolution (`DEFAULT_DYNAMICS_RUN_ID="7wohna95"`, SE(3)-delta D=7;
   `2seo56q5` is the joint-delta D=8 variant) against an anonymized entity
   placeholder — which will not resolve on machines without wandb access. Note that
   `--dynamics-run-id` is **silently ignored** whenever the local checkpoint exists
   (local takes precedence).
2. The docstring in `start_server_droid.py` advertising
   `python -m vera.server.start_server_droid [...]` is **stale** — that module is a
   builder with no `main()`. The only entrypoint is `start_vera_server` (below).
3. The robot-workstation bridge referenced in docstrings
   (`droid.okto_bridge.droid_ws_runner`) is not in this repo; the public client-side
   story is `vera/controller/` (see §4–§5).

## 3. Start the server

```bash
export CUDA_VISIBLE_DEVICES=0
python -m vera.server.start_vera_server --embodiment droid --port 8800 \
    --algo-config ./vera-ckpts/wan-droid-14b/algo_config.yaml \
    --sample-steps 10 \
    --debug-dump-dir ./droid_dumps
```

Healthy startup (~3 min):

```
WanModel loaded ... 16.39B params ...
[teacache] enabled rel_l1_thresh=0.100 on WanModel
vera policy server on 0.0.0.0:8800 | ... | H=10 dt=0.0667 | host=<gpu-node> git=<sha>
server listening on 0.0.0.0:8800
```

Knobs that matter:

| Knob | Effect |
|---|---|
| `--sample-steps` | Diffusion steps per generation; latency scales ~linearly. **10** for fast iteration (quality-validated on real rollouts); **25** was the setting for the best real-robot runs; 40 (the yaml quality default) gave no behavior gain at +9 s/chunk. |
| `--teacache-thresh` / `--no-teacache` | DiT step-skipping cache, default ON at 0.10 (~1.4× on top of flash_attn, near-lossless: PSNR 46.5 / SSIM 0.851). 0.10–0.15 is safe; **never exceed ~0.15** — at 0.25+ quality collapses (SSIM 0.85→0.40). |
| `--debug-dump-dir` / `--no-debug-dump` | Per-chunk npz rollout dumps (dream / jacobian / flow / actions), written on a background thread (no latency cost, disk + one CPU thread). DROID defaults ON — live robot data is too precious to lose — but the default path is a placeholder in the public repo, so **pass one of these two flags explicitly**. Precedence: `--no-debug-dump` > `--debug-dump-dir` > `$VERA_DEBUG_DUMP_DIR`. Cap disk with `configure_runtime(debug_dump_max_bytes=...)`. |
| `--vis-port N` | MJPEG live dashboard (`/policy.mjpg`, `/input.mjpg`, ...): dreamed vs executed frames per chunk. `0` (default) disables it entirely, including the per-infer frame copy on the hot path. |
| `--text "..."` | Default task prompt (also overridable per-request). |
| `--dynamics-run-id` | Selects the IDM checkpoint via wandb — **silently ignored when the local `./vera-ckpts/idm-droid/model.ckpt` exists** (local takes precedence; see §2 known gap 1). |
| `--host` / `--port` | Websocket bind; server binds `0.0.0.0`. Pick a dedicated port — connecting to the wrong port and getting a *different* policy server is a real foot-gun. |

Runtime `configure` endpoint (no restart): `sample_steps`, `lang_guidance` /
`hist_guidance`, `motion_plan_scale`, `n_action_steps`, `control_view_keys`,
debug-dump toggles. (Gripper-gate knobs are **not** server-configurable on the DROID
embodiment — sending `gripper_thresh` etc. returns `configure failed: TypeError`; they
exist only on the mimicgen policy. The DROID gripper z-gate is client-side, configured
via `ActionPlayer.enable_z_gate()` — see §5.) **Warning:** `motion_plan_scale` defaults
to **3.0 — too hot for first contact; set it to 1.0 via `configure` before any robot
motion** (§6). A `reload` endpoint hot-swaps the WAN ckpt,
IDM ckpt, or whole embodiment from the client (~3–4 min rebuild; process and port stay
up; `configure` state is cached and replayed onto the new policy). **After any reload,
re-read `get_server_metadata()`** — `context_frames`, horizon budget, and action scales
change per checkpoint. Never `reload` or change `motion_plan_scale` upward while a robot
episode is running — stop playback, finish the episode, reload, re-read
`get_server_metadata()`, and re-validate with a single supervised `--max-steps 1` chunk
before resuming (checkpoint swaps change action scales and gripper statistics, §8).

Guidance defaults: the yaml ships `lang_guidance: 3.0` / `hist_guidance: 2.0`; on
hardware **5.0 / 1.5** proved best (3.0 let the cold-start chunk hallucinate a full
grasp sequence; 5.0 suppresses it).

Networking: from the robot workstation, `ssh -N -L 8800:localhost:8800 <user>@<gpu-node>`
and connect the client to `127.0.0.1:8800`, or connect directly to the node IP.

## 4. Validate without a robot

Run the full stack against the bundled demo clips (one mp4 per view, in
`view_keys` order) before any hardware is involved:

```bash
python -m vera.controller.run_controller --mode sync --backend replay --host 127.0.0.1 --port 8800 \
    --videos vera-ckpts/droid-demo-clips/second_set/varied_camera_1_35317039.mp4 \
             vera-ckpts/droid-demo-clips/second_set/varied_camera_2_39509833.mp4 \
             vera-ckpts/droid-demo-clips/second_set/hand_camera_16779706.mp4 \
    --prompt "pick up the object" --max-steps 3
```

(`vera/server/example_client.py` — `python -m vera.server.example_client` — is a
minimal raw-websocket demo client that streams the same three `droid-demo-clips`;
`run_controller --backend replay` is the full-featured hardware-free path, and
`vera/server/protocol/websocket_policy_client.py` is the thin ws client underneath
both. Since the clips replay open-loop — observations never follow the returned
actions — expect a healthy cold-start chunk and then near-zero chunks with
`[collapse-canary]` warnings: that's the documented static-context behavior
(troubleshooting #2/#4), not a broken install. Round-trips completing with
`action chunk shape=(10, 7)` is the success signal here.)

**Expected handshake** (available via `client.get_server_metadata()` after connect):
`view_keys=["varied_1","varied_2","hand"]`, `proprio_keys=["q_robot","eef_pos","eef_quat","gripper_qpos"]`,
`action_space=se3_delta` (the server advertises the IDM's `action_mode`; the hosted
`idm-droid` is SE(3)-delta — the D=8 variant would report `joint_delta`),
`action_horizon=10`, `action_dim=7`, `context_frames=9`,
`control_dt=1/15`, `gripper_is_raw=True`, `actions_already_metric=True`,
`action_abs_scale` (per-dim), plus provenance (`git_head`, `hostname`, `argv`). The
client warns on CODE DRIFT if the server git sha differs from the client's.

**Expected result:** 3 chunks, each `action=(10, 7)`, `|mean| ~ 0.02`. A healthy
per-chunk log shows infer latency, the action shape, the action abs-mean, and (on the
server side) the context count (`ctx=9` once the rolling window is full). Client-side
(`vera/controller/controller.py`):

```
infer 5.20s action=(10, 7) |mean|=0.0210
```

Server-side (`vera/server/protocol/vera_policy_adapter.py`):

```
[infer #1] 5.20s H=10 chunk=24 ctx=9 |mean|=0.0210 cold_start
```

`|mean|` below `1e-3` fires the `[collapse-canary]` — a near-frozen chunk (see §8).

**Expected latency** (as measured on the authors' setup: n=20 cycles after warmup,
full 3-view stack; one cycle = 10 actions):

| Config | GPU | steps | teacache | full cycle | per-action |
|---|---|---|---|---|---|
| Default (quality) | H200 | 40 | off | 26.5 s | 2.65 s |
| Deployed | H200 | 10 | 0.10 | 5.2 s | **0.52 s** |
| Default (quality) | A100 | 40 | off | 55.6 s | 5.56 s |
| Deployed | A100 | 10 | 0.10 | 10.4 s | **1.04 s** |

If your numbers are ~3× worse than the table, your env is missing `flash_attn` (§2).

## 5. Connect a real robot

> **Before wiring anything to a real arm, read §6 (Safety) in full.** Do not proceed
> past this point without: a hardware e-stop within reach, a second person or your own
> full attention on the robot, client-side delta clamps enabled, and workspace limits
> enforced in your robot env.

The only robot-specific glue you write is a `RobotBackend` (`vera/controller/robot_iface.py`)
returning a state dict with per-view frames under `varied_1` / `varied_2` / `hand`
plus `joint_position`, `cartesian_position` (`[xyz, euler]`), and gripper position.
`ObsBuilder` (`vera/controller/obs_builder.py`) does the rest: resizes each view to
**128×192 RGB**, width-concatenates in `view_keys` order, maintains the rolling context
window, and emits the wire dict.

### Obs sent per `infer`

| wire key | content | shape / dtype |
|---|---|---|
| `context_rgb` | last `context_frames` frames, per-view resized then **width-concatenated** in `view_keys` order | `[T, 128, ΣWᵢ, 3]` uint8 |
| `view_keys` / `view_widths` | `["varied_1","varied_2","hand"]` + per-view widths (summing to ΣW) | lists |
| `q_robot` | joint positions (7) | float32 |
| `cartesian_position` | eef pose `[xyz, euler]` (6) | float32 |
| `gripper_position` | (1) | float32 |
| `session_id` | uuid, **one per episode** | str |
| `prompt` | task text (re-conditioned on change, ~50 ms) | str |

**Cold start:** the FIRST infer of an episode sends a **single frame (T=1)** — never a
repeated window. The server pads internally; the client must NOT pre-repeat the seed
frame (a perfectly static context prefix is the strongest "stay static" prior and
collapses the planner — see §8). Then append one frame at **every executed control
step** and send the most recent `context_frames` each infer.

### Actions returned

`{"action": float32 (10, 7)}` — SE(3) deltas plus a raw gripper channel.

**SE(3) delta convention (exact, from the training loader):** `dT = T1 @ inv(T0)` —
**world/base frame** (left-multiplied); translation in **meters** (not eef-projected,
not divided by dt); rotation = axis-angle rotvec of `R1·R0ᵀ` in **radians**; gripper =
raw delta. The DROID server sets `actions_already_metric=True` in the handshake: the
actions arrive in physical units and the client **must not rescale them** —
`ActionPlayer` (`vera/controller/action_player.py`) enforces this. Never invent fudge
factors; the denorm must match training.

**Gripper:** raw float, last dim. Simple binarize is `>0.5 → close` (position-mode
bang-bang), but the **z-score gate** (`GripperZGate` in `action_player.py`: Welford
z-score with arrival gating — commit close only once descent has decayed) is strongly
recommended: raw gripper statistics vary wildly across IDM checkpoints and absolute
thresholds do not transfer (§8). Enable it via
`ActionPlayer.enable_z_gate(k, arrival_dz)` — `arrival_dz` has no default and must be
passed explicitly (e.g. `0.002` m/step). Optionally, a multi-vote open debounce is
available by constructing `GripperZGate(..., open_steps=2)` yourself; the default is
`open_steps=1` and `enable_z_gate()` does not expose it. Make sure your robot env uses
`gripper_action_space="position"`.

**Playback:** `control_dt = 1/15 s`. The client applies each of the 10 actions at 15 Hz
via a non-blocking robot command; the low-level 1 kHz impedance controller interpolates
between the 15 Hz setpoints. One chunk = 0.67 s of motion.

### Controller modes and lifecycle

`VeraController` (`vera/controller/controller.py`) runs two modes:

- `sync_hold` — obs → infer → play chunk → repeat; robot frozen during infer.
  Default and the right mode for first contact.
- `async_pipeline` — infer for chunk N+1 runs on the obs snapshot from the start of
  chunk N (1-chunk stale); robot holds last pose on underrun. Necessary for smooth
  motion: chunk playback is 0.67 s vs 5–20 s infer, so the sync loop is ≥87% frozen.
  Switch to `async_pipeline` only after multiple clean, fully supervised `sync_hold`
  episodes at reduced `motion_plan_scale`. Async does not relax any safety
  requirement — the operator must remain at the e-stop for every episode; there is no
  unattended mode.

Episode lifecycle: new episode → new uuid → `client.reset({"session_id", "reason"})`;
the server also auto-resets on a changed `session_id` (defense in depth). On
stop/SIGTERM: stop playback, send one final `reset` (flushes server-side artifacts),
close the socket.

**Bring-up order:** clean server log → replay smoke (§4) → wire your camera reader →
**dry-run: run the full loop with robot commands disabled (log actions, send nothing)
and sanity-check magnitudes against `action_abs_scale`** → set `motion_plan_scale=1.0`
via `configure` (§6) → `sync` mode with `--max-steps 1`, hand on the e-stop, and watch a
single chunk → raise `--max-steps` under continuous supervision → switch to async
**only after several clean supervised sync episodes**.

## 6. Safety

**Research code — no safety certification.** VERA is an experimental research policy
with no functional-safety testing or certification; it can command arbitrary,
unpredictable motions at any time (see the disclaimer in §1).

**This stack commands a real arm. Non-negotiables:**

- **E-stop within reach at all times. Never leave the robot unattended** while the
  controller is running.
- **Client-side clamps are the primary rail, always on:** clip / rate-limit deltas
  before sending (proven values: **1–3 cm and 3° per step**), plus a **per-chunk
  translation budget of 0.12–0.15 m with uniform scaling** (preserves direction). This
  budget is what prevented a 0.45 m/s dive during development. Verify the clamps are
  live before the first commanded chunk: in the dry-run (§5), inject a synthetic
  oversized action (e.g. 0.3 m tz) through your playback path and confirm the logged
  post-clip command respects the 1–3 cm/step and 0.12–0.15 m/chunk rails. If your
  integration constructs `ActionPlayer` yourself, confirm the clip/rate-limit/budget
  parameters are set — they are your responsibility, not the server's.
- **Start with reduced action scaling.** `motion_plan_scale` is runtime-settable via
  the `configure` endpoint. Before the first commanded chunk, set the scale down
  explicitly — do not rely on the builder default:
  `client.configure(motion_plan_scale=1.0)` (or patch the `MOTION_PLAN_SCALE`
  constant in `start_server_droid.py`). The shipped default of **3.0** is a legacy tuning value,
  not a recommended starting point; 2.5 already overshoots, and 1.0–1.5 gave the best
  tempo. Begin low and raise deliberately — and if the robot moves far less than the
  dream intends, check this knob *before* loosening your clamps (§8).
- **Enforce workspace limits in your robot env** independently of anything the policy
  sends — the server knows nothing about your table. For first sessions: set
  conservative virtual walls (a box a few cm above the table surface and well inside
  the rig), clear the workspace of anything fragile or rigid the arm could crash into,
  and start from a home pose with margin below and around it — the planner has a known
  tendency to dive on the first chunk (§8).
- **Validate open-loop first:** run §4's replay smoke, then a single chunk
  (`--max-steps 1`) with your hand on the e-stop, before any closed-loop episode.
- Keep debug dumps ON for robot sessions — post-hoc analysis of a bad chunk needs the
  dream/jacobian/action npz.

## 7. Performance expectations

Beyond the latency table in §4:

- **flash_attn is the single biggest free win (3×)** over the silent SDPA fallback.
- **teacache 0.10** adds ~1.4× on top, quality-validated at the deployed 10-step point
  (PSNR 46.5 / SSIM 0.851 / LPIPS 0.087, 33% of DiT steps skipped). ≥0.25 destroys
  quality (SSIM → 0.40). Cache-on at other step counts is not yet quality-validated.
- Per-DiT-step cost ≈ 0.64 s (H200) / 0.52 s (A100) at 3 forward passes per step.
  Denoise dominates the cycle (~4.3 s of the 5.2 s H200 deployed cycle); tracker
  ~0.28 s, Jacobian IDM ~0.22 s, action solve ~0.02 s.
- **Context length is latency-free** in the benchmark (the DiT pads to a fixed latent
  budget), but behaviorally it matters a lot — see §8 on ctx=9 vs ctx=21.
- Duty-cycle math: 0.67 s of motion per chunk vs 5–20 s infer means `sync` mode is
  mostly frozen. Levers: async pipelining (§5), fewer steps, a longer horizon (the WAN
  predicts up to 24 future frames = 1.6 s), or accept stop-and-go.

## 8. Troubleshooting

Field-tested symptom → cause → fix, from real FR3 deployment:

1. **Chunk-boundary jump / discontinuous first action of each chunk** → client sent 1–2
   context frames instead of the full window, so each plan is ungrounded relative to
   the last → maintain the rolling per-step context buffer; verify `ctx=<expected>` in
   the log on every infer.
2. **`[collapse-canary]` / frozen chunks (`|mean| < 1e-3`)** → starved or *static*
   context → fix context plumbing; never pad with repeated frames.
3. **Temporal state wiped every chunk** → client created a new `session_id` per infer
   (e.g. by looping the `--max-steps 1` first-contact recipe) → ONE session id per
   episode, exactly one reset pair, `cold_start` at most once per episode.
4. **Static-frame poison** (tested twice on hardware, both froze the robot): prefilling
   the context with held frames collapses the dream within ~2 chunks (inter-frame diffs
   8–10 px → ~1 px). Use a true 1-frame cold start; validated upgrades are a short
   scripted real motion before infer #1, or settle-capture of ~9 *real* (sensor-noisy)
   frames — both supply genuine inter-frame signal.
5. **First-chunk dream weak or hallucinated** → 1-frame cold start (not teacache —
   teacache resets per generation) → execute only 1–2 actions of the first chunk then
   replan; `lang_guidance=5.0` suppresses the cold-start grasp hallucination that 3.0
   produced.
6. **Freezing near contact with long context** → a 21-frame context acted as a static
   attractor near contact (every ctx=21 episode froze during approach) → ctx=9
   eliminated freezes and produced emergent retry behavior (close → reopen → re-close).
7. **Robot moves far less than the dream intends** → planner-side scale interacting with
   client clamps (commanded steps ~10× `action_abs_scale` were 90% clipped by a 1 cm
   clamp) → check `motion_plan_scale` first (3.0 vs 1.0 changed everything), then relax
   clamps deliberately; keep the per-chunk budget as the primary rail.
8. **Adaptive-gain windup** → an achieved-vs-desired gain estimator integrating against
   the clip rails climbs monotonically to its cap → make gain updates rails-aware (skip
   updates on clamped chunks) or feed back post-clip executed translation (the
   `rails_bound` / `executed_trans` infer extras exist for this).
9. **Gripper never re-opens** → robot env defaulted the gripper to *velocity* mode, so a
   binarized 0.0 "open" was a zero-velocity no-op → pass
   `gripper_action_space="position"` to match binarize semantics.
10. **Gripper never closes** (|raw| < 0.1 vs a 0.5 threshold) → weak raw gripper channel
    + naive absolute threshold → use the z-score gate with arrival gating and open
    debounce (§5); this is what closed the full pick-place loop.
11. **Gripper thresholds don't transfer across IDM checkpoints** → gripper-delta
    statistics differ wildly between IDMs (one had median −1.05, σ≈15 — legacy
    thresholds sat inside the noise core: 1/10 success; recalibrating from measured
    percentiles → 4/10 in one pass) → after any IDM swap, measure the gripper delta
    distribution from the first debug dumps BEFORE trusting any gate.
12. **EEF orientation drifts on rotation-zeroed dims** → the cartesian impedance
    controller drifts pitch open-loop (+20°/episode) → client-side rot-lock: P-control
    to the episode-start orientation snapshot (~3.0 rad/s per rad) on the zeroed dims.
13. **Early release / early close / carry undershoot** → the dream's internal task
    progress runs ahead of reality (monocular geometry + generated content — the dream
    "believed" it was already at the goal), and the gate faithfully executes generated
    content; per-checkpoint close-commit depth bias exists (+2 to +10 cm across exports
    in the same scene) → arrival gating, an optional proprio z-floor on close commits,
    or scene finetuning.
14. **Weak vertical (tz) actions** → world-vertical motion lies near the wrist camera's
    optical axis → near-zero pixel flow → weakly observed tz Jacobian column, shrunk by
    ridge regularization (normalization scales are NOT the culprit) → diagnose with the
    planner-flow vs J-column vs solved-tz numbers in the debug dump; consider view
    weighting.
15. **Context-tail contamination on gripper-transition chunks** → fingers still moving
    when the last context frame is captured → settle-capture (wait until two consecutive
    frames differ below threshold) instead of a fixed frame count.
16. **Weird behavior after `reload`** → stale client-side metadata → always re-read
    `get_server_metadata()` after a reload or restart (`context_frames`, horizon budget,
    and action scales change per checkpoint). Ctrl-C triggers a final reset so dump
    artifacts flush; a mid-infer crash can leave one half-written npz (known, tolerated).
17. **Client dies mid-inference with `keepalive ping timeout` (close code 1011)** →
    the server's event loop blocks for the full planner inference (tens of seconds on
    cold start), so it cannot answer websocket keepalive pings and the client hangs up
    before the action arrives → disable client-side keepalive (`ping_interval=None` in
    `websockets`' connect, as `example_client.py` does) or raise the ping timeout well
    above your worst-case inference latency.

Config sweet spots from the best real-robot sessions, as a starting point:
`sample_steps=25`, `lang/hist guidance = 5.0/1.5`, `motion_plan_scale=1.0–1.5`,
context 9 frames, horizon 12–16 (context lengths must satisfy `k·4+1` — the latent
stride). In one scene, a hand-camera-only view config out-committed the full 3-view
default on grasping; a slightly lower episode home pose shortens the planner's beloved
first-chunk dive.
