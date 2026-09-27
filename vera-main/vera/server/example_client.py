"""Example client for start_server_droid.py (MotionPolicyGripper / FR3 + DROID).

Usage:
    python -m vera.server.example_client
    python -m vera.server.example_client --host <gpu-node> --port 8765

Cross-network:
    # SSH tunnel (simplest):
    #     ssh -N -L 8765:<gpu-node>:8765 user@login-node
    #     python -m vera.server.example_client --host localhost --port 8765

Streams three synchronized DROID camera clips frame-by-frame as observations.
By default it uses the demo clips hosted with the checkpoints
(hf download sizhe-lester-li/VERA --include "droid-demo-clips/*" --local-dir ./vera-ckpts):
pass --videos-dir to point at your own three per-view clips instead. Each view
is trimmed/resampled to a common FPS, stitched width-wise at the model's
per-view resolution, and sent one frame per step.

Note this is a TRANSPORT demo, not a policy rollout: the clip plays open-loop, so
observations never follow the returned actions. Expect a healthy first (cold-start)
chunk and then near-zero chunks with `[collapse-canary]` warnings server-side — the
documented static-context behavior (docs/DROID_SERVING.md troubleshooting #2/#4).
On a real robot the arm's own motion keeps the dream alive.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import websockets.sync.client

from vera.server.server_utils import Packer, unpackb

# DROID views and image size. Per-view size should match the WAN algo config
# (height, width/num_views). Adjust if your server uses a different resolution.
DROID_VIEW_KEYS = ["varied_1", "varied_2", "hand"]
# Per-view size must match the WAN algo config (height, width/num_views).
# droid_8C_6T_fullsteps uses height=128, width=576 → 192 per view for 3 views.
PER_VIEW_H = 128
PER_VIEW_W = 192
NUM_VIEWS = len(DROID_VIEW_KEYS)
IMAGE_H = PER_VIEW_H
IMAGE_W = PER_VIEW_W * NUM_VIEWS  # width-concatenated multi-view frame

# Per-view clip filename patterns, resolved inside --videos-dir. Order matches
# DROID_VIEW_KEYS so the stitched frame lines up with the server's view order.
DEFAULT_VIDEOS_DIR = Path("./vera-ckpts/droid-demo-clips/second_set")
VIDEO_PATTERNS = {
    "varied_1": ("varied_camera_1*", "varied1*"),
    "varied_2": ("varied_camera_2*", "varied2*"),
    "hand":     ("hand_camera*", "hand*"),
}


def resolve_video_files(videos_dir: Path) -> dict:
    files = {}
    for key, patterns in VIDEO_PATTERNS.items():
        for pat in patterns:
            hits = sorted(videos_dir.glob(pat))
            if hits:
                files[key] = hits[0]
                break
        else:
            raise FileNotFoundError(
                f"No clip for view '{key}' in {videos_dir} (tried {patterns}). "
                "Download the demo clips: hf download sizhe-lester-li/VERA "
                '--include "droid-demo-clips/*" --local-dir ./vera-ckpts'
            )
    return files

# Playback / alignment defaults. The three cameras are not time-synchronized;
# tune START_SEC per view if the first visible frames don't match up.
TARGET_FPS = 15
START_SEC = {"varied_1": 0.0, "varied_2": 0.0, "hand": 0.0}
NUM_STEPS = 50



def _recv(ws):
    """Server replies may be binary msgpack or plain text frames (reset/configure
    acks) — unpack only the binary ones."""
    msg = ws.recv()
    if isinstance(msg, (bytes, bytearray, memoryview)):
        return unpackb(msg)
    return msg


def load_video_frames(path: Path) -> tuple[np.ndarray, float]:
    """Decode an entire video into uint8 [T, H, W, 3] RGB, return (frames, fps)."""
    import av  # imported lazily so the module still loads without PyAV
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()
    return np.stack(frames), fps


def resize_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize uint8 [T, H_in, W_in, 3] -> [T, h, w, 3] via PIL/LANCZOS."""
    from PIL import Image
    out = [np.array(Image.fromarray(f).resize((w, h), Image.LANCZOS)) for f in frames]
    return np.stack(out)


def trim_and_resample(frames: np.ndarray, src_fps: float, start_sec: float,
                      target_fps: float) -> np.ndarray:
    """Trim from start_sec and resample to target_fps by index selection."""
    start = int(round(start_sec * src_fps))
    frames = frames[start:]
    ratio = src_fps / target_fps
    n = int(len(frames) / ratio)
    indices = [int(round(i * ratio)) for i in range(n)]
    indices = [i for i in indices if i < len(frames)]
    return frames[indices]


def build_stitched_video(video_files: dict) -> np.ndarray:
    """Load all three views, align to TARGET_FPS, resize, stitch width-wise."""
    aligned = {}
    for key in DROID_VIEW_KEYS:
        path = video_files[key]
        raw, fps = load_video_frames(path)
        trimmed = trim_and_resample(raw, fps, START_SEC[key], TARGET_FPS)
        aligned[key] = trimmed
        print(f"  loaded {key:<9} {path.name}  raw={raw.shape} @ {fps:.1f}fps"
              f"  -> aligned={trimmed.shape} @ {TARGET_FPS}fps")

    min_len = min(v.shape[0] for v in aligned.values())
    aligned = {k: v[:min_len] for k, v in aligned.items()}

    tiles = [resize_frames(aligned[k], PER_VIEW_H, PER_VIEW_W) for k in DROID_VIEW_KEYS]
    stitched = np.concatenate(tiles, axis=2)  # [T, H, W*num_views, 3]
    assert stitched.shape[1:] == (IMAGE_H, IMAGE_W, 3), stitched.shape
    return stitched


def main():
    parser = argparse.ArgumentParser(description="DROID MotionPolicyGripper client example")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps to stream. Defaults to the length of the stitched video.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Wrap around to frame 0 when --steps exceeds the video length.",
    )
    parser.add_argument(
        "--request-every",
        type=int,
        default=1,
        help="Run the policy every Nth step; other steps only enqueue the frame.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=16,
        help="Number of observations to warmup the policy with before running the policy.",
    )
    parser.add_argument(
        "--prompt",
        default="a white robot arm reaches down to pick up a line of crackers and then places it into a white tray.",
        help="Text conditioning to send on configure",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Directory holding the three per-view clips (default: the hosted demo clips).",
    )
    args = parser.parse_args()

    print(f"Loading videos from {args.videos_dir}...")
    video_files = resolve_video_files(args.videos_dir)
    stitched_video = build_stitched_video(video_files)
    video_len = stitched_video.shape[0]
    print(f"Stitched video: {stitched_video.shape} @ {TARGET_FPS} fps")

    num_steps = args.steps if args.steps is not None else video_len
    if not args.loop and num_steps > video_len:
        print(f"Warning: --steps {num_steps} > video length {video_len}, capping."
              " Pass --loop to wrap around instead.")
        num_steps = video_len

    uri = f"ws://{args.host}:{args.port}"
    print(f"Connecting to {uri}...")
    ws = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=10.0,
        # The planner blocks the server's event loop during inference (tens of
        # seconds on cold start), so it cannot answer keepalive pings; leave
        # client-side pings off or the socket dies mid-inference with
        # "keepalive ping timeout".
        ping_interval=None,
    )
    packer = Packer()

    # 1. Receive server metadata on connect.
    metadata = _recv(ws)
    print(f"Server metadata: {metadata}")
    context_frames = int(metadata.get("context_frames") or 9)
    context_buf: list = []

    # 2. Reset the policy (clears frame buffer + action queue).
    ws.send(packer.pack({"endpoint": "reset"}))
    print(f"Reset response: {_recv(ws)}")

    # 3. Configure runtime: set the text prompt for this episode.
    ws.send(packer.pack({
        "endpoint": "configure",
        "text_conditioning": args.prompt,
    }))
    print(f"Configure response: {_recv(ws)}")

    # 4. Stream observations and receive actions.
    print(f"\nStreaming {num_steps} observations to {uri}...")
    for step in range(num_steps):
        # Pull the next frame from the stitched video. Layout:
        #   rgb: (H, W*num_views, 3) uint8 — views concatenated width-wise.
        frame_idx = step % video_len if args.loop else min(step, video_len - 1)
        rgb = stitched_video[frame_idx]

        # FR3 gripper state. Dummy values — replace with real robot state.
        eef_pos = np.zeros(3, dtype=np.float32)
        eef_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # (x, y, z, w)
        gripper_qpos = np.zeros(2, dtype=np.float32)

        # Public wire contract: each infer request carries the ROLLING CONTEXT
        # STACK context_rgb [T, H, W*V, 3] uint8 (the server advertises
        # context_frames in its handshake), not a single frame. Accumulate
        # locally; only send a request every `request_every` steps.
        context_buf.append(rgb)
        if len(context_buf) > context_frames:
            context_buf.pop(0)

        run_policy = ((step + 1) % args.request_every) == 0 and step >= args.warmup_steps
        if not run_policy:
            continue

        obs_msg = {
            "context_rgb": np.stack(context_buf),   # [T, H, W*V, 3] uint8
            "view_keys": DROID_VIEW_KEYS,
            "view_widths": [PER_VIEW_W] * NUM_VIEWS,
            "session_id": "example-client-0",
            "q_robot": np.zeros(7, dtype=np.float32),
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "gripper_qpos": gripper_qpos,
        }

        t0 = time.monotonic()
        ws.send(packer.pack(obs_msg))
        resp = _recv(ws)
        dt_ms = (time.monotonic() - t0) * 1000

        if isinstance(resp, str) or (isinstance(resp, dict) and resp.get("status") == "error"):
            print("Server returned an error:")
            print(resp if isinstance(resp, str) else resp.get("traceback", resp))
            break

        action = np.asarray(resp["action"])
        print(
            f"  step {step:3d} | action chunk shape={action.shape} "
            f"abs_max={np.abs(action).max():.4f} "
            f"infer_s={resp.get('info', {}).get('infer_s', '?')} | {dt_ms:.1f}ms round-trip"
        )

    ws.close()
    print("Done.")


if __name__ == "__main__":
    main()
