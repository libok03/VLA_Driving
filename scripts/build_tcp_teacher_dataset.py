#!/usr/bin/env python3
"""Build a compact TCP training store from recorded MORAI teacher bags.

The source bags are never modified.  Frames are sampled at 4 Hz; numeric
future pose/control labels are interpolated/aligned from their raw timestamps.
Recorded privileged-expert behavior is used only as a sampling stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


TOPICS = {
    "front": "/camera/front/image/compressed",
    "pose": "/teacher/ground_truth_state",
    "localization": "/localization/kinematic_state",
    "vehicle": "/morai/ego_vehicle_status",
    "route": "/local_route",
    "gps": "/gps",
    "control": "/ctrl_cmd",
    "behavior": "/privileged_expert/behavior",
}
ACTION = {"DRIVE": 0, "STOP": 1, "AVOID": 2}
DT = 0.25
FUTURE_DT = 0.2
FUTURE_COUNT = 20
CHUNK = 100


def yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def route_value(msg) -> np.ndarray:
    if len(msg.poses) < 2:
        raise ValueError("local route has fewer than two poses")
    out = np.zeros((64, 4), np.float32)
    poses = list(msg.poses)
    for i in range(64):
        p = poses[min(i, len(poses) - 1)].pose
        a = yaw(p.orientation)
        out[i] = (p.position.x, p.position.y, math.cos(a), math.sin(a))
    return out


def series_prior(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 1)


def pack_jpegs(values: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray([len(x) for x in values], np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(sizes, dtype=np.int64)]
    data = np.frombuffer(b"".join(values), np.uint8).copy()
    return data, offsets


def digest8(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()[:8]


def convert_bag(bag: Path, output: Path, control_root: Path) -> dict:
    numeric = {k: [] for k in ("pose", "localization", "vehicle", "route", "gps", "control", "behavior")}
    front_times, fronts = [], []
    inverse = {v: k for k, v in TOPICS.items()}
    with AnyReader([bag]) as reader:
        origin = reader.start_time
        connections = [c for c in reader.connections if c.topic in inverse]
        for connection, stamp, raw in reader.messages(connections=connections):
            name = inverse[connection.topic]
            t = (stamp - origin) * 1e-9
            msg = reader.deserialize(raw, connection.msgtype)
            if name == "front":
                front_times.append(t); fronts.append(bytes(msg.data)); continue
            if name in {"pose", "localization"}:
                p = msg.pose.pose
                value = (p.position.x, p.position.y, yaw(p.orientation))
            elif name == "vehicle":
                value = abs(float(msg.signed_vel)) / 3.6
            elif name == "route":
                value = route_value(msg)
            elif name == "gps":
                value = int(msg.status.status) >= 0
            elif name == "control":
                # cmd_type=1: accelerator/brake and steering are normalized.
                value = (float(msg.accel) - float(msg.brake), float(msg.steer))
            elif name == "behavior":
                value = ACTION.get(str(msg.data).strip().upper(), 0)
            numeric[name].append((t, value))
    # GPS is optional. Some otherwise complete teacher bags were recorded
    # without /gps; retain them and mark every sample as GPS blackout instead
    # of fabricating measurements or discarding valid camera/control labels.
    required = {k: v for k, v in numeric.items() if k != "gps"}
    missing = [k for k, v in {**required, "front": fronts}.items() if not v]
    if missing:
        raise RuntimeError(f"{bag.name}: missing {missing}")

    times = {k: np.asarray([x[0] for x in v], np.float64) for k, v in numeric.items()}
    values = {k: np.asarray([x[1] for x in v]) for k, v in numeric.items()}
    ft = np.asarray(front_times, np.float64)
    synchronized_times = [value for key, value in times.items() if key != "gps"]
    start = max(ft[0], *(x[0] for x in synchronized_times)) + 0.5
    end = min(ft[-1], *(x[-1] for x in synchronized_times)) - 4.05
    master = np.arange(start, end + 1e-8, DT)
    if len(master) < 20:
        raise RuntimeError(f"{bag.name}: insufficient synchronized duration")
    indices = {k: series_prior(t, master) for k, t in times.items() if len(t)}
    fi = series_prior(ft, master)
    selected_front = [fronts[int(i)] for i in fi]
    selected_route = values["route"][indices["route"]].astype(np.float32)
    selected_pose = values["pose"][indices["pose"]].astype(np.float64)
    selected_localization = values["localization"][indices["localization"]].astype(np.float64)
    selected_speed = values["vehicle"][indices["vehicle"]].astype(np.float32)
    selected_control = values["control"][indices["control"]].astype(np.float32)
    selected_action = values["behavior"][indices["behavior"]].astype(np.int64)
    if len(times["gps"]):
        gps_age = master - times["gps"][indices["gps"]]
        gps_valid = values["gps"][indices["gps"]].astype(bool) & (gps_age <= 0.5)
        gps_policy = "recorded /gps status with 0.5 s freshness limit"
    else:
        gps_valid = np.zeros(len(master), dtype=bool)
        gps_policy = "missing /gps topic; every sample explicitly marked blackout"

    run_id = f"{bag.stem}_{digest8(bag)}"
    run_dir = output / run_id
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for ci, begin in enumerate(range(0, len(master), CHUNK)):
        finish = min(begin + CHUNK, len(master)); n = finish - begin
        front_data, front_offsets = pack_jpegs(selected_front[begin:finish])
        empty_offsets = np.zeros(n + 1, np.int64)
        filename = f"frames/frames_{ci:05d}.npz"
        np.savez_compressed(
            run_dir / filename,
            front_jpeg_data=front_data, front_jpeg_offsets=front_offsets,
            left_jpeg_data=np.zeros(0, np.uint8), left_jpeg_offsets=empty_offsets,
            right_jpeg_data=np.zeros(0, np.uint8), right_jpeg_offsets=empty_offsets,
            lidar_bev=np.zeros((n, 1), np.uint8), imu=np.zeros((n, 1)),
            vehicle=np.c_[selected_speed[begin:finish], np.zeros((n, 4), np.float32)],
            # Privileged pose is retained only for future trajectory labels;
            # it is never placed in the model input state or used to index the
            # reviewed command map.
            pose=selected_pose[begin:finish],
            # The reviewed command map is indexed using the vehicle's actual
            # localization output, never privileged teacher ground truth.
            localization_pose=selected_localization[begin:finish],
            health=np.c_[gps_valid[begin:finish].astype(np.float32), np.zeros((n, 4), np.float32)],
            route=selected_route[begin:finish], mgeo=np.zeros((n, 1), np.float32),
        )
        chunks.append({"file": filename, "start_frame": begin, "end_frame_exclusive": finish})
    (run_dir / "frame_chunks.json").write_text(json.dumps({"chunks": chunks}, indent=2))

    pose_t, pose_v = times["pose"], values["pose"].astype(np.float64)
    pose_yaw = np.unwrap(pose_v[:, 2])
    sample = {k: [] for k in ("sample_id", "current_frame_idx", "history_frame_idx", "future_frame_idx", "relative_x", "relative_y", "relative_yaw", "future_speed", "gps_blackout", "action_state")}
    for current in range(4, len(master)):
        now = master[current]
        query = now + np.arange(1, FUTURE_COUNT + 1) * FUTURE_DT
        if query[-1] > pose_t[-1]: break
        xy = np.stack([np.interp(query, pose_t, pose_v[:, j]) for j in (0, 1)], 1)
        here = np.asarray([np.interp(now, pose_t, pose_v[:, j]) for j in (0, 1)])
        heading = float(np.interp(now, pose_t, pose_yaw)); delta = xy - here
        c, s = math.cos(heading), math.sin(heading)
        local = np.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), 1)
        # Reject only unsynchronised/reset labels.  This does not alter pose
        # values and does not remove ordinary GPS-blackout/tunnel samples.
        # At the dataset's <=65 km/h, a >10 m jump in 0.2 s is impossible.
        segment = np.linalg.norm(
            np.diff(np.vstack((np.zeros((1, 2)), local)), axis=0), axis=1
        )
        if segment.max() > 6.0:
            continue
        future_yaw = (np.interp(query, pose_t, pose_yaw) - heading + np.pi) % (2*np.pi) - np.pi
        future_speed = np.interp(query, times["vehicle"], values["vehicle"])
        ff = np.clip(np.searchsorted(master, query), 0, len(master)-1)
        sample["sample_id"].append(len(sample["sample_id"])); sample["current_frame_idx"].append(current)
        sample["history_frame_idx"].append(np.arange(current-4, current+1)); sample["future_frame_idx"].append(ff)
        sample["relative_x"].append(local[:, 0]); sample["relative_y"].append(local[:, 1])
        sample["relative_yaw"].append(future_yaw); sample["future_speed"].append(future_speed)
        sample["gps_blackout"].append(not gps_valid[current]); sample["action_state"].append(selected_action[current])
    arrays = {k: np.asarray(v) for k, v in sample.items()}
    np.savez_compressed(run_dir / "sample_index.npz", **arrays)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 2, "dataset_version": "tcp_teacher_bags_v1", "run_id": run_id,
        "source_run_id": bag.stem, "source_bag": str(bag), "sample_count": len(arrays["sample_id"]),
        "action_label_policy": "recorded /privileged_expert/behavior", "raw_data_modified": False,
        "command_position_topic": "/localization/kinematic_state",
        "trajectory_label_topic": "/teacher/ground_truth_state",
        "gps_policy": gps_policy,
    }, indent=2))
    control_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(control_root / f"{bag.stem}.npz", timestamp=master, control=selected_control)
    counts = np.bincount(arrays["action_state"], minlength=3)
    return {"run_id": run_id, "bag": bag.name, "frames": len(master), "samples": len(arrays["sample_id"]), "actions": dict(zip(ACTION, counts.tolist())), "blackout_samples": int(arrays["gps_blackout"].sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bag-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control-cache", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output / "manifest.json"
    done = {}
    if manifest_path.exists():
        done = {x["bag"]: x for x in json.loads(manifest_path.read_text()).get("runs", [])}
    bags = sorted(a.bag_dir.glob("*.bag"))
    failures = {}
    failure_path = a.output / "conversion_failures.json"
    if failure_path.exists():
        failures = json.loads(failure_path.read_text()).get("failures", {})
    for i, bag in enumerate(bags, 1):
        if bag.name in done:
            print(f"[{i}/{len(bags)}] SKIP {bag.name}", flush=True); continue
        print(f"[{i}/{len(bags)}] START {bag.name}", flush=True)
        try:
            result = convert_bag(bag, a.output, a.control_cache)
        except (RuntimeError, ValueError) as exc:
            failures[bag.name] = str(exc)
            failure_path.write_text(json.dumps({"failures": failures}, indent=2))
            print(f"[{i}/{len(bags)}] FAILED {bag.name}: {exc}", flush=True)
            continue
        done[bag.name] = result
        failures.pop(bag.name, None)
        manifest_path.write_text(json.dumps({"runs": list(done.values())}, indent=2))
        failure_path.write_text(json.dumps({"failures": failures}, indent=2))
        print(f"[{i}/{len(bags)}] DONE {json.dumps(result, sort_keys=True)}", flush=True)
    ids = sorted(x["run_id"] for x in done.values())
    rng = np.random.default_rng(a.seed); rng.shuffle(ids)
    n = len(ids); nv = max(1, round(n*.1)); nt = max(1, round(n*.1))
    split = {"train": sorted(ids[:n-nv-nt]), "val": sorted(ids[n-nv-nt:n-nt]), "test": sorted(ids[n-nt:])}
    (a.output / "split.json").write_text(json.dumps({"schema_version": 1, "split_unit": "run", "seed": a.seed, "data_root": str(a.output), "splits": split}, indent=2))


if __name__ == "__main__":
    main()
