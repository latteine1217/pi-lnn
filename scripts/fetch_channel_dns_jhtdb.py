#!/usr/bin/env python3
"""抓 JHTDB channel flow Re_τ=1000 DNS cutout（Plan 3 Task 3a）。

What: getCutout strided cutout，per-frame resumable 存 .npy。
Why:  getCutout 限制 box index-span < 2²⁸ points（**regardless of strides**）；
      全 domain (2048×512×1536=1.6e9) 超限 → 沿 x 分塊抓再拼接。
      channel chunk 64³；per-frame skip-existing 可中斷續抓。

軸序 INVARIANT：getCutout 回 [Nz, Ny, Nx, 3] = arr[z,y,x,c]，c: 0=u,1=v,2=w。
物理域：x∈[0,8π] (uniform 2048)、z∈[0,3π] (uniform 1536)、y∈[-1,1] (B-spline 512)。

用法（token 從 .env）:
  set -a; . ./.env; set +a
  uv run --with givernylocal python scripts/fetch_channel_dns_jhtdb.py --smoke
  uv run --with givernylocal python scripts/fetch_channel_dns_jhtdb.py --t-stride 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# channel full grid（JHTDB 官方：2048 × 512 × 1536）
NX, NY, NZ = 2048, 512, 1536
CUTOUT_MAX_POINTS = 2 ** 28  # getCutout box index-span 上限（regardless of strides）


def fetch_frame_tiled(getCutout, ds, snap, out_dir, sx, sy, sz, x_block, max_blocks):
    """沿 x 分塊抓單一 snapshot 並拼接成完整 [Nz,Ny,Nx,3]（per-block resumable）。

    getCutout box span 受 2²⁸ 限制，全 domain 超限；沿 x 切塊（每塊 box span < 限制）。
    每塊存 .partial_t{snap}/b{NN}.npz，HTTP error 中斷後只重抓未完成的塊（不重整 frame）。
    全塊齊了沿 x (axis 2) 拼接、清 partial 暫存。回 (arr, coords_dict)。
    """
    import shutil

    n_blocks = NX // x_block
    if max_blocks > 0:
        n_blocks = min(n_blocks, max_blocks)
    pdir = out_dir / f".partial_t{snap:04d}"
    pdir.mkdir(parents=True, exist_ok=True)
    for b in range(n_blocks):
        bpath = pdir / f"b{b:02d}.npz"
        if bpath.exists():
            print(f"  [skip block {b + 1}/{n_blocks}] 已存", flush=True)
            continue
        x_lo = 1 + b * x_block
        x_hi = x_lo + x_block - 1
        axes = np.array([[x_lo, x_hi], [1, NY], [1, NZ], [snap, snap]])
        strides = np.array([sx, sy, sz, 1])
        print(f"  [block {b + 1}/{n_blocks}] x index [{x_lo},{x_hi}] ...", flush=True)
        res = getCutout(ds, "velocity", axes, strides)
        da = res[f"velocity_{snap:04d}"]
        np.savez(
            bpath,
            vel=np.asarray(da.values, dtype=np.float32),       # [Nz, Ny, Nx_b, 3]
            xcoor=np.asarray(da.coords["xcoor"].values),
            ycoor=np.asarray(da.coords["ycoor"].values),
            zcoor=np.asarray(da.coords["zcoor"].values),
        )
    # 全塊齊 → 沿 x (axis 2) 拼接
    blocks: list[np.ndarray] = []
    xcoor_parts: list[np.ndarray] = []
    ycoor = zcoor = None
    for b in range(n_blocks):
        d = np.load(pdir / f"b{b:02d}.npz")
        blocks.append(d["vel"])
        xcoor_parts.append(d["xcoor"])
        if ycoor is None:
            ycoor, zcoor = d["ycoor"], d["zcoor"]
    arr = np.concatenate(blocks, axis=2)
    coords = {"xcoor": np.concatenate(xcoor_parts), "ycoor": ycoor, "zcoor": zcoor}
    shutil.rmtree(pdir)
    return arr, coords


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch JHTDB channel DNS cutout (x-tiled, resumable)")
    ap.add_argument("--out-dir", default="data/channel_dns")
    ap.add_argument("--cache-dir", default="data/jhtdb_cache")
    ap.add_argument("--x-stride", type=int, default=4)
    ap.add_argument("--y-stride", type=int, default=4)
    ap.add_argument("--z-stride", type=int, default=4)
    ap.add_argument("--t-start", type=int, default=1, help="1-indexed snapshot number")
    ap.add_argument("--t-stride", type=int, default=50, help="snapshot 間隔（CfC 時序窗跨度）")
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--x-block-size", type=int, default=256,
                    help="x-tiling 每塊 index 寬度（繞過 getCutout box-span 2²⁸ 上限；需整除 2048 與 x-stride）")
    ap.add_argument("--max-blocks", type=int, default=0, help="只抓前 N 塊（測試用）；0=全部")
    ap.add_argument("--smoke", action="store_true", help="32³ 單 frame 管線驗證（無 tiling）")
    args = ap.parse_args()

    token = os.environ.get("JHTDB_AUTH_TOKEN")
    if not token:
        sys.exit("ERROR: JHTDB_AUTH_TOKEN 未設定；先執行 `set -a; . ./.env; set +a`")

    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getCutout

    ds = turb_dataset(dataset_title="channel", output_path=args.cache_dir, auth_token=token)

    if args.smoke:
        out_dir = Path("data/channel_dns_smoke")
        frames = [args.t_start]
        strides_meta = [1, 1, 1]

        def fetch(snap):
            axes = np.array([[1, 32], [1, 32], [1, 32], [snap, snap]])
            strides = np.array([1, 1, 1, 1])
            res = getCutout(ds, "velocity", axes, strides)
            da = res[f"velocity_{snap:04d}"]
            arr = np.asarray(da.values, dtype=np.float32)
            coords = {cn: np.asarray(da.coords[cn].values) for cn in da.coords}
            return arr, coords
    else:
        # 驗證 x_block 合法（整除 + box span < 限制）
        if NX % args.x_block_size or args.x_block_size % args.x_stride:
            sys.exit(f"ERROR: x_block_size={args.x_block_size} 必須整除 NX={NX} 且被 x_stride={args.x_stride} 整除")
        span = args.x_block_size * NY * NZ
        if span >= CUTOUT_MAX_POINTS:
            sys.exit(f"ERROR: x_block box span {span:,} ≥ 上限 {CUTOUT_MAX_POINTS:,}；減小 --x-block-size")
        out_dir = Path(args.out_dir)
        frames = [args.t_start + i * args.t_stride for i in range(args.n_frames)]
        strides_meta = [args.x_stride, args.y_stride, args.z_stride]

        def fetch(snap):
            return fetch_frame_tiled(getCutout, ds, snap, out_dir, args.x_stride, args.y_stride,
                                     args.z_stride, args.x_block_size, args.max_blocks)

    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "dataset": "channel", "Re_tau": 1000, "full_grid_xyz": [NX, NY, NZ],
        "strides_xyz": strides_meta, "frames_snapshot": frames,
        "axis_order": "z,y,x,c", "components": ["u", "v", "w"],
        "domain": {"x": [0.0, float(8 * np.pi)], "y": [-1.0, 1.0], "z": [0.0, float(3 * np.pi)]},
    }

    for snap in frames:
        fpath = out_dir / f"channel_dns_t{snap:04d}.npy"
        if fpath.exists():
            print(f"[skip] {fpath.name} 已存在", flush=True)
            continue
        print(f"[fetch] snapshot {snap} ...", flush=True)
        arr, coords = fetch(snap)
        np.save(fpath, arr)
        print(f"[saved] {fpath.name} shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}", flush=True)
        if "coord_xcoor" not in meta:
            for cn, vals in coords.items():
                meta[f"coord_{cn}"] = np.asarray(vals).tolist()
            meta["shape_zyx"] = list(arr.shape[:3])
        # per-frame 寫 metadata：中斷也保留 coords（避免迴圈中被 kill 丟失）
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] {len(frames)} frames → {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
