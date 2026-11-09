#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把一份 policy.npz 打包成 .gmod 模型包。

    python scripts/pack_gmod.py --policy <policy.npz> --id release \
        --name "Release 发布模型" --description "..." --out models/release.gmod

.gmod 是一个 zip，只包含两个成员：

    meta.json    id / name / obsDim / actDim / hidden / version / createdAt
                 description / color / sha256 / training(可选)
    policy.npz   与 training/gladiator_train/policy.py 保存的权重同构

meta.json 里的 sha256 是 policy.npz 字节的摘要，dashboard 扫描时会校验。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone

META_NAME = "meta.json"
POLICY_NAME = "policy.npz"


def read_policy_meta(path: str) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        if "_meta" not in data:
            raise SystemExit("policy.npz 缺少 _meta：请用 training/gladiator_train/policy.py 保存")
        return json.loads(str(data["_meta"]))


def build_meta(args, policy_meta: dict, policy_bytes: bytes) -> dict:
    meta = {
        "id": args.id,
        "name": args.name or args.id,
        "description": args.description or "",
        "color": args.color or None,
        "version": 1,
        "obsDim": int(policy_meta["obs_dim"]),
        "actDim": int(policy_meta["n_actions"]),
        "hidden": int(policy_meta["hidden"]),
        "updates": int(policy_meta.get("updates", 0)),
        "sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "sizeBytes": len(policy_bytes),
        "createdAt": args.created_at
        or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "packedBy": "scripts/pack_gmod.py",
    }
    if args.training:
        with io.open(args.training, "r", encoding="utf-8") as handle:
            meta["training"] = json.load(handle)
    return meta


def write_gmod(out_path: str, meta: dict, policy_bytes: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(META_NAME, json.dumps(meta, ensure_ascii=False, indent=2))
        archive.writestr(POLICY_NAME, policy_bytes)
    os.replace(tmp, out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="把 policy.npz 打包为 .gmod")
    parser.add_argument("--policy", required=True, help="policy.npz 路径")
    parser.add_argument("--id", required=True, help="模型 id（决定列表里的 gmod:<id>）")
    parser.add_argument("--name", default=None, help="显示名")
    parser.add_argument("--description", default=None, help="说明")
    parser.add_argument("--color", default=None, help="可选颜色（如 §b）")
    parser.add_argument("--training", default=None, help="可选 model.json，写入 meta.training")
    parser.add_argument("--created-at", default=None, help="可选创建时间（默认当前时间）")
    parser.add_argument("--out", default=None, help="输出路径，默认 models/<id>.gmod")
    args = parser.parse_args(argv)

    if not os.path.exists(args.policy):
        raise SystemExit("找不到 policy：%s" % args.policy)
    policy_meta = read_policy_meta(args.policy)
    with open(args.policy, "rb") as handle:
        policy_bytes = handle.read()
    meta = build_meta(args, policy_meta, policy_bytes)
    out_path = args.out or os.path.join("models", "%s.gmod" % args.id)
    write_gmod(out_path, meta, policy_bytes)
    print(json.dumps({
        "out": out_path,
        "id": args.id,
        "sha256": meta["sha256"],
        "sizeBytes": len(policy_bytes),
        "obsDim": meta["obsDim"],
        "actDim": meta["actDim"],
        "hidden": meta["hidden"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
