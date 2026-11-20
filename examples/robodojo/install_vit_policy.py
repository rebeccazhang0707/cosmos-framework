"""Register the versioned joint/ViT adapters in a separate RoboDojo checkout."""

import argparse
import json
import os
from pathlib import Path


def install(bench_root: Path) -> dict:
    bench_root = bench_root.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    policy_root = bench_root / "XPolicyLab/policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"Initialize the RoboDojo XPolicyLab submodule first: {policy_root}")
    sources = {
        "Cosmos3_Edge_RoboDojo": repo / "examples/robodojo/joint_policy",
        "Cosmos3_Edge_RoboDojo_ViT": repo / "examples/robodojo/reasoner_vit/policy",
    }
    pending = []
    for name, source in sources.items():
        if not all((source / file).is_file() for file in ("__init__.py", "model.py", "deploy.py", "deploy.yml")):
            raise FileNotFoundError(f"Incomplete adapter source: {source}")
        destination = policy_root / name
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Keep or relocate the existing adapter before installing: {destination}")
        pending.append((source, destination))

    client = bench_root / "src/eval_client/main.py"
    original = client.read_text()
    old = "parser = argparse.ArgumentParser()"
    new = "parser = argparse.ArgumentParser(allow_abbrev=False)"
    if new in original:
        patched = original
    elif original.count(old) == 1:
        patched = original.replace(old, new, 1)
    else:
        raise ValueError(f"Unexpected client argument parser; review patches/eval_client_no_abbrev.patch: {client}")

    # Preflight both adapters and the native client before changing the checkout.
    if patched != original:
        client.write_text(patched)
    for source, destination in pending:
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
    return {
        "bench_root": str(bench_root),
        "installed_links": [str(destination) for _, destination in pending],
        "client_patched": patched != original,
        "policy_name": "Cosmos3_Edge_RoboDojo_ViT",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-root", type=Path, required=True, help="Parent workspace containing XPolicyLab and env_cfg"
    )
    args = parser.parse_args()
    print(json.dumps(install(args.bench_root), indent=2))


if __name__ == "__main__":
    main()
