#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Update the AIE4 Peano wheel pinned for medusa-dev.

The AIE4 target is distributed through the internal Artifactory repository,
not the public Xilinx/llvm-aie nightly.  Artifactory permits downloading a
known wheel anonymously, but listing candidate artifacts requires a read
token.  The scheduled workflow supplies that token through
ARTIFACTORY_TOKEN; it is never written to the requirements file or a PR.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version

ARTIFACTORY_REPOSITORY = "scc-peano-wheel"
ARTIFACTORY_BASE_URL = "https://xcoartifactory.xilinx.com/artifactory"
ARTIFACTORY_LIST_URL = (
    f"{ARTIFACTORY_BASE_URL}/api/storage/{ARTIFACTORY_REPOSITORY}"
    "?list&deep=0&listFolders=0"
)
WHEEL_BASE_URL = f"{ARTIFACTORY_BASE_URL}/{ARTIFACTORY_REPOSITORY}"
REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = REPO_ROOT / "utils" / "aie4-peano-requirements.txt"

# The NPU3 runner is Linux x86_64 and currently needs the glibc 2.38 wheel.
# Keep the platform explicit: selecting another manylinux wheel can make the
# wheel appear installable in the update job while failing on the hardware
# runner that validates the resulting PR.
WHEEL_RE = re.compile(
    r"^/?(?P<filename>llvm_aie-(?P<version>[^-]+)"
    r"-py3-none-manylinux_2_38_x86_64\.whl)$"
)
PIN_RE = re.compile(r"^" + re.escape(WHEEL_BASE_URL) + r"/(?P<filename>\S+)$", re.MULTILINE)


def get_request(url):
    token = os.environ.get("ARTIFACTORY_TOKEN")
    if not token:
        sys.exit(
            "error: ARTIFACTORY_TOKEN is required to list "
            f"{ARTIFACTORY_REPOSITORY}. Configure it as an Actions secret."
        )
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=60)


def fetch_wheels():
    """Return compatible Artifactory wheel filenames grouped by version."""
    with get_request(ARTIFACTORY_LIST_URL) as response:
        payload = json.load(response)

    wheels = {}
    for entry in payload.get("files", []):
        match = WHEEL_RE.match(entry.get("uri", ""))
        if match:
            wheels[match.group("version")] = match.group("filename")

    if not wheels:
        sys.exit(
            "error: no manylinux_2_38_x86_64 llvm_aie wheels found at "
            f"{ARTIFACTORY_LIST_URL}."
        )
    return wheels


def read_current():
    match = PIN_RE.search(REQUIREMENTS_FILE.read_text())
    if not match:
        sys.exit(
            f"error: no Artifactory AIE4 Peano wheel URL found in "
            f"{REQUIREMENTS_FILE.relative_to(REPO_ROOT)}."
        )
    return match.group("filename")


def write_current(filename):
    text = REQUIREMENTS_FILE.read_text()
    replacement = f"{WHEEL_BASE_URL}/{filename}"
    new_text, count = PIN_RE.subn(replacement, text)
    if count != 1:
        sys.exit(
            f"error: expected exactly one Artifactory AIE4 Peano wheel URL in "
            f"{REQUIREMENTS_FILE.relative_to(REPO_ROOT)}, found {count}."
        )
    REQUIREMENTS_FILE.write_text(new_text)


def write_output(**kwargs):
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a") as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identify-only",
        action="store_true",
        help="Report the selected wheel without editing the pin.",
    )
    parser.add_argument(
        "--peano-version",
        default="",
        help="Pin this exact PEP 440 wheel version instead of the newest one.",
    )
    args = parser.parse_args()

    current = read_current()
    wheels = fetch_wheels()
    explicit = args.peano_version.strip()
    if explicit:
        target = wheels.get(explicit)
        if not target:
            sys.exit(
                f"error: no compatible AIE4 Peano wheel for {explicit} in "
                f"{ARTIFACTORY_REPOSITORY}."
            )
    else:
        target = wheels[max(wheels, key=Version)]

    print(f"current: {current}")
    print(f"target:  {target}")
    if target == current:
        print("Already up to date.")
        write_output(target_wheel=target, changes="false")
        return

    write_output(
        target_wheel=target,
        changes="true",
        bump_reason=f"{current} -> {target}",
    )
    if args.identify_only:
        return

    write_current(target)
    print(f"Wrote {REQUIREMENTS_FILE.relative_to(REPO_ROOT)}: {target}")


if __name__ == "__main__":
    main()
