"""Build the optional serial hypre 3.1.0 bridge locally (C/C++ compiler, make).

python -m axy_lgf.build_boomer [--archive /path/to/v3.1.0.tar.gz]
The archive is checksum-verified. No system packages or MPI runtime are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

PROJECT = Path(__file__).resolve().parent
VERSION = "3.1.0"
ARCHIVE_URL = f"https://github.com/hypre-space/hypre/archive/refs/tags/v{VERSION}.tar.gz"
ARCHIVE_SHA256 = "a6879ae9375d95c26afd97141d61e7a8092807333bf40cd180b385aed7351b2d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    target = PROJECT / ".deps"
    target.mkdir(exist_ok=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    bridge = target / f"libaxy_boomer{suffix}"
    if bridge.exists():
        parser.error(f"Bridge already exists: {bridge}; preserve/remove it explicitly before rebuilding")
    commands = []
    with tempfile.TemporaryDirectory(prefix="axy-hypre-") as temporary:
        work = Path(temporary)
        archive = args.archive or work / "hypre.tar.gz"
        if args.archive is None:
            urllib.request.urlretrieve(ARCHIVE_URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError("hypre archive checksum mismatch")
        with tarfile.open(archive) as bundle:
            bundle.extractall(work, filter="data")
        source = work / f"hypre-{VERSION}" / "src"
        configure = ["./configure", "--without-MPI", "--without-openmp",
                     "--disable-fortran", "--disable-shared", "--without-blas",
                     "--without-lapack", "CFLAGS=-O3 -fPIC", "CXXFLAGS=-O3 -fPIC"]
        with (target / "boomer_build.log").open("w") as log:
            for command in (configure, ["make", "-j4"]):
                commands.append(command)
                print("Running", command[0], flush=True)
                subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            includes = [source / "hypre/include", source / "utilities", source]
            # hypre's make builds a public include directory under src/hypre.
            public = source / "hypre/include"
            if not (public / "HYPRE.h").exists():
                includes.extend(source / name for name in ("IJ_mv", "parcsr_ls", "parcsr_mv", "seq_mv"))
            library = source / "lib/libHYPRE.a"
            if not library.exists():
                library = source / "hypre/lib/libHYPRE.a"
            command = ["cc", "-O3", "-std=c11", "-fPIC",
                       "-dynamiclib" if sys.platform == "darwin" else "-shared",
                       *[f"-I{path}" for path in includes],
                       str(PROJECT / "native/boomer_bridge.c"), str(library),
                       "-lc++" if sys.platform == "darwin" else "-lstdc++", "-lm", "-o", str(bridge)]
            commands.append(command)
            print("Linking BoomerAMG bridge", flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copyfile(work / f"hypre-{VERSION}" / "COPYRIGHT", target / "hypre_COPYRIGHT")
    metadata = dict(hypre_version=VERSION, archive_url=ARCHIVE_URL,
                    archive_sha256=ARCHIVE_SHA256, build="serial CPU, real float64, int32, no MPI/OpenMP/GPU",
                    linkage="hypre static library linked into the bridge", commands=commands,
                    compiler=subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
                    bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                    bridge_source_sha256=hashlib.sha256((PROJECT / "native/boomer_bridge.c").read_bytes()).hexdigest())
    (target / "boomer_build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built hypre {VERSION}: {bridge}")


if __name__ == "__main__":
    main()
