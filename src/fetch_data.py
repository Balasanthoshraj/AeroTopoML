"""Download the SELTO disc_simple archives (Zenodo record 7781392), verify md5, extract safely.

Usage: python -I src/fetch_data.py data
Each archive is extracted into its own empty directory under <root>/raw/ because the contents are untrusted data.
"""
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/records/7781392/files"
MD5 = {  # from the Zenodo record metadata
    "disc_simple_train": "560c82dae34b8f1e0b92c93c43e96017",
    "disc_simple_test": "d80f1c03327a5138753eb1ce6d09ee05",
    # test splits of the other subsets: 6 MB for real examples of other shapes and several loads (their train splits are 113 MB)
    "disc_complex_test": "d198e5171a5ec48014ef11d91d14461c",
    "sphere_simple_test": "d8e9b9c302cb05fc4984245843e5866c",
    "sphere_complex_test": "1a20ff728b731897f31daae29ea9fdc2",
}


def main(root: Path):
    for name, md5 in MD5.items():
        archive = root / "downloads" / f"{name}.tar.gz"
        out = root / "raw" / name
        if out.exists():
            print(name, "already extracted")
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            with urllib.request.urlopen(f"{BASE}/{archive.name}?download=1", timeout=60) as r, open(archive, "wb") as f:
                shutil.copyfileobj(r, f)
        got = hashlib.md5(archive.read_bytes()).hexdigest()
        if got != md5:
            archive.unlink()  # corrupt or partial download: drop it so a rerun fetches again
            raise SystemExit(f"{archive.name}: md5 {got} != {md5}")
        tmp = out.with_name(name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with tarfile.open(archive) as t:
            t.extractall(tmp, filter="data")  # 'data' filter rejects absolute paths and path traversal
        tmp.rename(out)  # rename last, so an interrupted extraction is never mistaken for a finished one
        print(name, "downloaded, md5 ok, extracted")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
