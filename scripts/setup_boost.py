import hashlib
from pathlib import Path
import tarfile
import urllib.request

root = Path(__file__).resolve().parents[1]
external = root / "external"
archive = external / "boost_1_88_0.tar.bz2"
expected = "46d9d2c06637b219270877c9e16155cbd015b6dc84349af064c088e9b5b12f7b"
url = "https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.bz2"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    external.mkdir(exist_ok=True)
    header = external / "boost_1_88_0/boost/version.hpp"
    if header.exists() and "#define BOOST_VERSION 108800" in header.read_text():
        print("Boost 1.88.0 headers already present.")
        return
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if checksum(partial) != expected:
            raise RuntimeError("Boost archive SHA-256 mismatch.")
        partial.replace(archive)
    if checksum(archive) != expected:
        raise RuntimeError("Boost archive SHA-256 mismatch.")
    with tarfile.open(archive, "r:bz2") as source:
        members = [m for m in source.getmembers() if
                   m.name.startswith("boost_1_88_0/boost/") or
                   m.name in {"boost_1_88_0/boost", "boost_1_88_0/LICENSE_1_0.txt"}]
        source.extractall(external, members=members, filter="data")
    print("Boost 1.88.0 headers ready.")


if __name__ == "__main__":
    main()
