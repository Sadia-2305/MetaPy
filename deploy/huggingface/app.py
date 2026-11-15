"""Hugging Face Space entry point for MetaPy (website mode: nothing is saved on the server).

The MetaPy package can be uploaded either as a `metapy/` folder or inside a zip file (for example
MetaPy_HuggingFace_Space.zip): if the folder is missing, it is unpacked from the first zip that contains it.
"""
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("METAPY_WEB", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def _unpack_metapy() -> None:
    if (HERE / "metapy" / "__init__.py").is_file():
        return
    for zp in sorted(HERE.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            inits = sorted((n for n in names if n.endswith("metapy/__init__.py") and "legacy" not in n.lower()), key=len)
            if not inits:
                continue
            prefix = inits[0][: -len("metapy/__init__.py")]          # folder(s) above metapy/ inside the zip, if any
            for n in names:
                if not n.startswith(prefix + "metapy/") or n.endswith("/"):
                    continue
                rel = n[len(prefix):]
                target = (HERE / rel).resolve()
                if HERE not in target.parents:                         # refuse unsafe paths
                    raise ValueError(f"Unsafe path in {zp.name}: {n}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(n))
            print(f"MetaPy unpacked from {zp.name}")
            return
    raise FileNotFoundError("MetaPy package not found: upload the metapy folder or a zip that contains it.")


_unpack_metapy()
sys.path.insert(0, str(HERE))
from metapy.app import launch  # noqa: E402

launch()
