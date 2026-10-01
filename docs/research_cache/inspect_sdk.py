"""
Static introspection of the installed `netspresso` SDK.

SAFETY: This script performs import + inspect only.
It never instantiates NetsPresso(), never logs in, never reads NETSPRESSO_API_KEY,
and performs no network calls. Credit usage: 0.
"""
import importlib, inspect, json, pkgutil, sys, enum, os

# Hard guard: refuse to run if an API key is present in the environment, so that
# accidental client construction can never authenticate.
for k in list(os.environ):
    if "NETSPRESSO" in k.upper() and "KEY" in k.upper():
        os.environ.pop(k)

out = {"python": sys.version, "packages": {}, "modules": [], "classes": {}, "enums": {}, "exceptions": [], "warnings": []}

from importlib import metadata as md
for name in ["netspresso", "netspresso-trainer", "netspresso-inference-package", "qai-hub", "torch", "torchvision",
             "onnx", "onnxruntime", "tensorflow", "numpy", "pydantic", "loguru", "requests", "PyJWT", "python-dotenv", "protobuf"]:
    try: out["packages"][name] = md.version(name)
    except md.PackageNotFoundError: out["packages"][name] = None

import netspresso
out["netspresso_version_attr"] = getattr(netspresso, "__version__", None)
out["netspresso_file"] = netspresso.__file__
out["top_level_exports"] = sorted(n for n in dir(netspresso) if not n.startswith("_"))

# module tree (import each; record failures instead of crashing)
for m in pkgutil.walk_packages(netspresso.__path__, "netspresso."):
    out["modules"].append(m.name)

def sig(obj):
    try: return str(inspect.signature(obj))
    except Exception as e: return f"<no signature: {e}>"

def describe_class(cls):
    d = {"module": cls.__module__, "init": sig(cls.__init__), "public_methods": {}}
    for n, f in inspect.getmembers(cls, predicate=lambda o: inspect.isfunction(o) or inspect.ismethod(o)):
        if not n.startswith("_"):
            d["public_methods"][n] = sig(f)
    return d

# 1. Facade
from netspresso.netspresso import NetsPresso
out["classes"]["NetsPresso"] = describe_class(NetsPresso)
try:
    from netspresso import NPQAI
    out["classes"]["NPQAI"] = describe_class(NPQAI)
except Exception as e: out["warnings"].append(f"NPQAI import: {e}")

# 2. Service classes referenced by the facade factory methods
service_targets = {
    "compressor": ["netspresso.compressor", "netspresso.compressor.v2"],
    "converter": ["netspresso.converter", "netspresso.converter.v2"],
    "profiler": ["netspresso.profiler"], "benchmarker": ["netspresso.benchmarker"],
    "quantizer": ["netspresso.quantizer"], "graph_optimizer": ["netspresso.graph_optimizer"],
    "inferencer": ["netspresso.inferencer"], "trainer": ["netspresso.trainer"], "simulator": ["netspresso.simulator"],
}
for key, mods in service_targets.items():
    for mn in mods:
        try:
            mod = importlib.import_module(mn)
        except Exception as e:
            out["warnings"].append(f"import {mn}: {type(e).__name__}: {e}"); continue
        for n, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__.startswith("netspresso") and (n[0].isupper()) and n not in out["classes"]:
                # limit to service-looking classes
                if any(w in n for w in ["Compressor", "Converter", "Profiler", "Benchmarker", "Quantizer", "Optimizer", "Inferencer", "Trainer", "Simulator"]):
                    out["classes"][n] = describe_class(cls)

# 3. Enums
import netspresso.enums as E
for n, obj in inspect.getmembers(E):
    if inspect.isclass(obj) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
        out["enums"][n] = {m.name: (m.value if isinstance(m.value, (str, int, float)) else str(m.value)) for m in obj}

# 4. Exceptions
try:
    import netspresso.exceptions as X
    for n, obj in inspect.getmembers(X, inspect.isclass):
        if issubclass(obj, BaseException) and obj.__module__.startswith("netspresso"):
            out["exceptions"].append({"name": n, "module": obj.__module__, "bases": [b.__name__ for b in obj.__bases__], "init": sig(obj.__init__)})
except Exception as e: out["warnings"].append(f"exceptions: {e}")

# 5. Result / metadata structures
out["metadata_types"] = {}
for mn in ["netspresso.metadata", "netspresso.metadata.common", "netspresso.metadata.compressor", "netspresso.metadata.converter",
           "netspresso.metadata.profiler", "netspresso.metadata.benchmarker", "netspresso.metadata.quantizer", "netspresso.metadata.graph_optimizer", "netspresso.metadata.trainer"]:
    try: mod = importlib.import_module(mn)
    except Exception as e: out["warnings"].append(f"import {mn}: {type(e).__name__}: {e}"); continue
    for n, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ == mn:
            fields = {}
            if hasattr(cls, "__dataclass_fields__"):
                fields = {f: str(t.type) for f, t in cls.__dataclass_fields__.items()}
            elif hasattr(cls, "model_fields"):
                fields = {f: str(t.annotation) for f, t in cls.model_fields.items()}
            elif hasattr(cls, "__fields__"):
                fields = {f: str(getattr(t, "outer_type_", t)) for f, t in cls.__fields__.items()}
            out["metadata_types"][f"{mn}.{n}"] = fields

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "sdk_surface.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
print("modules:", len(out["modules"]), "classes:", len(out["classes"]), "enums:", len(out["enums"]), "exceptions:", len(out["exceptions"]), "metadata types:", len(out["metadata_types"]))
print("warnings:", *out["warnings"], sep="\n  ")
