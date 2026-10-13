"""Public condensation API, imported on demand."""

from graphslim._lazy import lazy_exports

__all__, __getattr__, __dir__ = lazy_exports(
    globals(),
    {
        "CGC": (".cgc", "CGC"),
        "ClustGDD": (".clustgdd", "ClustGDD"),
        "GCGP": (".gcgp", "GCGP"),
        "DosCond": (".doscond", "DosCond"),
        "DosCondX": (".doscondx", "DosCondX"),
        "GCond": (".gcond", "GCond"),
        "GCondX": (".gcondx", "GCondX"),
        "GCSNTK": (".gcsntk", "GCSNTK"),
        "GEOM": (".geom", "GEOM"),
        "MSGC": (".msgc", "MSGC"),
        "SFGC": (".sfgc", "SFGC"),
        "SGDD": (".sgdd", "SGDD"),
        "SimGC": (".simgc", "SimGC"),
        "GCDM": (".gcdm", "GCDM"),
        "GCDMX": (".gcdmx", "GCDMX"),
        "GDEM": (".gdem", "GDEM"),
        "GECC": (".gecc", "GECC"),
    },
)
