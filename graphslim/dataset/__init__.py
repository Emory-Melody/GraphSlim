"""Public dataset API, imported on demand."""

from graphslim._lazy import lazy_exports

__all__, __getattr__, __dir__ = lazy_exports(
    globals(),
    {
        "perturb_graph": (".noise", "perturb_graph"),
        "get_dataset": (".loader", "get_dataset"),
        "OgbDataLoader": (".loader", "OgbDataLoader"),
        "LargeDataLoader": (".loader", "LargeDataLoader"),
        "save_reduced": (".utils", "save_reduced"),
        "load_reduced": (".utils", "load_reduced"),
        "csr2ei": (".utils", "csr2ei"),
        "ei2csr": (".utils", "ei2csr"),
        "get_syn_data": (".utils", "get_syn_data"),
        "sparsify": (".utils", "sparsify"),
        "attack": (".attack", "attack"),
    },
)
