"""Keep public imports available without importing unrelated optional methods."""

from importlib import import_module


def lazy_exports(namespace, exports):
    def get_attribute(name):
        if name not in exports:
            raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
        module, attribute = exports[name]
        value = getattr(import_module(module, namespace["__name__"]), attribute)
        namespace[name] = value
        return value

    def list_attributes():
        return sorted(set(namespace) | set(exports))

    return list(exports), get_attribute, list_attributes
