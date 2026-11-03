"""Keep public imports available without importing unrelated optional methods."""

from importlib import import_module


def lazy_exports(namespace, exports):
    """Create package export names and lazy module attribute hooks.

    :param namespace: Package globals dictionary into which loaded exports are cached.
    :param exports: Mapping from exported names to ``(module_path, attribute_name)`` pairs.
    :returns: Tuple ``(export_names, getattr_hook, dir_hook)``.
    """

    def get_attribute(name):
        """Import a declared export on first attribute access and cache it in the package namespace.

        :param name: Declared package export name to resolve.
        :returns: Resolved exported attribute; unknown names raise ``AttributeError``.
        """
        if name not in exports:
            raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
        module, attribute = exports[name]
        value = getattr(import_module(module, namespace["__name__"]), attribute)
        namespace[name] = value
        return value

    def list_attributes():
        """Combine already loaded package names and declared lazy export names.

        :returns: Sorted list of available attribute names.
        """
        return sorted(set(namespace) | set(exports))

    return list(exports), get_attribute, list_attributes
