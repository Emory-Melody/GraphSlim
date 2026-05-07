from graphslim.reduction import get_method_spec, list_methods


def test_registry_resolves_public_aliases():
    assert get_method_spec("algebraic_JC").class_name == "AlgebraicJc"
    assert get_method_spec("affinity_GS").class_name == "AffinityGs"
    assert get_method_spec("tspanner").class_name == "TSpanner"


def test_registry_groups_reduction_families():
    condensation = set(list_methods("condensation"))
    coarsening = set(list_methods("coarsening"))
    sparsification = set(list_methods("sparsification"))

    assert {"gcond", "doscond", "gecc"} <= condensation
    assert {"variation_edges", "vng", "clustering"} <= coarsening
    assert {"kcenter", "random_edge", "t_spanner"} <= sparsification
