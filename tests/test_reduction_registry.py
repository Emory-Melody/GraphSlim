from graphslim.reduction import get_method_spec, list_methods


def test_current_roster_has_49_methods_and_retains_doscondx():
    import pytest
    import graphslim.condensation as condensation

    assert len(list_methods()) == 49
    assert {
        family: len(list_methods(family))
        for family in ("sparsification", "coarsening", "condensation")
    } == {
        "sparsification": 23,
        "coarsening": 10,
        "condensation": 16,
    }
    assert get_method_spec("doscondx").load_class() is condensation.DosCondX
    assert not hasattr(condensation, "GCondX")
    with pytest.raises(ValueError):
        get_method_spec("GCONDX")


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


def test_edge_sparsifiers_are_flagged_as_edge_reduction():
    edge = {name for name in list_methods() if get_method_spec(name).edge_reduction}
    assert edge == {
        "random_edge",
        "spanning_forest",
        "g_spar",
        "local_degree",
        "rank_degree",
        "scan",
        "t_spanner",
        "simmelian_parametric",
        "simmelian_non_parametric",
        "quadrilateral_simmelian",
        "simmelian_multiscale",
        "degree_multiscale",
        "local_similarity",
        "multiscale",
        "random_node_edge",
        "forest_fire",
        "triangle",
        "algebraic_distance",
    }
    assert not get_method_spec("kcenter").edge_reduction
    assert not get_method_spec("gcond").edge_reduction
