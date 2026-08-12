import os
# Check if the required output files exist

def test_ra_plot():
    assert os.path.exists("outputs/pareto_front_decision_tree_random_search.png"), "The file pareto_front_decision_tree_random_search.png does not exist"

def test_es_plot():
    assert os.path.exists("outputs/pareto_front_decision_tree_evolutionary_search.png"), "The file pareto_front_decision_tree_evolutionary_search.png does not exist"

def test_scalarization_plot():
    assert os.path.exists("outputs/NEPS-scalarization.png"), "The file NEPS-scalarization.png does not exist"

def test_hypervolumes_plot():
    assert os.path.exists("outputs/pareto_fronts_hv.png"), "The file pareto_fronts_hv.png does not exist"

def test_observations():
    assert os.path.exists("outputs/our_observations.txt"), "The file our_observations.txt does not exist"
    with open("outputs/our_observations.txt", "r") as f:
        content = f.read().strip()
        assert content, "The file our_observations.txt exists but is empty"