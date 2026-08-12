import os
# Check if the required output files exist

def test_bo_plot_exists():
    assert os.path.exists("outputs/BO.png"), "The file BO.png does not exist"

def test_bo_comparison():
    assert os.path.exists("outputs/BO_comp.png"), "The file BO_comp.png does not exist"

def test_neps_hpo_bo():
    assert os.path.exists("outputs/NEPS-trajectory.png"), "The file NEPS-trajectory.png does not exist"

def test_observations():
    assert os.path.exists("outputs/our_observations.txt"), "The file our_observations.txt does not exist"
    with open("outputs/our_observations.txt", "r") as f:
        content = f.read().strip()
        assert content, "The file our_observations.txt exists but is empty"