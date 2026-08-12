import os
# Check if the required output files exist

def test_sh_plot():
    assert os.path.exists("outputs/successive_halving_results.pdf"), "The file succesive_halving_results.pdf does not exist"

def test_hb_plot():
    assert os.path.exists("outputs/hyperband_results.pdf"), "The file hyperband_results.pdf does not exist"

def test_neps_plot():
    assert os.path.exists("outputs/neps_results.pdf"), "The file neps_results.pdf does not exist"

def test_observations():
    assert os.path.exists("outputs/our_observations.txt"), "The file our_observations.txt does not exist"
    with open("outputs/our_observations.txt", "r") as f:
        content = f.read().strip()
        assert content, "The file our_observations.txt exists but is empty"