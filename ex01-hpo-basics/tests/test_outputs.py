import os
# Check if the required output files exist

def test_members_exist():
    assert os.path.exists("outputs/members.txt"), "The file members.txt does not exist"
    with open("outputs/members.txt", "r") as f:
        content = f.read().strip()
        assert content, "The file members.txt exists but is empty"

def test_neps_plot_exists():
    assert os.path.exists("outputs/neps-on-ackley.png"), "The file neps-on-ackley.png does not exist"

def test_ackley_comparison():
    assert os.path.exists("outputs/ackley-comparison.png"), "The file ackley-comparison.png does not exist"

def test_your_observations():
    assert os.path.exists("outputs/our_observations.txt"), "The file our_observations.txt does not exist"

    # load the file and check if it contains the expected content
    with open("outputs/our_observations.txt", "r") as f:
        content = f.read().strip()
        assert content, "The file our_observations.txt exists but is empty"