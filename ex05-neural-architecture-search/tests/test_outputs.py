import os
# Check if the required output files exist
def test_log_txt():
    assert os.path.exists("outputs/logs/log.txt"),"The file log.txt does not exist in outputs/ or logs/ directory"

def test_plot_png():
    assert os.path.exists("outputs/logs/plot.png"), "The file plot.png does not exist in outputs/ or logs/ directory"

def test_visualizations():
    assert os.path.exists("outputs/normal_cell_darts.pdf"),"Missing file: normal_cell_darts.pdf"
    assert os.path.exists("outputs/reduction_cell_darts.pdf"), "Missing file: reduction_cell_darts.pdf"

def test_darts_log_dir():
    assert os.path.isdir("outputs/logs_search/darts"),  "Directory logs_search/darts does not exist"
