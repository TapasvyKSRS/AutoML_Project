import os
from pathlib import Path

def test_evaluation_pipeline():
    """
    Verifies that both the markdown table and the student observations 
    exist and are not empty. Assumes `aslib.py --eval_all` was already run.
    """
    BASE_DIR = Path(__file__).parent.parent
    outputs_dir = BASE_DIR / "outputs"
    
    # 1. Assert evaluation.md exists and is not empty
    md_path = outputs_dir / "evaluation.md"
    assert md_path.exists(), f"The file {md_path.name} does not exist. Did you run aslib.py --eval_all?"
    
    with open(md_path, "r") as f:
        md_content = f.read().strip()
        assert len(md_content) > 0, f"The file {md_path.name} exists but is empty."
        assert "SAT11-INDU" in md_content, "Markdown table is missing the expected scenario rows."
        assert "SAT11-RAND" in md_content, "Markdown table is missing the expected scenario rows."

    # 2. Assert our_observations.txt exists and is not empty
    obs_path = outputs_dir / "our_observations.txt"
    assert obs_path.exists(), f"The file {obs_path.name} does not exist in the outputs directory."
    
    with open(obs_path, "r") as f:
        obs_content = f.read().strip()
        assert len(obs_content) > 20, f"The file {obs_path.name} exists but seems too short to be a complete answer."