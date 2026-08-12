from pathlib import Path
from src.nas_cifar10 import NASCifar10

# Ensure 'logs' directory exists
log_dir = Path("outputs/logs")
log_dir.mkdir(parents= True, exist_ok=True)

# Open log file for writing
with open(log_dir / "log.txt", "w") as log_file:
    # Load the benchmark
    b = NASCifar10()
    cs = b.get_configuration_space()

    # Sample one random configuration/architecture
    cs.seed(43)
    config = cs.sample_configuration()

    # Query validation error and runtime
    y, cost = b.objective_function(config)

    # Write results to log file
    print("Writing results to outputs/logs/log.txt")
    log_file.write(f"Numpy representation: {config.get_array()}\n")
    log_file.write(f"Dict representation: {config.get_dictionary()}\n")
    log_file.write(f"Validation error: {y}\n")
    log_file.write(f"Runtime: {cost}\n")
