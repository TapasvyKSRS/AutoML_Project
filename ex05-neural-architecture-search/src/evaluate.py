import sys
import pickle
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from src.model_search import Genotype  # noqa: E402


def evaluate(genotype_name, genotype, benchmark_path, log_path):
    with open(benchmark_path, "rb") as f:
        benchmark = pickle.load(f)
    print("Genotype name: {}".format(genotype_name))
    print("Genotype: {}".format(genotype))
    print("Test Accuracy: {}".format(benchmark[str(genotype)]))
    # Write the accuracy to a file
    acc_log_path = log_path + "/accuracy.txt"
    with open(acc_log_path, 'w') as f:
        f.write("Genotype name: {}\n".format(genotype_name))
        f.write("Genotype: {}\n".format(genotype))
        f.write("Test Accuracy: {}\n".format(benchmark[str(genotype)]))


if __name__ == '__main__':
    genotype_name = sys.argv[1]
    benchmark_path = "src/benchmark.pkl"
    log_path = 'outputs/logs_search/{}'.format(genotype_name)
    with open(log_path+'/architecture', 'r') as f:
        genotype = f.readline()
    try:
        genotype = eval(genotype)
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)
    evaluate(genotype_name, genotype, benchmark_path, log_path)
