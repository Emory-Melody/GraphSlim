import csv
import subprocess
from itertools import product


def run_test(record):
    # Order of the record is changed to ensure 'filename' comes first.
    filename, dataset, params, settings = record
    # warning: the path to the python executable may vary
    command = f"/public/gsb/anaconda3/envs/pyg/bin/python {filename} {params} {settings} --dataset {dataset}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print('stdout:', result.stdout)
    print('stderr:', result.stderr)
    success = "ok" if result.returncode == 0 else "failed"
    return success


def write_to_csv(data):
    with open('test_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Change the header order accordingly
        writer.writerow(["pyname", "dataset", "para", "settings", "result"])
        writer.writerows(data)


if __name__ == "__main__":
    datasets = ["cora", "ogbn-arxiv"]
    params = ["--epochs 1 --hidden 16"]
    settings = ["--setting ind", "--setting trans"]
    filenames = ["train_coarsen.py", "train_coreset.py", "train_gcond.py"]

    # Change the order in the product to match the desired CSV structure.
    records = list(product(filenames, datasets, params, settings))

    results = []
    for record in records:
        success = run_test(record)
        result_record = list(record) + [success]
        results.append(result_record)
    write_to_csv(results)
