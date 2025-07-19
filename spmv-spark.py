import sys
import time
import numpy as np
import random
from pyspark import SparkContext, SparkConf, StorageLevel

def read_coo(file_path):
    
    entries = []
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
        if not header_line.startswith("%%MatrixMarket"):
            raise ValueError("Invalid MatrixMarket header.")
        header_parts = header_line.split()
        if len(header_parts) < 5:
            raise ValueError("Invalid MatrixMarket header.")
        object_type = header_parts[1].lower()
        format_type = header_parts[2].lower()
        field_type = header_parts[3].lower()
        symmetry   = header_parts[4].lower()
        
        line = f.readline().strip()
        while line.startswith('%') or line == "":
            line = f.readline().strip()
        dims = line.split()
        if len(dims) < 3:
            raise ValueError("Invalid dimensions line.")
        num_rows = int(dims[0])
        num_cols = int(dims[1])
        num_nonzeros_initial = int(dims[2])
        
        for _ in range(num_nonzeros_initial):
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            if field_type == "pattern":
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
                val = 1.0
            else:
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
                val = float(parts[2])
            entries.append((row, col, val))
        
        if symmetry == "symmetric":
            symmetric_entries = []
            for (row, col, val) in entries:
                symmetric_entries.append((row, col, val))
                if row != col:
                    symmetric_entries.append((col, row, val))
            entries = symmetric_entries
        
        entries.sort(key=lambda x: (x[0], x[1]))
        
    return entries, num_rows, num_cols

def spmv(coo_rdd, broadcast_x):
    products = coo_rdd.map(lambda t: (t[0], t[2] * broadcast_x.value[t[1]]))
    return products.reduceByKey(lambda a, b: a + b)

def main():
    if len(sys.argv) < 2:
        print("Usage: spmv-spark.py [matrix_file] [num_cores (optional)]")
        sys.exit(1)
    
    matrix_file = sys.argv[1]
    
    # if len(sys.argv) >= 3:
    #     cores = sys.argv[2]
    #     master = f"local[{cores}]"
    # else:
    #     master = "local[*]"
    
    conf = SparkConf().setAppName("SpMV").setMaster("local[1]") \
                  .set("spark.driver.memory", "8g") \
                  .set("spark.executor.memory", "8g")
    conf.set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    print(f"Using {sc.defaultParallelism} cores.")
    
    entries, num_rows, num_cols = read_coo(matrix_file)
    if not entries:
        print("No valid data found in the matrix file.")
        sc.stop()
        sys.exit(1)
    
    random.seed(13)
    randomized_entries = [(row, col, 1.0 - 2.0 * random.random()) for (row, col, val) in entries]
    entries = randomized_entries
    
    coo_rdd = sc.parallelize(entries)
    
    np.random.seed(13)
    x_length = num_cols
    x = np.random.rand(x_length).astype(np.float32)
    
    broadcast_x = sc.broadcast(x)
    
    _ = spmv(coo_rdd, broadcast_x).count()
    
    num_iterations = 10
    print(f"Performing {num_iterations} iterations for benchmarking...")
    
    y_accumulated = [0.0] * num_rows
    start_time = time.time()
    for _ in range(num_iterations):
        partial_result = spmv(coo_rdd, broadcast_x).collect()
        for row, val in partial_result:
            y_accumulated[row] += val
    end_time = time.time()

    #coo_rdd.unpersist()
    
    avg_time = (end_time - start_time) / num_iterations
    print("Average time per iteration: {:.4f} seconds".format(avg_time))
    
    with open("test_x.txt", "w") as fx:
        for val in x:
            fx.write(f"{val}\n")
    
    with open("test_y.txt", "w") as fy:
        for v in y_accumulated:
            fy.write(f"{v}\n")
    # print("x")
    # print(x)
    #print("\nFinal vector y values (after accumulation):")
    # for i, v in enumerate(y_accumulated):
        # print(f"Row {i}: {v}")
    
    sc.stop()

if __name__ == "__main__":
    main()
