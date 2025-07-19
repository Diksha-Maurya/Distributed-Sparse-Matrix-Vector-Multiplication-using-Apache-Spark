SpMV-Spark: Sparse Matrix-Vector Multiplication with PySpark

Overview
This project implements sparse matrix-vector multiplication (SpMV) using Apache Spark's Python API (PySpark). The script, spmv-spark.py, reads a sparse matrix in Matrix Market (.mtx) format, performs SpMV with a randomly generated vector, and benchmarks the operation over multiple iterations. It is designed to run in Spark's local mode, leveraging parallelism across multiple cores, and is optimized for execution on a single node, such as on an ARC cluster with SLURM.

Code Explanation

Structure
The script is organized into three main functions:

1. read_coo(file_path):
   - Purpose: This function reads a sparse matrix from a Matrix Market (.mtx) file and converts it into Coordinate (COO) format, a list of (row, column, value) tuples that Spark can distribute for parallel processing. It serves as the data ingestion step, preparing the matrix for subsequent computations.
   - Process:
     - Parses header (%%MatrixMarket) and dimensions.
     - Extracts row, column, and value entries, adjusting indices (1-based to 0-based).
     - Handles symmetry (e.g., duplicates off-diagonal entries for symmetric matrices).
     - Sorts entries by row and column for consistency.
   - Returns: List of (row, col, val) tuples, number of rows, and columns.

2. spmv(coo_rdd, broadcast_x):
   - Purpose: Performs the sparse matrix-vector multiplication (SpMV) operation in a distributed manner using Spark’s RDD framework. It multiplies each non-zero matrix element by the corresponding vector element and sums results by row, producing the output vector’s components.
   - Process:
     - Maps each entry (row, col, val) to (row, val * x[col]) using broadcast vector x.
     - Reduces by row index, summing products.
   - Returns: RDD of (row, sum) pairs, where sum is the dot product of the matrix row and vector x.

3. main():
   - Purpose: Orchestrates the entire SpMV process, from Spark setup to data preparation, computation, benchmarking, and output generation. It ties together the other functions and manages the script’s execution flow.
   - Steps:
     - Configures Spark with local[1] (1 core) and 8 GB memory.
     - Reads matrix, randomizes values (seed 13), and creates an RDD.
     - Generates vector x with NumPy (seed 13).
     - Broadcasts x and performs SpMV over 10 iterations, accumulating results in y.
     - Times iterations and writes x and y to files.

Key Features
- Randomization: Matrix values are randomized between -1 and 1 for consistency with a reference implementation.
- Broadcasting: Vector x is broadcast to all tasks, minimizing data transfer.
- Benchmarking: Measures average time over 10 iterations for performance analysis.

Notes
- The script assumes a single-node setup (local mode). 
- Tested on an ARC cluster with SLURM (salloc -N 1 -n 8 -p skylake).

Usage
Run the script with spark-submit to perform SpMV and benchmark execution time. 
Example command:
spark-submit --driver-memory 8g --executor-memory 8g spmv-spark.py <Matrix file name>

Command Breakdown
- --driver-memory 8g: Allocates 8 GB to the driver’s JVM heap.
- --executor-memory 8g: Sets executor memory (relevant in cluster mode, ignored in local mode).
- spmv-spark.py: The script file.
- D6-6.mtx: Input matrix file in Matrix Market format.

-----------------------------------------------------------------------------------------------------------------------------------------

Performance Evaluation of SpMV on Different Matrices

The bar chart above presents the runtime performance of our Sparse Matrix-Vector Multiplication (SpMV) implementation using Apache Spark on various sparse matrices from the SuiteSparse collection.

Experimental Setup:
- Execution Platform: Apache Spark (submitted via spark-submit)
- Memory Allocations: 
  - Driver Memory: 8 GB
  - Executor Memory: 8 GB
- Input Format: Matrix Market (.mtx)
- Script: spmv-spark.py

Matrices Used and Their Runtime:
Matrix         | Runtime (s)
----------------|-------------
D6-6           | 0.2536
dictionary28   | 0.2667
Ga3As3H12      | 3.4136
bfly           | 0.2929
pkustk14       | 8.2371
roadNet-CA     | 6.3065

Observations:
- Despite being similar in size (~90 MB), Ga3As3H12 and pkustk14 show very different runtimes.

  - Ga3As3H12: This matrix benefits from its entries being ordered in increasing row order and including all (row, col, value) triples explicitly. The ordered structure enhances data locality during Spark’s RDD partitioning, as consecutive rows are more likely to reside in the same partition, reducing shuffling during the reduceByKey operation in spmv. The ordered nature also aids in better cache utilization on the driver and worker threads, leading to more efficient processing and a shorter runtime.

  - pkustk14, in contrast, is unordered with entries not sorted by row or column and lacks explicit values in its Matrix Market format, which increases overhead during shuffling and aggregation, leading to a longer runtime, as Spark must redistribute entries across partitions to group by row index, incurring significant network and serialization costs.

- roadNet-CA is smaller in file size (39 MB), but due to its sparse, graph-like structure and lack of value column, it still incurs significant computation time.

- Small structured matrices like D6-6, dictionary28, and bfly are processed almost instantly due to their compact size and regular patterns. They have pattern which minimize computational overhead, shuffling, and data transfer.

Conclusion:
This experiment illustrates that runtime performance is influenced not only by matrix size but also by structure, ordering, and completeness of data. Well-ordered and dense matrices yield better performance, while randomly ordered or structurally sparse matrices increase the computation and communication overhead in Spark.
