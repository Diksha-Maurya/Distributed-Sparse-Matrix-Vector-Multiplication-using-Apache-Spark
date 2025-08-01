SpMV-Spark: Sparse Matrix-Vector Multiplication with PySpark

This project implements sparse matrix-vector multiplication (SpMV) using Apache Spark's Python API (PySpark). The script, spmv-spark.py, reads a sparse matrix in Matrix Market (.mtx) format, performs SpMV with a randomly generated vector, and benchmarks the operation over multiple iterations. It is designed to run in Spark's local mode, leveraging parallelism across multiple cores, and is optimized for execution on a single node, such as on an ARC cluster with SLURM.
