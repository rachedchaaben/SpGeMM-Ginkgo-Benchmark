# SpGeMM-Ginko-Benchmark
In this repository we implement an OpenMP SpGeMM kernel and we detail the instructions to perform benchmarks using Benchmarking implemented in Ginkgo.

Ginkgo is a high-performance linear algebra library for manycore systems. SpGeMM is a linear algebra kernel that we have worked on and we implemented an openMP version of it. 

To benchmark this new SpGeMM you need to follow those instruction.    
1- Clone the Ginkgo repository : git clone https://github.com/ginkgo-project/ginkgo.git  
2- replace the sparse_blas directory in ginkgo : (/ginkgo/benchmark/sparse_blas) with the sparse_spla folder added in this repository.   
3- Follow the building instruction in Ginkgo   
 
To run the benchmark executable, the input should be given by a JSON file consisting of matrix file name 
```sh
[{"filename":"path/to/file.mtx"}, {"filename":"path/to/file2.mtx"}]
``` 
saved to file (e.g. data.JSON). 
then from the build folder your run : 
``` sh
cat data.JSON | benchmark/sparse_blas/sparse_blas -operations new_spgemm -executor omp -detailed
```
new_spgemm is the option to run the SpGeMM implemented in this repository and spgemm is to run the one implemented in Ginkgo. 

