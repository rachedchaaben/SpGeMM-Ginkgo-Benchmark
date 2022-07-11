find_package(OpenMP REQUIRED)
ginkgo_add_typed_benchmark_executables(sparse_blas "NO"  sparse_blas.cpp spgemm/spgemm.cpp)
target_link_libraries(sparse_blas OpenMP::OpenMP_CXX)
