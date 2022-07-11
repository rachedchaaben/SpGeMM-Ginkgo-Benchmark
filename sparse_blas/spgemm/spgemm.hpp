#ifndef SPGEMM_HPP
#define SPGEMM_HPP
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <core/components/prefix_sum_kernels.hpp>
#include <core/matrix/csr_builder.hpp>
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using mtx = gko::matrix::Csr<double, int>;
using size_type = std::size_t;

// Binary search for the best split
template <typename ValueType, typename IndexType>
IndexType split_idx(IndexType* row_ptrs, ValueType num, IndexType start,
                    IndexType end);

// Binary search for the best split
template <typename ValueType, typename IndexType>
IndexType binary_search(const IndexType* row_ptrs, ValueType num,
                        IndexType start, IndexType end);
// generates offsets for a given matrix based on the number of recursive splits
// needed

template <typename IndexType>
void generate_offsets(std::vector<IndexType>& vect, IndexType* row_ptrs,
                      int rec_s, IndexType st, IndexType en);

// checks if there is an overlapping region between two matricies
template <typename ValueType, typename IndexType>
bool overlap(gko::matrix::Csr<ValueType, IndexType>* A,
             gko::matrix::Csr<ValueType, IndexType>* B_T, IndexType start_a,
             IndexType end_a, IndexType start_b, IndexType end_b);


template <typename ValueType, typename IndexType>
struct col_heap_element {
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = gko::matrix::Csr<ValueType, IndexType>;

    IndexType idx;
    IndexType end;
    IndexType col;

    ValueType val() const { return gko ::zero<ValueType>(); }

    col_heap_element(IndexType idx, IndexType end, IndexType col, ValueType)
        : idx{idx}, end{end}, col{col}
    {}
};


template <typename ValueType, typename IndexType>
struct val_heap_element {
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = gko::matrix::Csr<ValueType, IndexType>;

    IndexType idx;
    IndexType end;
    IndexType col;
    ValueType val_;

    ValueType val() const { return val_; }

    val_heap_element(IndexType idx, IndexType end, IndexType col, ValueType val)
        : idx{idx}, end{end}, col{col}, val_{val}
    {}
};


template <typename HeapElement>
void sift_down(HeapElement* heap, typename HeapElement::index_type idx,
               typename HeapElement::index_type size);

template <typename ValueType, typename IndexType>
ValueType checked_load(const ValueType* p, IndexType i, IndexType size,
                       ValueType sentinel);

template <typename HeapElement, typename InitCallback, typename StepCallback,
          typename ColCallback, typename IndexType>
auto hspgemm_multiway_merge(size_type row,
                            const typename HeapElement::matrix_type* a,
                            const typename HeapElement::matrix_type* b,
                            IndexType* boundaries, HeapElement* heap,
                            InitCallback init_cb, StepCallback step_cb,
                            ColCallback col_cb, size_type start_b)
    -> decltype(init_cb(0));

template <typename ValueType, typename IndexType>
void sub_rows_boundaries(gko::matrix::Csr<ValueType, IndexType>* b,
                         std::vector<IndexType> b_offsets, size_type num_rows_b,
                         IndexType* boundaries);

template <typename ValueType, typename IndexType>
void new_spgemm(std::shared_ptr<gko::OmpExecutor> omp_exec,
                gko::matrix::Csr<ValueType, IndexType>* a,
                gko::matrix::Csr<ValueType, IndexType>* b,
                gko::matrix::Csr<ValueType, IndexType>* b_T,
                gko::matrix::Csr<ValueType, IndexType>* c,
                IndexType recursive_splits);

#endif