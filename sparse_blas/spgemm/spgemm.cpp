#include "spgemm.hpp"
#include <omp.h>
#include <algorithm>
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
#include "core/base/allocator.hpp"


using mtx = gko::matrix::Csr<double, int>;
using size_type = std::size_t;

// Binary search for the best split
template <typename ValueType, typename IndexType>
IndexType split_idx(IndexType* row_ptrs, ValueType num, IndexType start,
                    IndexType end)
{
    // Traverse the search space
    while (start <= end) {
        IndexType mid = (start + end) / 2;
        // case where the perfect split exists
        if (row_ptrs[mid] == num)
            return mid;
        else if (row_ptrs[mid] < num)
            start = mid + 1;
        else
            end = mid - 1;
    }
    // Return the split position for the most possible balanced split
    auto result =
        (num - row_ptrs[end] < row_ptrs[end + 1] - num) ? end : end + 1;
    return result;
}

// Binary search for the best split
template <typename ValueType, typename IndexType>
IndexType binary_search(const IndexType* row_ptrs, ValueType num,
                        IndexType start, IndexType end)
{
    // Traverse the search space
    while (start <= end) {
        IndexType mid = (start + end) / 2;
        // case where the perfect split exists
        if (row_ptrs[mid] == num)
            return mid;
        else if (row_ptrs[mid] < num)
            start = mid + 1;
        else
            end = mid - 1;
    }
    // Return the split position for the most possible balanced split
    return end + 1;
}

// generates offsets for a given matrix based on the number of recursive splits
// needed
template <typename IndexType>
void generate_offsets(std::vector<IndexType>& vect, IndexType* row_ptrs,
                      int rec_s, IndexType st, IndexType en)
{
    if ((en - st > 1) && (rec_s > 0)) {
        // Lower and upper bounds
        IndexType s = split_idx<gko::size_type, IndexType>(
            row_ptrs, (row_ptrs[en] + row_ptrs[st]) / 2, st, en);
        generate_offsets(vect, row_ptrs, rec_s - 1, st, s);
        vect.push_back(s);
        generate_offsets(vect, row_ptrs, rec_s - 1, s, en);
    }
}

// checks if there is an overlapping region between two matricies
template <typename ValueType, typename IndexType>
bool overlap(gko::matrix::Csr<ValueType, IndexType>* A,
             gko::matrix::Csr<ValueType, IndexType>* B_T, IndexType start_a,
             IndexType end_a, IndexType start_b, IndexType end_b)
{
    auto A_row_ptrs = A->get_row_ptrs();
    auto B_T_row_ptrs = B_T->get_row_ptrs();
    auto A_cols_index = A->get_col_idxs();
    auto B_T_cols_index = B_T->get_col_idxs();
    auto na = A->get_size()[0];
    auto nb = B_T->get_size()[0];
    for (int i = start_a; i < end_a; i++) {
        for (int j = start_b; j < end_b; j++) {
            if (A_cols_index[A_row_ptrs[i]] <=
                B_T_cols_index[B_T_row_ptrs[j]]) {
                if (A_cols_index[A_row_ptrs[i + 1] - 1] >=
                    B_T_cols_index[B_T_row_ptrs[j]])
                    return true;
            } else {
                if (B_T_cols_index[B_T_row_ptrs[j + 1] - 1] >=
                    A_cols_index[A_row_ptrs[i]])
                    return true;
            }
        }
    }
    return false;
}


template <typename HeapElement>
void sift_down(HeapElement* heap, typename HeapElement::index_type idx,
               typename HeapElement::index_type size)
{
    auto curcol = heap[idx].col;
    while (idx * 2 + 1 < size) {
        auto lchild = idx * 2 + 1;
        auto rchild = gko::min(lchild + 1, size - 1);
        auto lcol = heap[lchild].col;
        auto rcol = heap[rchild].col;
        auto mincol = gko::min(lcol, rcol);
        if (mincol >= curcol) {
            break;
        }
        auto minchild = lcol == mincol ? lchild : rchild;
        std::swap(heap[minchild], heap[idx]);
        idx = minchild;
    }
}

template <typename ValueType, typename IndexType>
ValueType checked_load(const ValueType* p, IndexType i, IndexType size,
                       ValueType sentinel)
{
    return i < size ? p[i] : sentinel;
}

template <typename HeapElement, typename InitCallback, typename StepCallback,
          typename ColCallback, typename IndexType>
auto hspgemm_multiway_merge(size_type row,
                            const typename HeapElement::matrix_type* a,
                            const typename HeapElement::matrix_type* b,
                            IndexType* boundaries, HeapElement* heap,
                            InitCallback init_cb, StepCallback step_cb,
                            ColCallback col_cb, size_type start_b)
    -> decltype(init_cb(0))
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_cols = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_cols = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    auto a_begin = a_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto num_rows_b = b->get_size()[0];

    using index_type = typename HeapElement::index_type;
    constexpr auto sentinel = std::numeric_limits<index_type>::max();
    // std::vector<IndexType> nnz_row_idx;
    auto state = init_cb(row);
    auto a_size = 0;
    // initialize the heap
    for (auto a_nz = a_begin; a_nz < a_end; ++a_nz) {
        auto b_row = a_cols[a_nz];
        auto b_begin = boundaries[b_row + (start_b - 1) * num_rows_b]; /**/
        auto b_end = boundaries[b_row + start_b * num_rows_b];         /**/
        heap[a_begin + a_size] = {
            b_begin, b_end, checked_load(b_cols, b_begin, b_end, sentinel),
            a_vals[a_nz]};
        a_size += (b_begin != b_end);
    }

    if (a_size != 0) {  //} a_begin != a_end) {
        // make heap:
        //     auto a_size = a_end - a_begin;
        for (auto i = (a_size - 2) / 2; i >= 0; --i) {
            sift_down(heap + a_begin, i, a_size);
        }
        auto& top = heap[a_begin];
        auto& bot = heap[a_begin + a_size - 1];
        auto col = top.col;
        while (top.col != sentinel) {
            step_cb(b_vals[top.idx] * top.val(), top.col, state);
            // move to the next element
            top.idx++;
            top.col = checked_load(b_cols, top.idx, top.end, sentinel);
            sift_down(heap + a_begin, index_type{}, a_size);
            if (top.col != col) {
                col_cb(col, state);
            }
            col = top.col;
        }
    }
    return state;
}

template <typename ValueType, typename IndexType>
void sub_rows_boundaries(gko::matrix::Csr<ValueType, IndexType>* b,
                         std::vector<IndexType> b_offsets, size_type num_rows_b,
                         IndexType* boundaries)
{
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
#pragma omp parallel for collapse(2)
    for (auto i = 0; i < b_offsets.size(); i++)
        for (auto row = 0; row < num_rows_b; row++) {
            boundaries[row + i * num_rows_b] =
                binary_search(b_col_idxs, b_offsets[i], b_row_ptrs[row],
                              b_row_ptrs[row + 1] - 1);
        }
}


template <typename ValueType, typename IndexType>
void new_spgemm(std::shared_ptr<gko::OmpExecutor> omp_exec,
                gko::matrix::Csr<ValueType, IndexType>* a,
                gko::matrix::Csr<ValueType, IndexType>* b,
                gko::matrix::Csr<ValueType, IndexType>* b_T,
                gko::matrix::Csr<ValueType, IndexType>* c,
                IndexType recursive_splits)
{
    auto num_rows = a->get_size()[0];
    auto num_cols_c = b->get_size()[1];
    auto a_row_ptrs = a->get_row_ptrs();
    auto num_rows_b = b->get_size()[0];
    auto b_T_row_ptrs = b_T->get_row_ptrs();
    auto b_row_ptrs = b->get_row_ptrs();
    auto b_cols = b->get_const_col_idxs();
    std::vector<int> a_offsets{0};
    std::vector<int> b_offsets{0};
    generate_offsets<int>(a_offsets, a_row_ptrs, recursive_splits, 0, num_rows);
    a_offsets.push_back((int)num_rows);
    generate_offsets<int>(b_offsets, b_T_row_ptrs, recursive_splits, 0,
                          num_cols_c);
    b_offsets.push_back((int)num_cols_c);

    auto c_row_ptrs = c->get_row_ptrs();

    gko::array<col_heap_element<ValueType, IndexType>> col_heap_array(
        omp_exec, a->get_num_stored_elements());

    auto col_heap = col_heap_array.get_data();

    size_type sub_row_pointers[(num_rows + 1) * b_offsets.size()];
    // first sweep: count nnz for each row
    IndexType boundaries[b_offsets.size() * num_rows_b];
    sub_rows_boundaries(b, b_offsets, num_rows_b, boundaries);
#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        auto n = 0;
        sub_row_pointers[a_row * b_offsets.size()] = n;
        for (IndexType i = 1; i < b_offsets.size(); i++) {
            n += hspgemm_multiway_merge(
                a_row, a, b, boundaries, col_heap,
                [](size_type) { return IndexType{}; },
                [](ValueType, IndexType, IndexType&) {},
                [](IndexType, IndexType& nnz) { nnz++; }, i);
            sub_row_pointers[a_row * b_offsets.size() + i] = n;
        }
        c_row_ptrs[a_row] = n;
    }
    col_heap_array.clear();

    gko::array<val_heap_element<ValueType, IndexType>> heap_array(
        omp_exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    // build row pointers
    gko::kernels::omp::components::prefix_sum(omp_exec, c_row_ptrs,
                                              num_rows + 1);
    auto new_nnz = c_row_ptrs[num_rows];

    gko::matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

#pragma omp parallel for
    for (int i = 1; i < num_rows + 1; i++) {
        for (int j = 0; j < b_offsets.size(); j++) {
            sub_row_pointers[i * b_offsets.size() + j] += c_row_ptrs[i];
        }
    }
    IndexType starts[num_rows];

#pragma omp parallel for
    for (IndexType a_row = 0; a_row < num_rows; a_row++)
        starts[a_row] = c_row_ptrs[a_row];
        // Calculation of all the small blocks
#pragma omp parallel for collapse(2)
    for (auto row_idx = 1; row_idx < a_offsets.size(); row_idx++) {
        for (auto col_idx = 1; col_idx < b_offsets.size(); col_idx++) {
            // checking overlapping regions :
            if (overlap<ValueType, IndexType>(
                    a, b_T, a_offsets[row_idx - 1], a_offsets[row_idx],
                    b_offsets[col_idx - 1], b_offsets[col_idx])) /**/ {
                {
                    gko::array<val_heap_element<ValueType, IndexType>>
                        heap_array(omp_exec, a->get_num_stored_elements());
                    auto heap2 = heap_array.get_data();
                    for (size_type a_row = a_offsets[row_idx - 1];
                         a_row < a_offsets[row_idx]; ++a_row) {
                        hspgemm_multiway_merge(
                            a_row, a, b, boundaries, heap2,
                            [&](size_type row) {
                                return gko::zero<ValueType>();
                            },
                            [](ValueType val, IndexType, ValueType& state) {
                                state += val;
                            },
                            [&](IndexType col, ValueType& state) {
                                auto h = a_row * b_offsets.size() + col_idx - 1;
                                c_col_idxs[sub_row_pointers[h]] = col;
                                c_vals[sub_row_pointers[h]] = state;
                                sub_row_pointers[h] += 1;
                                state = gko::zero<ValueType>();
                            },
                            col_idx);
                    }
                }
            }
        }
    }
}
template void new_spgemm<double, int>(std::shared_ptr<gko::OmpExecutor>,
                                      gko::matrix::Csr<double, int>*,
                                      gko::matrix::Csr<double, int>*,
                                      gko::matrix::Csr<double, int>*,
                                      gko::matrix::Csr<double, int>*, int);
template void new_spgemm<float, int>(std::shared_ptr<gko::OmpExecutor>,
                                     gko::matrix::Csr<float, int>*,
                                     gko::matrix::Csr<float, int>*,
                                     gko::matrix::Csr<float, int>*,
                                     gko::matrix::Csr<float, int>*, int);
template void new_spgemm<std::complex<double>, int>(
    std::shared_ptr<gko::OmpExecutor>,
    gko::matrix::Csr<std::complex<double>, int>*,
    gko::matrix::Csr<std::complex<double>, int>*,
    gko::matrix::Csr<std::complex<double>, int>*,
    gko::matrix::Csr<std::complex<double>, int>*, int);
template void new_spgemm<std::complex<float>, int>(
    std::shared_ptr<gko::OmpExecutor>,
    gko::matrix::Csr<std::complex<float>, int>*,
    gko::matrix::Csr<std::complex<float>, int>*,
    gko::matrix::Csr<std::complex<float>, int>*,
    gko::matrix::Csr<std::complex<float>, int>*, int);