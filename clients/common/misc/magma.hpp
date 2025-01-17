#pragma once

#include "magma_operators.h"
#include "magma_v2.h"
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#ifdef ROCSOLVER_CLIENTS_TEST
#include <gtest/gtest.h>

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_MAGMA_ERROR2(ERROR) ASSERT_EQ(ERROR, 0)
#define CHECK_MAGMA_ERROR(ERROR) CHECK_MAGMA_ERROR2(ERROR)

#else // ROCSOLVER_CLIENTS_TEST
#define CHECK_MAGMA_ERROR(err)                                                              \
    do                                                                                      \
    {                                                                                       \
        magma_int_t err_ = (err);                                                           \
        if(err_ != 0)                                                                       \
        {                                                                                   \
            fprintf(stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", #err, __FILE__, \
                    __LINE__, (long long)err_, magma_strerror(err_));                       \
            exit(1);                                                                        \
        }                                                                                   \
    } while(0)
#endif // ROCSOLVER_CLIENTS_TEST

template <typename T>
struct rocblas2magma_type;

template <>
struct rocblas2magma_type<float>
{
    using type = float;
};

template <>
struct rocblas2magma_type<double>
{
    using type = double;
};

template <>
struct rocblas2magma_type<rocblas_complex_num<float>>
{
    using type = magmaFloatComplex;
};

template <>
struct rocblas2magma_type<rocblas_complex_num<double>>
{
    using type = magmaDoubleComplex;
};

template <typename T>
using rocblas2magma_type_t = typename rocblas2magma_type<T>::type;

inline magma_vec_t rocblas2magma_evect(rocblas_evect evect)
{
    switch(evect)
    {
    case rocblas_evect_none: return MagmaNoVec;
    case rocblas_evect_original: return MagmaVec;
    default: throw std::invalid_argument("rocblas2magma_evect cannot convert value");
    }
}

inline magma_uplo_t rocblas2magma_fill(rocblas_fill uplo)
{
    switch(uplo)
    {
    case rocblas_fill_lower: return MagmaLower;
    case rocblas_fill_upper: return MagmaUpper;
    case rocblas_fill_full: return MagmaFull;
    default: throw std::invalid_argument("rocblas2magma_fill cannot convert value");
    }
}

inline magma_range_t rocblas2magma_erange(rocblas_erange erange)
{
    switch(erange)
    {
    case rocblas_erange_all: return MagmaRangeAll;
    case rocblas_erange_value: return MagmaRangeV;
    case rocblas_erange_index: return MagmaRangeI;
    default: throw std::invalid_argument("rocblas2magma_erange cannot convert value");
    }
}

// namespace std
// {
// __host__ __device__ inline double real(const magmaDoubleComplex &x) { return ::real(x); }
// __host__ __device__ inline float  real(const magmaFloatComplex  &x) { return ::real(x); }

// __host__ __device__ inline double imag(const magmaDoubleComplex &x) { return ::real(x); }
// __host__ __device__ inline float  imag(const magmaFloatComplex  &x) { return ::real(x); }
// }

// __host__ __device__ inline operator magmaFloatComplex(float x)
// {
//     return MAGMA_C_MAKE(x, 0);
// }
// __host__ __device__ inline operator magmaDoubleComplex(double x)
// {
//     return MAGMA_Z_MAKE(x, 0);
// }

inline magma_int_t magma_malloc(magma_int_t** ptr, size_t n)
{
    return magma_imalloc(ptr, n);
}
inline magma_int_t magma_malloc(float** ptr, size_t n)
{
    return magma_smalloc(ptr, n);
}
inline magma_int_t magma_malloc(double** ptr, size_t n)
{
    return magma_dmalloc(ptr, n);
}
inline magma_int_t magma_malloc(magmaFloatComplex** ptr, size_t n)
{
    return magma_cmalloc(ptr, n);
}
inline magma_int_t magma_malloc(magmaDoubleComplex** ptr, size_t n)
{
    return magma_zmalloc(ptr, n);
}

inline magma_int_t magma_malloc_cpu(magma_int_t** ptr, size_t n)
{
    return magma_imalloc_cpu(ptr, n);
}
inline magma_int_t magma_malloc_cpu(float** ptr, size_t n)
{
    return magma_smalloc_cpu(ptr, n);
}
inline magma_int_t magma_malloc_cpu(double** ptr, size_t n)
{
    return magma_dmalloc_cpu(ptr, n);
}
inline magma_int_t magma_malloc_cpu(magmaFloatComplex** ptr, size_t n)
{
    return magma_cmalloc_cpu(ptr, n);
}
inline magma_int_t magma_malloc_cpu(magmaDoubleComplex** ptr, size_t n)
{
    return magma_zmalloc_cpu(ptr, n);
}

inline magma_int_t magma_malloc_pinned(magma_int_t** ptr, size_t n)
{
    return magma_imalloc_pinned(ptr, n);
}
inline magma_int_t magma_malloc_pinned(float** ptr, size_t n)
{
    return magma_smalloc_pinned(ptr, n);
}
inline magma_int_t magma_malloc_pinned(double** ptr, size_t n)
{
    return magma_dmalloc_pinned(ptr, n);
}
inline magma_int_t magma_malloc_pinned(magmaFloatComplex** ptr, size_t n)
{
    return magma_cmalloc_pinned(ptr, n);
}
inline magma_int_t magma_malloc_pinned(magmaDoubleComplex** ptr, size_t n)
{
    return magma_zmalloc_pinned(ptr, n);
}

/* SYEVD/HEEVD */
inline magma_int_t magma_syevd_heevd_gpu(magma_vec_t jobz,
                                         magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaFloat_ptr dA,
                                         magma_int_t ldda,
                                         float* w,
                                         float* wA,
                                         magma_int_t ldwa,
                                         float* work,
                                         magma_int_t lwork,
                                         float* rwork,
                                         magma_int_t lrwork,
                                         magma_int_t* iwork,
                                         magma_int_t liwork,
                                         magma_int_t* info)
{
    return magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
}

inline magma_int_t magma_syevd_heevd_gpu(magma_vec_t jobz,
                                         magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaDouble_ptr dA,
                                         magma_int_t ldda,
                                         double* w,
                                         double* wA,
                                         magma_int_t ldwa,
                                         double* work,
                                         magma_int_t lwork,
                                         double* rwork,
                                         magma_int_t lrwork,
                                         magma_int_t* iwork,
                                         magma_int_t liwork,
                                         magma_int_t* info)
{
    return magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
}

inline magma_int_t magma_syevd_heevd_gpu(magma_vec_t jobz,
                                         magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaFloatComplex_ptr dA,
                                         magma_int_t ldda,
                                         float* w,
                                         magmaFloatComplex* wA,
                                         magma_int_t ldwa,
                                         magmaFloatComplex* work,
                                         magma_int_t lwork,
                                         float* rwork,
                                         magma_int_t lrwork,
                                         magma_int_t* iwork,
                                         magma_int_t liwork,
                                         magma_int_t* info)
{
    return magma_cheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork,
                            liwork, info);
}

inline magma_int_t magma_syevd_heevd_gpu(magma_vec_t jobz,
                                         magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaDoubleComplex_ptr dA,
                                         magma_int_t ldda,
                                         double* w,
                                         magmaDoubleComplex* wA,
                                         magma_int_t ldwa,
                                         magmaDoubleComplex* work,
                                         magma_int_t lwork,
                                         double* rwork,
                                         magma_int_t lrwork,
                                         magma_int_t* iwork,
                                         magma_int_t liwork,
                                         magma_int_t* info)
{
    return magma_zheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork,
                            liwork, info);
}

/* SYEVDX/HEEVDX */
inline magma_int_t magma_syevdx_heevdx_gpu(magma_vec_t jobz,
                                           magma_range_t range,
                                           magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaFloat_ptr dA,
                                           magma_int_t ldda,
                                           float vl,
                                           float vu,
                                           magma_int_t il,
                                           magma_int_t iu,
                                           magma_int_t* mout,
                                           float* w,
                                           float* wA,
                                           magma_int_t ldwa,
                                           float* work,
                                           magma_int_t lwork,
                                           float* rwork,
                                           magma_int_t lrwork,
                                           magma_int_t* iwork,
                                           magma_int_t liwork,
                                           magma_int_t* info)
{
    return magma_ssyevdx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa,
                             work, lwork, iwork, liwork, info);
}

inline magma_int_t magma_syevdx_heevdx_gpu(magma_vec_t jobz,
                                           magma_range_t range,
                                           magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaDouble_ptr dA,
                                           magma_int_t ldda,
                                           double vl,
                                           double vu,
                                           magma_int_t il,
                                           magma_int_t iu,
                                           magma_int_t* mout,
                                           double* w,
                                           double* wA,
                                           magma_int_t ldwa,
                                           double* work,
                                           magma_int_t lwork,
                                           double* rwork,
                                           magma_int_t lrwork,
                                           magma_int_t* iwork,
                                           magma_int_t liwork,
                                           magma_int_t* info)
{
    return magma_dsyevdx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa,
                             work, lwork, iwork, liwork, info);
}

inline magma_int_t magma_syevdx_heevdx_gpu(magma_vec_t jobz,
                                           magma_range_t range,
                                           magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaFloatComplex_ptr dA,
                                           magma_int_t ldda,
                                           float vl,
                                           float vu,
                                           magma_int_t il,
                                           magma_int_t iu,
                                           magma_int_t* mout,
                                           float* w,
                                           magmaFloatComplex* wA,
                                           magma_int_t ldwa,
                                           magmaFloatComplex* work,
                                           magma_int_t lwork,
                                           float* rwork,
                                           magma_int_t lrwork,
                                           magma_int_t* iwork,
                                           magma_int_t liwork,
                                           magma_int_t* info)
{
    return magma_cheevdx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa,
                             work, lwork, rwork, lrwork, iwork, liwork, info);
}

inline magma_int_t magma_syevdx_heevdx_gpu(magma_vec_t jobz,
                                           magma_range_t range,
                                           magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaDoubleComplex_ptr dA,
                                           magma_int_t ldda,
                                           double vl,
                                           double vu,
                                           magma_int_t il,
                                           magma_int_t iu,
                                           magma_int_t* mout,
                                           double* w,
                                           magmaDoubleComplex* wA,
                                           magma_int_t ldwa,
                                           magmaDoubleComplex* work,
                                           magma_int_t lwork,
                                           double* rwork,
                                           magma_int_t lrwork,
                                           magma_int_t* iwork,
                                           magma_int_t liwork,
                                           magma_int_t* info)
{
    return magma_zheevdx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa,
                             work, lwork, rwork, lrwork, iwork, liwork, info);
}

/* POTRF Hybrid*/
inline magma_int_t magma_potrf_gpu(magma_uplo_t uplo,
                                   magma_int_t n,
                                   magmaFloat_ptr dA,
                                   magma_int_t ldda,
                                   magma_int_t* info)
{
    return magma_spotrf_gpu(uplo, n, dA, ldda, info);
}

inline magma_int_t magma_potrf_gpu(magma_uplo_t uplo,
                                   magma_int_t n,
                                   magmaDouble_ptr dA,
                                   magma_int_t ldda,
                                   magma_int_t* info)
{
    return magma_dpotrf_gpu(uplo, n, dA, ldda, info);
}

inline magma_int_t magma_potrf_gpu(magma_uplo_t uplo,
                                   magma_int_t n,
                                   magmaFloatComplex_ptr dA,
                                   magma_int_t ldda,
                                   magma_int_t* info)
{
    return magma_cpotrf_gpu(uplo, n, dA, ldda, info);
}

inline magma_int_t magma_potrf_gpu(magma_uplo_t uplo,
                                   magma_int_t n,
                                   magmaDoubleComplex_ptr dA,
                                   magma_int_t ldda,
                                   magma_int_t* info)
{
    return magma_zpotrf_gpu(uplo, n, dA, ldda, info);
}

/* POTRF Native*/
inline magma_int_t magma_potrf_native(magma_uplo_t uplo,
                                      magma_int_t n,
                                      magmaFloat_ptr dA,
                                      magma_int_t ldda,
                                      magma_int_t* info)
{
    return magma_spotrf_native(uplo, n, dA, ldda, info);
}

inline magma_int_t magma_potrf_native(magma_uplo_t uplo,
                                      magma_int_t n,
                                      magmaDouble_ptr dA,
                                      magma_int_t ldda,
                                      magma_int_t* info)
{
    return magma_dpotrf_native(uplo, n, dA, ldda, info);
}

inline magma_int_t magma_potrf_native(magma_uplo_t uplo,
                                      magma_int_t n,
                                      magmaFloatComplex_ptr dA,
                                      magma_int_t ldda,
                                      magma_int_t* info)
{
    return magma_cpotrf_native(uplo, n, dA, ldda, info);
}

inline magma_int_t magma_potrf_native(magma_uplo_t uplo,
                                      magma_int_t n,
                                      magmaDoubleComplex_ptr dA,
                                      magma_int_t ldda,
                                      magma_int_t* info)
{
    return magma_zpotrf_native(uplo, n, dA, ldda, info);
}

/* GEQRF */
inline magma_int_t magma_geqrf2_gpu(magma_int_t m,
                                    magma_int_t n,
                                    magmaFloat_ptr dA,
                                    magma_int_t ldda,
                                    float* tau,
                                    magma_int_t* info)
{
    return magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
}

inline magma_int_t magma_geqrf2_gpu(magma_int_t m,
                                    magma_int_t n,
                                    magmaDouble_ptr dA,
                                    magma_int_t ldda,
                                    double* tau,
                                    magma_int_t* info)
{
    return magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
}

inline magma_int_t magma_geqrf2_gpu(magma_int_t m,
                                    magma_int_t n,
                                    magmaFloatComplex_ptr dA,
                                    magma_int_t ldda,
                                    magmaFloatComplex* tau,
                                    magma_int_t* info)
{
    return magma_cgeqrf2_gpu(m, n, dA, ldda, tau, info);
}

inline magma_int_t magma_geqrf2_gpu(magma_int_t m,
                                    magma_int_t n,
                                    magmaDoubleComplex_ptr dA,
                                    magma_int_t ldda,
                                    magmaDoubleComplex* tau,
                                    magma_int_t* info)
{
    return magma_zgeqrf2_gpu(m, n, dA, ldda, tau, info);
}

/* SYTRD/HETRD SYTRD2/HETRD2*/
inline magma_int_t magma_sytrd_hetrd_gpu(magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaFloat_ptr dA,
                                         magma_int_t ldda,
                                         float* d,
                                         float* e,
                                         float* tau,
                                         float* A,
                                         magma_int_t lda,
                                         float* work,
                                         magma_int_t lwork,
                                         magma_int_t* info)
{
    return magma_ssytrd_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, info);
}

inline magma_int_t magma_sytrd_hetrd_gpu(magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaDouble_ptr dA,
                                         magma_int_t ldda,
                                         double* d,
                                         double* e,
                                         double* tau,
                                         double* A,
                                         magma_int_t lda,
                                         double* work,
                                         magma_int_t lwork,
                                         magma_int_t* info)
{
    return magma_dsytrd_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, info);
}

inline magma_int_t magma_sytrd_hetrd_gpu(magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaFloatComplex_ptr dA,
                                         magma_int_t ldda,
                                         float* d,
                                         float* e,
                                         magmaFloatComplex* tau,
                                         magmaFloatComplex* A,
                                         magma_int_t lda,
                                         magmaFloatComplex* work,
                                         magma_int_t lwork,
                                         magma_int_t* info)
{
    return magma_chetrd_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, info);
}

inline magma_int_t magma_sytrd_hetrd_gpu(magma_uplo_t uplo,
                                         magma_int_t n,
                                         magmaDoubleComplex_ptr dA,
                                         magma_int_t ldda,
                                         double* d,
                                         double* e,
                                         magmaDoubleComplex* tau,
                                         magmaDoubleComplex* A,
                                         magma_int_t lda,
                                         magmaDoubleComplex* work,
                                         magma_int_t lwork,
                                         magma_int_t* info)
{
    return magma_zhetrd_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, info);
}

inline magma_int_t magma_sytrd2_hetrd2_gpu(magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaFloat_ptr dA,
                                           magma_int_t ldda,
                                           float* d,
                                           float* e,
                                           float* tau,
                                           float* A,
                                           magma_int_t lda,
                                           float* work,
                                           magma_int_t lwork,
                                           magmaFloat_ptr dwork,
                                           magma_int_t ldwork,
                                           magma_int_t* info)
{
    return magma_ssytrd2_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, dwork, ldwork, info);
}

inline magma_int_t magma_sytrd2_hetrd2_gpu(magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaDouble_ptr dA,
                                           magma_int_t ldda,
                                           double* d,
                                           double* e,
                                           double* tau,
                                           double* A,
                                           magma_int_t lda,
                                           double* work,
                                           magma_int_t lwork,
                                           magmaDouble_ptr dwork,
                                           magma_int_t ldwork,
                                           magma_int_t* info)
{
    return magma_dsytrd2_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, dwork, ldwork, info);
}

inline magma_int_t magma_sytrd2_hetrd2_gpu(magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaFloatComplex_ptr dA,
                                           magma_int_t ldda,
                                           float* d,
                                           float* e,
                                           magmaFloatComplex* tau,
                                           magmaFloatComplex* A,
                                           magma_int_t lda,
                                           magmaFloatComplex* work,
                                           magma_int_t lwork,
                                           magmaFloatComplex_ptr dwork,
                                           magma_int_t ldwork,
                                           magma_int_t* info)
{
    return magma_chetrd2_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, dwork, ldwork, info);
}

inline magma_int_t magma_sytrd2_hetrd2_gpu(magma_uplo_t uplo,
                                           magma_int_t n,
                                           magmaDoubleComplex_ptr dA,
                                           magma_int_t ldda,
                                           double* d,
                                           double* e,
                                           magmaDoubleComplex* tau,
                                           magmaDoubleComplex* A,
                                           magma_int_t lda,
                                           magmaDoubleComplex* work,
                                           magma_int_t lwork,
                                           magmaDoubleComplex_ptr dwork,
                                           magma_int_t ldwork,
                                           magma_int_t* info)
{
    return magma_zhetrd2_gpu(uplo, n, dA, ldda, d, e, tau, A, lda, work, lwork, dwork, ldwork, info);
}