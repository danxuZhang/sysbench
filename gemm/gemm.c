#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#if USE_MPI
#include <mpi.h>
#endif

#if USE_SINGLE
#define DTYPE float
#define GEMM sgemm_
#else
#define DTYPE double
#define GEMM dgemm_
#endif

extern void dgemm_(const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const double *alpha,
                   const double *a, const int *lda, const double *b,
                   const int *ldb, const double *beta, double *c,
                   const int *ldc);

extern void sgemm_(const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const float *alpha,
                   const float *a, const int *lda, const float *b,
                   const int *ldb, const float *beta, float *c, const int *ldc);

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void loop(int N, int its, DTYPE *A, DTYPE *B, DTYPE *C, double *elapsed) {
  DTYPE alpha = 1.0;
  DTYPE beta = 0.0;
  char transa = 'N';
  char transb = 'N';
  int lda = N;
  int ldb = N;
  int ldc = N;
  int m = N;
  int n = N;
  int k = N;

#ifdef DEBUG
  printf("Debug: Start Iterations\n");
#endif
  for (int it = 0; it < its; ++it) {
    double start = get_time();

#if USE_SINGLE
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
           &ldc);
#else
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
           &ldc);
#endif

    double end = get_time();
    // elapsed[it] = (end - start);
#ifdef DEEBUG
    printf("Debug: Iteration %d took %.6f seconds\n", it + 1, elapsed[it]);
#endif
  } // for it : its
#ifdef DEBUG
  // printf("Debug: End Iterations\n");
#endif
}

void run_benchmark(int mpi_rank, int mpi_size, int N, int its, int warmup_its) {

  // Matrix A(NxN) B(NxN) C(NxN)
  // Floating operations (2 * N^3 + 3 * N^2)
  const double ops = 2.0 * N * N * N + 3.0 * N * N;
  // Problem size in MB: 3 matrices * N^2 elements * sizeof(DTYPE) bytes
  const double problem_size_mb =
      (3.0 * N * N * sizeof(DTYPE)) / (1024.0 * 1024.0);

  // TODO accomodate for single
  DTYPE *A = (DTYPE *)malloc(N * N * sizeof(DTYPE));
  DTYPE *B = (DTYPE *)malloc(N * N * sizeof(DTYPE));
  DTYPE *C = (DTYPE *)malloc(N * N * sizeof(DTYPE));

  if (A == NULL || B == NULL || C == NULL) {
    if (A) {
      free(A);
    }
    if (B) {
      free(B);
    }
    if (C) {
      free(C);
    }
    fprintf(stderr, "Failed to malloc matrix dimension %d x %d\n", N, N);
    return;
  }

  double *elapsed = (double *)malloc(its * sizeof(double));

  if (elapsed == NULL) {
    fprintf(stderr, "Failed to malloc elapsed array\n");
    free(A);
    free(B);
    free(C);
    return;
  }

  // Init Array
  for (int i = 0; i < N * N; i++) {
    A[i] = (DTYPE)rand() / RAND_MAX;
    B[i] = (DTYPE)rand() / RAND_MAX;
    C[i] = 0.0;
  }

  // Warm up
  if (mpi_rank == 0) {
    printf("Warm up started...\n");
  }
  double *dummy_elapsed = (double *)malloc(warmup_its * sizeof(double));
  double warmup_start = get_time();
  loop(N, warmup_its, A, B, C, dummy_elapsed);
  free(dummy_elapsed);
  double warmup_end = get_time();
  if (mpi_rank == 0) {
    printf("Warm up finished in %.2f s\n", (warmup_end - warmup_start));
  }

  // Benchmark loop
  if (mpi_rank == 0) {
    printf("Benchmark started...\n");
  }
#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  double loop_start = get_time();
  loop(N, its, A, B, C, elapsed);
#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  double loop_end = get_time();
  if (mpi_rank == 0) {
    printf("Benchmark finished in %.2f s\n", (loop_end - loop_start));
  }

  /**
  double sum_gflops = 0.0;
  double max_gflops = 0.0;
  printf("N,it,size_mb,time_s,gflops\n");
  for (int it = 0; it < its; ++it) {
      double gflops = ops / elapsed[it] / 1e9;
      printf("%d,%d,%.2f,%.2f,%.2f\n", N, it+1, problem_size_mb, elapsed[it],
  gflops); sum_gflops += gflops; max_gflops = gflops > max_gflops ? gflops :
  max_gflops;
  }

  printf("\nMax GFLOPS: %.2f\nAVG GFLOPS: %.2f\n", max_gflops, sum_gflops /
  its);
  */

  if (mpi_rank == 0) {
    double loop_ops = its * ops;
    double world_ops = loop_ops * mpi_size;
    double world_gflops = world_ops / (loop_end - loop_start);
    printf("GFLOP/S: %.2f\n", world_gflops / 1e9);
  }

  free(A);
  free(B);
  free(C);
  free(elapsed);
}

int main(int argc, char *argv[]) {
#if USE_MPI
  MPI_Init(&argc, &argv);
#endif

  int mpi_rank;
  int mpi_size;
#if USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
  mpi_rank = 0;
  mpi_size = 1;
#endif

  static const char *usage = "%s <2powerN> <bench_iters> <warmup_iters>";
  if (argc < 4) {
    if (mpi_rank == 0) {
      fprintf(stderr, usage, argv[0]);
    }
    return 1;
  }

  int n_pow;
  int its;
  int warmup;
  if (mpi_rank == 0) {
    n_pow = atoi(argv[1]);
    its = atoi(argv[2]);
    warmup = atoi(argv[3]);
  }

#if USE_MPI
  MPI_Bcast(&n_pow, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&its, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&warmup, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  int N = 1 << n_pow;
  if (mpi_rank == 0) {
    printf("Matrix Dim [%d,%d]\n", N, N);
    printf("Warmup Its: %d\n", warmup);
    printf("Benchmark Its: %d\n", its);
  }

  run_benchmark(mpi_rank, mpi_size, N, its, warmup);

#if USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
