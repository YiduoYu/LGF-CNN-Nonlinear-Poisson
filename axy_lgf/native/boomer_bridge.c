/* Serial, real-double hypre bridge. All multigrid work is performed by hypre.
 * Build against the pinned hypre source with build_boomer.py.
 */
#include <stdint.h>
#include <stdlib.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"

#ifndef HYPRE_SEQUENTIAL
#error "This bridge requires hypre configured --without-MPI"
#endif
#ifdef HYPRE_COMPLEX
#error "This bridge requires real arithmetic"
#endif
_Static_assert(sizeof(HYPRE_Int) == sizeof(int32_t), "requires 32-bit HYPRE_Int");
_Static_assert(sizeof(HYPRE_BigInt) == sizeof(int32_t), "requires 32-bit HYPRE_BigInt");
_Static_assert(sizeof(HYPRE_Real) == sizeof(double), "requires float64");

typedef struct {
    HYPRE_Int n;
    HYPRE_BigInt *rows;
    HYPRE_IJMatrix matrix;
    HYPRE_IJVector rhs, solution;
    HYPRE_ParCSRMatrix par_matrix;
    HYPRE_ParVector par_rhs, par_solution;
    HYPRE_Solver solver;
} Boomer;

const char *axy_boomer_version(void) { return HYPRE_RELEASE_VERSION; }

void axy_boomer_destroy(void *handle) {
    Boomer *b = handle;
    if (!b) return;
    if (b->solver) HYPRE_BoomerAMGDestroy(b->solver);
    if (b->solution) HYPRE_IJVectorDestroy(b->solution);
    if (b->rhs) HYPRE_IJVectorDestroy(b->rhs);
    if (b->matrix) HYPRE_IJMatrixDestroy(b->matrix);
    free(b->rows);
    free(b);
}

/* hypre's public MPI_Comm is an integer in the serial build; communicator 0
 * is hypre's sequential MPI_COMM_WORLD. No MPI runtime is loaded or initialized.
 */
#define CHECK(call) do { status = (call); if (status) goto fail; } while (0)

int axy_boomer_create(int n, const int32_t *indptr, const int32_t *columns,
                     const double *values, void **handle) {
    HYPRE_Int status = 0;
    Boomer *b = calloc(1, sizeof(*b));
    HYPRE_Int *counts = NULL;
    *handle = NULL;
    if (!b) return -1;
    if (!HYPRE_Initialized()) CHECK(HYPRE_Initialize());
    HYPRE_ClearAllErrors();
    b->n = n;
    b->rows = malloc((size_t)n * sizeof(*b->rows));
    counts = malloc((size_t)n * sizeof(*counts));
    if (!b->rows || !counts) { status = -1; goto fail; }
    for (int i = 0; i < n; ++i) {
        b->rows[i] = i;
        counts[i] = indptr[i+1] - indptr[i];
    }
    CHECK(HYPRE_IJMatrixCreate((MPI_Comm)0, 0, n-1, 0, n-1, &b->matrix));
    CHECK(HYPRE_IJMatrixSetObjectType(b->matrix, HYPRE_PARCSR));
    CHECK(HYPRE_IJMatrixSetRowSizes(b->matrix, counts));
    CHECK(HYPRE_IJMatrixInitialize(b->matrix));
    CHECK(HYPRE_IJMatrixSetValues(b->matrix, n, counts, b->rows, columns, values));
    CHECK(HYPRE_IJMatrixAssemble(b->matrix));
    CHECK(HYPRE_IJMatrixGetObject(b->matrix, (void **)&b->par_matrix));

    CHECK(HYPRE_IJVectorCreate((MPI_Comm)0, 0, n-1, &b->rhs));
    CHECK(HYPRE_IJVectorSetObjectType(b->rhs, HYPRE_PARCSR));
    CHECK(HYPRE_IJVectorInitialize(b->rhs));
    CHECK(HYPRE_IJVectorAssemble(b->rhs));
    CHECK(HYPRE_IJVectorGetObject(b->rhs, (void **)&b->par_rhs));
    CHECK(HYPRE_ParVectorSetConstantValues(b->par_rhs, 0.0));
    CHECK(HYPRE_IJVectorCreate((MPI_Comm)0, 0, n-1, &b->solution));
    CHECK(HYPRE_IJVectorSetObjectType(b->solution, HYPRE_PARCSR));
    CHECK(HYPRE_IJVectorInitialize(b->solution));
    CHECK(HYPRE_IJVectorAssemble(b->solution));
    CHECK(HYPRE_IJVectorGetObject(b->solution, (void **)&b->par_solution));
    CHECK(HYPRE_ParVectorSetConstantValues(b->par_solution, 0.0));

    CHECK(HYPRE_BoomerAMGCreate(&b->solver));
    CHECK(HYPRE_BoomerAMGSetCoarsenType(b->solver, 10)); /* HMIS */
    CHECK(HYPRE_BoomerAMGSetInterpType(b->solver, 6));   /* extended+i */
    CHECK(HYPRE_BoomerAMGSetPMaxElmts(b->solver, 4));
    CHECK(HYPRE_BoomerAMGSetStrongThreshold(b->solver, 0.25));
    CHECK(HYPRE_BoomerAMGSetAggNumLevels(b->solver, 0));
    CHECK(HYPRE_BoomerAMGSetRestriction(b->solver, 0)); /* R = P^T */
    CHECK(HYPRE_BoomerAMGSetRelaxType(b->solver, 6));   /* symmetric GS */
    CHECK(HYPRE_BoomerAMGSetRelaxOrder(b->solver, 0));
    CHECK(HYPRE_BoomerAMGSetNumSweeps(b->solver, 1));
    CHECK(HYPRE_BoomerAMGSetCycleRelaxType(b->solver, 9, 3)); /* coarse GE */
    CHECK(HYPRE_BoomerAMGSetMaxCoarseSize(b->solver, 9));
    CHECK(HYPRE_BoomerAMGSetMaxLevels(b->solver, 25));
    CHECK(HYPRE_BoomerAMGSetCycleType(b->solver, 1));    /* V-cycle */
    CHECK(HYPRE_BoomerAMGSetTol(b->solver, 0.0));
    CHECK(HYPRE_BoomerAMGSetMinIter(b->solver, 1));
    CHECK(HYPRE_BoomerAMGSetMaxIter(b->solver, 1));
    CHECK(HYPRE_BoomerAMGSetPrintLevel(b->solver, 0));
    CHECK(HYPRE_BoomerAMGSetup(b->solver, b->par_matrix, b->par_rhs, b->par_solution));
    free(counts);
    *handle = b;
    return 0;
fail:
    free(counts);
    axy_boomer_destroy(b);
    HYPRE_ClearAllErrors();
    return status;
}

static int load_rhs(Boomer *b, const double *rhs) {
    HYPRE_Int status = 0;
    HYPRE_ClearAllErrors();
    CHECK(HYPRE_IJVectorInitialize(b->rhs));
    CHECK(HYPRE_IJVectorSetValues(b->rhs, b->n, b->rows, rhs));
    CHECK(HYPRE_IJVectorAssemble(b->rhs));
    CHECK(HYPRE_IJVectorGetObject(b->rhs, (void **)&b->par_rhs));
    CHECK(HYPRE_ParVectorSetConstantValues(b->par_solution, 0.0));
fail:
    return status;
}

int axy_boomer_apply(void *handle, const double *rhs, double *result) {
    Boomer *b = handle;
    HYPRE_Int status = 0;
    CHECK(load_rhs(b, rhs));
    CHECK(HYPRE_BoomerAMGSolve(b->solver, b->par_matrix, b->par_rhs, b->par_solution));
    CHECK(HYPRE_IJVectorGetValues(b->solution, b->n, b->rows, result));
fail:
    return status;
}

int axy_boomer_matvec(void *handle, const double *rhs, double *result) {
    Boomer *b = handle;
    HYPRE_Int status = 0;
    CHECK(load_rhs(b, rhs));
    CHECK(HYPRE_ParCSRMatrixMatvec(1.0, b->par_matrix, b->par_rhs, 0.0, b->par_solution));
    CHECK(HYPRE_IJVectorGetValues(b->solution, b->n, b->rows, result));
fail:
    return status;
}

void axy_boomer_finalize(void) {
    if (HYPRE_Initialized() && !HYPRE_Finalized()) HYPRE_Finalize();
}
