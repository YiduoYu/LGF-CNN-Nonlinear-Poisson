#!/usr/bin/env python3
"""
Convergence tests for the LGF-BAE solver.

Tests the Python implementation against the method of manufactured solutions
to verify correctness and convergence rates.

Usage:
    python test_convergence.py
    python test_convergence.py --quick   # Fast subset of tests
    python test_convergence.py --full    # Full convergence study
"""

import sys
import time
import argparse
import numpy as np

# Add parent directory to path if run directly
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LGF_python.solver import solve_helmholtz_2d


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

QUICK_TESTS = [
    # (N, iOrder, ms, geom, sigma, alpha, beta, potential)
    # Use N >= 5 for asymptotic convergence rates
    (5, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),   # Dirichlet
    (6, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (7, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
]

STANDARD_TESTS = [
    # Poisson, 2nd order, single layer, ellipse, Dirichlet BC
    # (N, iOrder, ms, geom, sigma, alpha, beta, potential)
    (5, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (7, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (8, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
]

ROBIN_TESTS = [
    (5, 2, 'trig1', 'ellipse', 0.0, 1.0, 1.0, 'single'),
    (6, 2, 'trig1', 'ellipse', 0.0, 1.0, 1.0, 'single'),
    (7, 2, 'trig1', 'ellipse', 0.0, 1.0, 1.0, 'single'),
]

NEUMANN_TESTS = [
    (5, 2, 'trig1', 'ellipse', 0.0, 1.0, 0.0, 'single'),
    (6, 2, 'trig1', 'ellipse', 0.0, 1.0, 0.0, 'single'),
    (7, 2, 'trig1', 'ellipse', 0.0, 1.0, 0.0, 'single'),
]

DOUBLE_LAYER_TESTS = [
    (5, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'double'),
    (6, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'double'),
    (7, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'double'),
]

FOURTH_ORDER_TESTS = [
    (5, 4, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 4, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (7, 4, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
]

MODIFIED_HELMHOLTZ_TESTS = [
    (5, 2, 'trig1', 'ellipse', 1.0, 0.0, 1.0, 'single'),
    (6, 2, 'trig1', 'ellipse', 1.0, 0.0, 1.0, 'single'),
    (7, 2, 'trig1', 'ellipse', 1.0, 0.0, 1.0, 'single'),
]

GEOMETRY_TESTS = [
    # Different geometries, Dirichlet BC, N=6
    (6, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'trig1', 'star', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'trig1', 'leaf', 0.0, 0.0, 1.0, 'single'),
]

MS_TESTS = [
    # Different manufactured solutions, Dirichlet BC, N=6
    (6, 2, 'poly1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'poly2', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'poly3', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'trig1', 'ellipse', 0.0, 0.0, 1.0, 'single'),
    (6, 2, 'trig3', 'ellipse', 0.0, 0.0, 1.0, 'single'),
]


# ---------------------------------------------------------------------------
# Convergence rate computation
# ---------------------------------------------------------------------------

def compute_convergence_rate(errors, N_values):
    """
    Compute empirical convergence rate from error data.

    Uses pairwise rates between consecutive grid refinements.

    Parameters
    ----------
    errors : list of float
        Max-norm errors for each resolution.
    N_values : list of int
        Refinement parameter N for each data point.

    Returns
    -------
    rates : list of float
        Pairwise convergence rates.
    mean_rate : float
        Mean of pairwise rates.
    """
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            h_ratio = (2**N_values[i] - 1 + 1) / (2**N_values[i-1] - 1 + 1)
            rate = np.log(errors[i-1] / errors[i]) / np.log(h_ratio)
            rates.append(rate)
    mean_rate = np.mean(rates) if rates else 0.0
    return rates, mean_rate


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_suite(tests, label, expected_rate=None):
    """
    Run a suite of convergence tests and report results.

    Parameters
    ----------
    tests : list of tuple
        Test configurations (N, iOrder, ms, geom, sigma, alpha, beta, potential).
    label : str
        Label for this test suite.
    expected_rate : float, optional
        Expected convergence rate for validation.

    Returns
    -------
    passed : bool
        True if all tests passed.
    """
    print(f"{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    errors = []
    N_values = []
    all_passed = True

    for params in tests:
        N, iOrder, ms, geom, sigma, alpha, beta, potential = params
        try:
            err, c1, c2, c3, N_actual, info = solve_helmholtz_2d(
                N=N, iOrder=iOrder, ms=ms, geom=geom,
                sigma=sigma, alpha=alpha, beta=beta,
                potential=potential, plot_stuff=False
            )
            errors.append(err)
            N_values.append(N)

            bc_type = ('Dirichlet' if alpha == 0 else
                       'Neumann' if beta == 0 else 'Robin')
            print(f"  N={N} (grid={N_actual}x{N_actual}) | "
                  f"Order={iOrder} | {bc_type} | {potential} | "
                  f"err={err:.6e} | cond=({c1:.2e},{c2:.2e},{c3:.2e})")

            if err > 1.0:
                print(f"  WARNING: Large error ({err:.6e}) for N={N}")
                all_passed = False

        except Exception as e:
            print(f"  N={N} | FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            errors.append(np.nan)
            N_values.append(N)

    # Compute convergence rate
    valid = [i for i, e in enumerate(errors) if not np.isnan(e) and e > 0]
    if len(valid) >= 2:
        valid_errors = [errors[i] for i in valid]
        valid_N = [N_values[i] for i in valid]
        rates, mean_rate = compute_convergence_rate(valid_errors, valid_N)
        rate_str = ", ".join(f"N={valid_N[i]}→{valid_N[i+1]}: {r:.2f}" for i, r in enumerate(rates))
        print(f"  Convergence rates: {rate_str}")
        print(f"  Mean rate: {mean_rate:.2f}")

        if expected_rate is not None:
            if abs(mean_rate - expected_rate) < 1.0:
                print(f"  ✓ Mean rate matches expected {expected_rate} (within 1.0)")
            else:
                print(f"  ✗ Mean rate {mean_rate:.2f} differs from expected {expected_rate}")
                all_passed = False

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convergence tests for LGF-BAE solver'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run only quick tests')
    parser.add_argument('--full', action='store_true',
                        help='Run full convergence study (takes longer)')
    parser.add_argument('--test', type=str, default=None,
                        help='Run a specific test, e.g. "standard" or "robin"')
    args = parser.parse_args()

    print("LGF-BAE Solver Convergence Tests")
    print("=================================\n")
    t_start = time.time()

    all_results = []

    if args.quick:
        # Quick test: just a few resolutions
        print("Running QUICK tests...\n")
        passed = run_test_suite(QUICK_TESTS, "Quick Convergence (Dirichlet BC)",
                                expected_rate=2.0)
        all_results.append(('Quick', passed))

    elif args.full:
        # Full convergence study
        suites = [
            (STANDARD_TESTS, "2nd Order, Dirichlet BC, Single Layer, Ellipse", 2.0),
            (ROBIN_TESTS, "2nd Order, Robin BC, Single Layer, Ellipse", 2.0),
            (NEUMANN_TESTS, "2nd Order, Neumann BC, Single Layer, Ellipse", 2.0),
            (DOUBLE_LAYER_TESTS, "2nd Order, Dirichlet BC, Double Layer, Ellipse", 2.0),
            (FOURTH_ORDER_TESTS, "4th Order, Dirichlet BC, Single Layer, Ellipse", 4.0),
            (MODIFIED_HELMHOLTZ_TESTS, "Modified Helmholtz, Dirichlet BC, Single Layer", 2.0),
            (GEOMETRY_TESTS, "Various Geometries, Dirichlet BC", None),
            (MS_TESTS, "Various Manufactured Solutions, Dirichlet BC", None),
        ]

        for suite_tests, label, expected_rate in suites:
            passed = run_test_suite(suite_tests, label, expected_rate)
            all_results.append((label, passed))

    elif args.test:
        # Specific test suite
        test_map = {
            'standard': (STANDARD_TESTS, "Standard", 2.0),
            'robin': (ROBIN_TESTS, "Robin BC", 2.0),
            'neumann': (NEUMANN_TESTS, "Neumann BC", 2.0),
            'double': (DOUBLE_LAYER_TESTS, "Double Layer", 2.0),
            'fourth': (FOURTH_ORDER_TESTS, "4th Order", 4.0),
            'helmholtz': (MODIFIED_HELMHOLTZ_TESTS, "Modified Helmholtz", 2.0),
            'geometry': (GEOMETRY_TESTS, "Various Geometries", None),
            'ms': (MS_TESTS, "Various MS", None),
        }
        if args.test in test_map:
            suite_tests, label, expected_rate = test_map[args.test]
            passed = run_test_suite(suite_tests, label, expected_rate)
            all_results.append((label, passed))
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available: {list(test_map.keys())}")
            return

    else:
        # Default: standard convergence test
        print("Running DEFAULT convergence tests...\n")
        passed = run_test_suite(STANDARD_TESTS,
                                "2nd Order, Dirichlet BC, Single Layer, Ellipse",
                                expected_rate=2.0)
        all_results.append(('Standard', passed))

    # Summary
    t_total = time.time() - t_start
    print(f"{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for label, passed in all_results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {label}")
    print(f"  Total time: {t_total:.1f} s")
    print()

    # Return exit code
    all_passed = all(p for _, p in all_results)
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
