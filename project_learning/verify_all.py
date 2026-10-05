"""统一验证入口：确定性单元测试，以及每阶段的独立进程实验。

python verify_all.py           快速检查（小种群小代数，不代表实验收敛）
python verify_all.py --full    完整运行全部阶段和70次 RCM 参数实验
python verify_all.py --unit-only  仅执行确定性验证，不执行阶段实验
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
STAGES = ["01_nsga2_zdt3.py", "02_crashworthiness.py", "03_constraints_mw7.py",
          "04_rcm13.py", "05_parameter_experiments.py"]


def load_stage(filename):
    spec = importlib.util.spec_from_file_location("learning_stage_" + filename[:2], ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = load_stage(STAGES[-1])
NSGA2, CNSGA2, RCM13 = STAGE.NSGA2, STAGE.CNSGA2, STAGE.RCM13
METRIC_NAMES = ["non_dominated", "normalize_objectives", "normalized_igd", "normalized_gd",
                "normalized_spacing", "hypervolume_3d", "normalized_hypervolume_3d",
                "zdt3_reference_front", "mw7_reference_front", "ZDT3_INTERVALS"]
globals().update({name: getattr(STAGE, name) for name in METRIC_NAMES})


"""Local, deterministic correctness checks; no third-party test runner needed."""

import time

import unittest

import numpy as np

class BoxProblem:
    n_var, n_obj = 3, 2
    lower = np.array([-2.0, 1.0, 7.0])
    upper = np.array([3.0, 4.0, 7.0])

    def evaluate(self, X):
        return np.column_stack((np.sum(X[:, :2] ** 2, axis=1),
                                np.sum((X[:, :2] - 1) ** 2, axis=1)))

class IntegerProblem(BoxProblem):
    def repair(self, X):
        X[:, 0] = np.rint(X[:, 0])
        return X

class ConstrainedProblem(BoxProblem):
    def evaluate(self, X):
        F = super().evaluate(X)
        C = np.maximum(0.0, X[:, :1] - 0.5)
        return F, C

class ZDT3Sanity:
    n_var, n_obj = 30, 2
    lower, upper = np.zeros(30), np.ones(30)

    def evaluate(self, X):
        g = 1 + 9 * X[:, 1:].sum(axis=1) / (self.n_var - 1)
        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return np.column_stack((f1, f2))

class NSGACorrectness(unittest.TestCase):
    def test_known_fronts_duplicates_and_ties(self):
        F = np.array([[0, 2], [1, 1], [2, 0], [2, 2], [3, 3], [1, 1]])
        rank, fronts = NSGA2().non_dominated_sort(F)
        np.testing.assert_array_equal(rank, [0, 0, 0, 1, 2, 0])
        self.assertEqual(set(fronts[0]), {0, 1, 2, 5})
        self.assertFalse(np.any(np.diag(NSGA2._objective_dominance(F))))

    def test_constraint_dominance(self):
        # A feasible but objectively poor row beats every infeasible row.
        F = np.array([[100, 100], [0, 0], [-1, -1], [200, 200], [101, 101]])
        CV = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        rank, _ = CNSGA2().non_dominated_sort(F, CV)
        np.testing.assert_array_equal(rank, [0, 2, 3, 2, 1])
        # Objective superiority cannot break equal positive CV ties.
        D = CNSGA2()._dominance_matrix(F, CV)
        self.assertFalse(D[1, 3])
        self.assertFalse(D[3, 1])

    def test_normalized_crowding_and_constant_objectives(self):
        F = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], dtype=float)
        front = [np.arange(5)]
        distance = NSGA2.crowding_distance(F, front)
        np.testing.assert_array_equal(np.isinf(distance), [True, False, False, False, True])
        np.testing.assert_allclose(distance[1:-1], 1.0)
        scaled = F * np.array([10000.0, 0.002]) + np.array([87.0, -99.0])
        np.testing.assert_allclose(NSGA2.crowding_distance(scaled, front), distance)
        constant = NSGA2.crowding_distance(np.ones((5, 2)), front)
        np.testing.assert_array_equal(constant, np.zeros(5))

    def test_elitism_and_front_truncation(self):
        opt = NSGA2(pop_size=3, seed=5)
        F = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0], [5, 5]])
        X = np.arange(6)[:, None]
        Xout, Fout, Cout = opt.environmental_selection(X, F)
        self.assertEqual(Fout.shape, (3, 2))
        self.assertEqual(Cout.shape, (3, 0))
        self.assertTrue({0, 4}.issubset(set(Xout[:, 0])))
        self.assertNotIn(5, Xout[:, 0])

    def test_constrained_environment_keeps_feasible_elites(self):
        opt = CNSGA2(pop_size=2)
        F = np.array([[10, 11], [11, 10], [0, 0], [-1, -1]])
        C = np.array([[0], [0], [0.01], [1.0]])
        X, _, retained_C = opt.environmental_selection(np.arange(4)[:, None], F, C)
        self.assertEqual(set(X[:, 0]), {0, 1})
        self.assertEqual(float(retained_C.sum()), 0.0)

    def test_bounds_fixed_variable_and_odd_population(self):
        problem = BoxProblem()
        optimizer = NSGA2(pop_size=21, pc=1, pm=1, eta_c=0, eta_m=0,
                          seed=10, record_every=7)
        X, F = optimizer.run(problem, max_gen=25)
        self.assertEqual(X.shape, (21, 3))
        self.assertTrue(np.all(X >= problem.lower))
        self.assertTrue(np.all(X <= problem.upper))
        np.testing.assert_array_equal(X[:, 2], 7)
        self.assertTrue(np.isfinite(F).all())
        self.assertEqual(optimizer.n_eval, 21 * 26)
        self.assertEqual([s['gen'] for s in optimizer.history], [0, 7, 14, 21, 25])
        self.assertEqual(optimizer.history[-1]['n_eval'], 21 * 26)

    def test_repair_applies_to_initial_and_every_offspring_population(self):
        class Audited(IntegerProblem):
            def evaluate(self, X):
                np.testing.assert_array_equal(X[:, 0], np.rint(X[:, 0]))
                return super().evaluate(X)
        X, _ = NSGA2(pop_size=17, pm=1).run(Audited(), max_gen=10)
        np.testing.assert_array_equal(X[:, 0], np.rint(X[:, 0]))

    def test_reproducibility_and_history_copies(self):
        opt = CNSGA2(pop_size=25, seed=42, record_every=5)
        X1, F1 = opt.run(ConstrainedProblem(), max_gen=10)
        CV1 = opt.final_cv.copy()
        X2, F2 = opt.run(ConstrainedProblem(), max_gen=10)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(F1, F2)
        np.testing.assert_array_equal(CV1, opt.final_cv)
        np.testing.assert_array_equal(opt.final_cv, opt.final_constraints.sum(axis=1))
        self.assertEqual(opt.history[-1]['feasible_ratio'], 1.0)
        old_history = opt.history[-1]['F'].copy()
        F2[:] = -123
        np.testing.assert_array_equal(opt.history[-1]['F'], old_history)

    def test_zero_generations_and_invalid_signed_constraints(self):
        opt = NSGA2(pop_size=5)
        opt.run(BoxProblem(), max_gen=0)
        self.assertEqual(opt.n_eval, 5)
        self.assertEqual(len(opt.history), 1)
        class Invalid(BoxProblem):
            def evaluate(self, X):
                return super().evaluate(X), -np.ones((len(X), 1))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            CNSGA2(pop_size=5).run(Invalid(), max_gen=1)

    def test_variation_disabled_retains_parents(self):
        problem = BoxProblem()
        opt = NSGA2(pop_size=15, pc=0, pm=0)
        parents = opt.initialize(problem)
        children = opt.mutation(opt.crossover(parents, problem), problem)
        np.testing.assert_array_equal(parents, children)

    def test_zdt3_short_sanity_converges_toward_known_front(self):
        problem = ZDT3Sanity()
        opt = NSGA2(pop_size=60, seed=17, record_every=100)
        t0 = time.perf_counter()
        X, F = opt.run(problem, max_gen=160)
        initial_mean_f2 = opt.history[0]['F'][:, 1].mean()
        # g=1 is necessary for the true ZDT3 front; require clear progress.
        g = 1 + 9 * X[:, 1:].sum(axis=1) / 29
        self.assertLess(float(np.median(g)), 1.15)
        self.assertLess(float(F[:, 1].mean()), initial_mean_f2 - 1.5)
        self.assertGreater(float(np.ptp(F[:, 0])), 0.7)
        print(f'\nZDT3 sanity: N=60 G=160, {time.perf_counter()-t0:.3f}s, '
              f'median g={np.median(g):.5f}, evaluations={opt.n_eval}')

"""Independent numeric and domain checks for the RCM13 problem definition.

Golden values below were calculated separately from the implementation using
45-digit Decimal arithmetic directly from course PDF p. 9, equation (29).
No optimizer or random outcome is used as a correctness oracle.
"""

import unittest

import numpy as np

class TestRCM13(unittest.TestCase):
    def test_course_pdf_golden_feasible_designs(self):
        X = np.array([[3.5, 0.7, 17, 7.8, 8, 3.6, 5.4],
                      [3.6, 0.7, 28, 8, 8, 3.9, 5.5]])
        expected_F = np.array([
            [2807.2060504005, 887.574888292839394, 797.635660747925005],
            [5287.98536988796, 695.123757558828309, 754.535463665066420],
        ])
        p = RCM13()
        F, C = p.evaluate(X)
        np.testing.assert_allclose(F, expected_F, rtol=2e-14, atol=1e-11)
        np.testing.assert_array_equal(C, np.zeros((2, 11)))
        expected_G = np.array([
            -0.00273760297769901611, -0.0004981095022438974,
            -0.280709498025526337, -0.4675350128905408,
            -212.425111707160606, -52.364339252074995,
            -28.1, 0, -7, -0.5, -0.16,
        ])
        np.testing.assert_allclose(p.constraint_values(X)[0], expected_G,
                                   rtol=2e-14, atol=1e-11)

    def test_known_infeasible_design_has_expected_violations(self):
        p = RCM13()
        F, C = p.evaluate([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])
        np.testing.assert_allclose(F[0],
            [2099.2022192918, 1696.459443994370672, 1004.657519712271314],
            rtol=2e-14, atol=1e-11)
        expected_C = [0.00913527804284106805, 0.000200295263670225198,
                      0, 0, 596.459443994370672, 154.657519712271314,
                      0, 1.285714285714285714, 0, 0, 0.1]
        np.testing.assert_allclose(C[0], expected_C, rtol=2e-14, atol=1e-11)

    def test_endpoint_repair_and_nonmutation(self):
        p = RCM13()
        X = np.tile((p.lower + p.upper) / 2.0, (4, 1))
        X[:, 2] = [16, 22.49, 22.5, 29]
        original = X.copy()
        repaired = p.repair(X)
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(repaired[:, 2], [17, 17, 28, 28])
        np.testing.assert_array_equal(p.repair(repaired), repaired)
        self.assertTrue(np.all(repaired >= p.lower))
        self.assertTrue(np.all(repaired <= p.upper))

    def test_variants_are_explicit_and_not_pooled(self):
        x = [3.5, 0.7, 21.5, 7.8, 8, 3.6, 5.4]
        self.assertEqual(RCM13().repair(x)[2], 17)
        self.assertEqual(RCM13("platemo").repair(x)[2], 22)
        self.assertEqual(RCM13("author_code").repair(x)[2], 22)
        course = RCM13().evaluate(x)[0]
        platemo = RCM13("platemo").evaluate(x)[0]
        author = RCM13("author_code").evaluate(x)[0]
        self.assertFalse(np.allclose(course, platemo))
        self.assertEqual(platemo[0, 0], author[0, 0])
        self.assertGreater(platemo[0, 1], author[0, 1])
        np.testing.assert_array_equal(RCM13().stress_limits, [1100, 850])
        np.testing.assert_array_equal(RCM13("author_code").stress_limits,
                                      [1300, 1100])

    def test_vectorized_domain_safety(self):
        p = RCM13()
        rng = np.random.default_rng(14)
        X = rng.uniform(p.lower, p.upper, (1000, p.n_var))
        F, C = p.evaluate(X)
        self.assertEqual(F.shape, (1000, 3))
        self.assertEqual(C.shape, (1000, 11))
        self.assertTrue(np.all(np.isfinite(F)))
        self.assertTrue(np.all(np.isfinite(C)))
        self.assertTrue(np.all(C >= 0))
        # Vector and batch calls must describe exactly the same design.
        np.testing.assert_allclose(p.evaluate(X[37])[0][0], F[37])
        np.testing.assert_allclose(p.evaluate(X[37])[1][0], C[37])

    def test_invalid_input_is_rejected(self):
        p = RCM13()
        for x in ([1, 2], [3.5, 0.7, np.nan, 7.8, 8, 3.6, 5.4]):
            with self.assertRaises(ValueError):
                p.evaluate(x)
        with self.assertRaises(ValueError):
            RCM13("undocumented")

"""Meaningful metric checks, including independent HV inclusion-exclusion."""

import itertools

import unittest

import numpy as np

def hv_inclusion_exclusion(F, ref):
    total = 0.0
    for k in range(1, len(F) + 1):
        for selected in itertools.combinations(F, k):
            corner = np.max(np.asarray(selected), axis=0)
            total += (-1) ** (k + 1) * np.prod(np.maximum(ref - corner, 0))
    return total

class MetricsTests(unittest.TestCase):
    def test_pareto_duplicates_ties_and_empty(self):
        F = np.array([[0, 3], [1, 2], [2, 1], [3, 0], [2, 2],
                      [1, 2], [1, 3], [4, 0]])
        np.testing.assert_array_equal(non_dominated(F),
                                      [1, 1, 1, 1, 0, 1, 0, 0])
        self.assertEqual(non_dominated(np.empty((0, 3))).size, 0)
        rng = np.random.default_rng(1426)
        for m in (1, 2, 3, 5):
            F = rng.integers(0, 5, (70, m))
            expected = [not np.any(np.all(F <= p, axis=1) &
                                   np.any(F < p, axis=1)) for p in F]
            np.testing.assert_array_equal(non_dominated(F), expected)

    def test_distances_and_asymmetric_coverage(self):
        ref = np.array([[0, 1], [1, 0]])
        F = ref[:1]
        self.assertAlmostEqual(normalized_igd(ref, F, [0, 0], [1, 1]),
                               np.sqrt(2) / 2)
        self.assertEqual(normalized_gd(ref, F, [0, 0], [1, 1]), 0)
        self.assertEqual(normalized_igd(ref, ref, [0, 0], [1, 1]), 0)
        self.assertTrue(np.isinf(normalized_igd(ref, [], [0, 0], [1, 1])))
        self.assertTrue(np.isinf(normalized_gd(ref, [], [0, 0], [1, 1])))
        with self.assertRaises(ValueError):
            normalized_igd([], ref, [0, 0], [1, 1])

    def test_fixed_scaling_translation_invariance(self):
        F = np.array([[.2, .8, .5], [.8, .2, .3]])
        R = np.array([[.1, .7, .3], [.7, .1, .3]])
        shift, scale = np.array([-20, 40, 1]), np.array([3, 10, .2])
        for metric in (normalized_igd, normalized_gd):
            self.assertAlmostEqual(metric(R, F, np.zeros(3), np.ones(3)),
                                   metric(R*scale+shift, F*scale+shift,
                                          shift, shift+scale))
        self.assertAlmostEqual(
            normalized_hypervolume_3d(F, np.zeros(3), np.ones(3)),
            normalized_hypervolume_3d(F*scale+shift, shift, shift+scale))
        a = normalized_spacing(F, np.zeros(3), np.ones(3))
        b = normalized_spacing(F*scale+shift, shift, shift+scale)
        self.assertAlmostEqual(a['mean_nn'], b['mean_nn'])
        self.assertAlmostEqual(a['cv_nn'], b['cv_nn'])

    def test_spacing_and_duplicates(self):
        S = normalized_spacing([[0, 0], [1, 0], [3, 0]], [0, 0], [1, 1])
        self.assertAlmostEqual(S['mean_nn'], 4/3)
        self.assertAlmostEqual(S['cv_nn'], np.std([1, 1, 2])/(4/3))
        self.assertTrue(np.isnan(normalized_spacing([[0, 0]], [0, 0], [1, 1])['mean_nn']))
        repeated = normalized_spacing([[1, 1], [1, 1]], [0, 0], [1, 1])
        self.assertEqual(repeated['mean_nn'], 0)
        self.assertTrue(np.isnan(repeated['cv_nn']))

    def test_hv_single_point_exact_rectangle_and_two_points(self):
        self.assertEqual(hypervolume_3d([[1, 2, 3]], [3, 5, 7]), 24)
        self.assertAlmostEqual(hypervolume_3d([[0, 0, 0]]), 1.1**3)
        self.assertAlmostEqual(hypervolume_3d([[.2, .8, .4], [.8, .2, .4]], 1), .168)

    def test_hv_dominated_duplicates_and_reference_outliers(self):
        F = [[.2, .3, .4], [.5, .6, .7], [.2, .3, .4],
             [1.2, -10, -10], [.1, 1, .1], [.1, .1, 4]]
        self.assertAlmostEqual(hypervolume_3d(F, 1), .8*.7*.6)
        self.assertEqual(hypervolume_3d(np.empty((0, 3))), 0)
        self.assertEqual(hypervolume_3d([[2, 0, 0]], 1), 0)
        # Normalized values below zero are valid: fixed bounds need not bound
        # every later improvement and must not be silently recomputed or clipped.
        self.assertAlmostEqual(hypervolume_3d([[-.2, .3, .4]], 1), 1.2*.7*.6)

    def test_hv_matches_independent_inclusion_exclusion(self):
        rng = np.random.default_rng(14)
        for n in range(1, 9):
            for _ in range(4):
                F = rng.uniform(-.2, 1.4, (n, 3))
                ref = np.array([1, 1.1, 1.2])
                self.assertAlmostEqual(hypervolume_3d(F, ref),
                                       hv_inclusion_exclusion(F, ref), places=12)

    def test_invalid_inputs_rejected(self):
        for bad in ([[np.nan, 0, 0]], [[0, np.inf, 0]], [[0, 0]]):
            with self.assertRaises(ValueError):
                hypervolume_3d(bad)
        with self.assertRaises(ValueError):
            normalize_objectives([[0, 0]], [0, 0], [0, 1])
        with self.assertRaises(ValueError):
            hypervolume_3d([[0, 0, 0]], [1, 2])

    def test_zdt3_front_equation_and_five_segments(self):
        F = zdt3_reference_front(1000)
        self.assertEqual(F.shape, (1000, 2))
        x = F[:, 0]
        np.testing.assert_allclose(F[:, 1], 1-np.sqrt(x)-x*np.sin(10*np.pi*x))
        for lo, hi in ZDT3_INTERVALS:
            self.assertGreater(np.sum((x >= lo) & (x <= hi)), 2)
        # Rounded segment start endpoints can tie/weakly dominate at 1e-9 scale.
        self.assertGreaterEqual(np.count_nonzero(non_dominated(F)), 996)

    def test_mw7_feasibility_nondominance_and_attainability(self):
        F = mw7_reference_front(20001, n_points=2000)
        self.assertEqual(F.shape, (2000, 2))
        self.assertTrue(non_dominated(F).all())
        theta = np.arctan2(F[:, 1], F[:, 0])
        radius = np.linalg.norm(F, axis=1)
        wave = np.sin(4*theta)**8
        self.assertTrue(np.all(radius >= 1 - 1e-12))
        self.assertTrue(np.all(radius >= 1.15-.2*wave-1e-12))
        self.assertTrue(np.all(radius <= 1.2+.4*wave**2+1e-12))
        # Construct valid decision vectors that attain these objective points.
        X = np.empty((len(F), 15))
        X[:, 0] = F[:, 0] / radius
        for j in range(1, 15):
            X[:, j] = 1 - (X[:, j-1] - .5)**2
        X[:, -1] -= np.sqrt(np.maximum(radius-1, 0)/2)
        self.assertTrue(np.all((X >= 0) & (X <= 1)))
        g = 1+2*np.sum((X[:, 1:] + (X[:, :-1]-.5)**2-1)**2, axis=1)
        np.testing.assert_allclose(g, radius, atol=1e-12)
        # Boundary points dominated globally must actually have been removed.
        dense = mw7_reference_front(20001)
        self.assertLess(len(dense), 20001)

class StageIntegrityTests(unittest.TestCase):
    def test_original_export_untouched_and_explicitly_incomplete(self):
        text = (ROOT/"00_original.py").read_text(encoding="utf-8")
        manifest = json.loads((ROOT/"source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         manifest["original_export_sha256"])
        self.assertIn("NSGA2(pop_size=100, ??)", text)
        # 原始模板就不能编译：验证器记录 TEMPLATE，而不是把它伪装成可运行的完成版。
        with self.assertRaises(SyntaxError):
            ast.parse(text)

    def test_later_stages_preserve_previous_implementations(self):
        previous = None
        for filename in STAGES:
            text = (ROOT/filename).read_text(encoding="utf-8")
            body = text[text.index("# %% 阶段 01"):text.rindex('if __name__ == "__main__":')].rstrip()
            ast.parse(text)
            if previous is not None:
                self.assertTrue(body.startswith(previous), f"Earlier content changed: {filename}")
            previous = body

    def test_teacher_models_preserved_in_all_applicable_stages(self):
        text = (ROOT/"00_original.py").read_text(encoding="utf-8")
        mapping = {"ZDT3": (6, 0), "CrashworthinessDesign": (15, 1), "MW7": (21, 2)}
        for name, (cell_number, first_stage) in mapping.items():
            source = text.split(f"# %% 原始 Notebook 第 {cell_number} 单元\n", 1)[1].split("\n# %%", 1)[0]
            expected = ast.dump(ast.parse(source).body[0], include_attributes=False)
            for filename in STAGES[first_stage:]:
                classes = {node.name: node for node in ast.parse((ROOT/filename).read_text(encoding="utf-8")).body
                           if isinstance(node, ast.ClassDef)}
                self.assertEqual(ast.dump(classes[name], include_attributes=False), expected)

    def test_incremental_task_registration_and_import_has_no_experiment(self):
        tasks = ["zdt3", "crash", "mw7", "rcm13", "parameters"]
        for i, filename in enumerate(STAGES):
            stage = load_stage(filename)
            self.assertEqual(list(stage.EXPERIMENTS), tasks[:i+1])
            self.assertIsNone(stage.ARTIFACT_DIR)
            for check in stage.CHECKS:
                check()

    def test_smoke_reports_infeasibility_instead_of_false_success(self):
        import numpy as np
        result = dict(F=np.array([[0., 0.], [1., 1.]]), CV=np.array([1., 2.]))
        self.assertFalse(STAGE.feasible_mask(result).any())
        self.assertEqual(STAGE.feasible_front(result).shape, (0, 2))


def verify():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true")
    group.add_argument("--unit-only", action="store_true")
    args = parser.parse_args()
    mode = "unit" if args.unit_only else ("full" if args.full else "quick")
    dest = ROOT/"verification"/mode
    dest.mkdir(parents=True, exist_ok=True)
    report = dict(mode=mode, python=sys.version.split()[0], numpy=STAGE.np.__version__,
                  matplotlib=STAGE.matplotlib.__version__, original_template="PRESERVED_INCOMPLETE",
                  stages=[], warnings=[], passed=False)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report["unit_tests"] = dict(run=result.testsRun, failures=len(result.failures),
                                errors=len(result.errors), passed=result.wasSuccessful())
    try:
        if not result.wasSuccessful():
            raise RuntimeError("Unit checks failed; experiments were not run")
        if not args.unit_only:
            for index, filename in enumerate(STAGES):
                print(f"\n[STAGE {index+1}/5] {filename} ({mode})", flush=True)
                # 只复制一个阶段文件到临时目录，证明它不需要其它阶段或 Notebook。
                with tempfile.TemporaryDirectory(prefix="ceg5302-stage-") as isolated:
                    isolated_script = Path(isolated)/filename
                    shutil.copyfile(ROOT/filename, isolated_script)
                    output = dest/Path(filename).stem
                    command = [sys.executable, str(isolated_script), "--out", str(output)]
                    if not args.full:
                        command.append("--quick")
                    if index == 4:
                        command.append("--all")
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    stage_started = time.perf_counter()
                    process = subprocess.run(command, cwd=isolated, env=env, text=True,
                                             encoding="utf-8", errors="replace", capture_output=True, timeout=900)
                    log = dest/(Path(filename).stem + ".log")
                    log.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
                stage_report = dict(file=filename, returncode=process.returncode,
                                    seconds=time.perf_counter()-stage_started, log=str(log.relative_to(ROOT)))
                report["stages"].append(stage_report)
                if process.returncode:
                    raise RuntimeError(f"{filename} failed; see {log}")
                expected_tasks = list(load_stage(filename).EXPERIMENTS) if index == 4 else [list(load_stage(filename).EXPERIMENTS)[-1]]
                stage_report["tasks"] = {}
                for task in expected_tasks:
                    folder = output/task
                    summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
                    images = list(folder.glob("*.png"))
                    if not images or not all(path.stat().st_size > 1000 for path in images):
                        raise RuntimeError(f"Missing or empty plot: {filename}/{task}")
                    if task == "parameters":
                        expected_runs = 70 if args.full else 28
                        if summary["total_rcm_runs"] != expected_runs:
                            raise RuntimeError("Incomplete parameter experiment matrix")
                        if set(summary["tuning_seeds"]) & set(summary["holdout_seeds"]):
                            raise RuntimeError("Tuning and holdout seeds overlap")
                        controls = [c for label, c in summary["configs"].items() if label.startswith("budget_")]
                        if len({c["pop_size"]*(c["max_gen"]+1) for c in controls}) != 1:
                            raise RuntimeError("Equal-budget comparisons have unequal evaluation counts")
                        matrix = STAGE.np.loadtxt(folder/"rcm_final_designs.csv", delimiter=",", skiprows=1, ndmin=2)
                        if matrix.shape[1] != 21 or not len(matrix):
                            raise RuntimeError("RCM final design table has wrong dimensions")
                        problem = RCM13()
                        F, C = problem.evaluate(matrix[:, :7])
                        STAGE.np.testing.assert_allclose(F, matrix[:, 7:10], rtol=1e-12, atol=1e-12)
                        STAGE.np.testing.assert_allclose(problem.constraint_values(matrix[:, :7]), matrix[:, 10:], atol=1e-10)
                        if STAGE.np.any(C > 0):
                            raise RuntimeError("Claimed final feasible designs violate constraints")
                        stage_report["tasks"][task] = dict(total_runs=expected_runs, selected=summary["best_label"],
                                                         final_members=len(matrix))
                    else:
                        matrix = STAGE.np.loadtxt(folder/"population.csv", delimiter=",", skiprows=1, ndmin=2)
                        problem = {"zdt3": STAGE.ZDT3, "crash": STAGE.CrashworthinessDesign,
                                   "mw7": STAGE.MW7, "rcm13": RCM13}[task]()
                        values = problem.evaluate(matrix[:, :problem.n_var])
                        F, C = values if isinstance(values, tuple) else (values, STAGE.np.zeros((len(matrix), 0)))
                        STAGE.np.testing.assert_allclose(F, matrix[:, problem.n_var:problem.n_var+problem.n_obj], rtol=1e-12, atol=1e-12)
                        STAGE.np.testing.assert_allclose(C.sum(axis=1), matrix[:, -2], rtol=1e-12, atol=1e-12)
                        if len(matrix) != summary["config"]["pop_size"]:
                            raise RuntimeError("Saved population size does not match configuration")
                        if summary.get("segments_hit", 0) < summary.get("total_segments", 0):
                            report["warnings"].append(f"{filename}/{task}: incomplete front segment coverage ({mode})")
                        if not summary["n_feasible_nd"]:
                            report["warnings"].append(f"{filename}/{task}: no feasible nondominated solution")
                        if args.full and summary["feasible_ratio"] != 1:
                            raise RuntimeError(f"Fixed-seed full-run feasibility regression: {filename}/{task}")
                        if args.full and task == "zdt3" and (summary["igd"] >= .02 or summary["median_g"] >= 1.1):
                            raise RuntimeError("Fixed-seed ZDT3 convergence regression")
                        stage_report["tasks"][task] = summary
                print(f"[PASS] {filename}: isolated execution, data re-evaluation and output checks", flush=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        print(f"[FAIL] {error}", file=sys.stderr)
    finally:
        report["seconds"] = time.perf_counter()-started
        (dest/"verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"\n{'PASS' if report['passed'] else 'FAIL'} | {result.testsRun} unit checks | {mode} stage validation")
    for warning in report["warnings"]:
        print("[WARN]", warning)
    print(f"Report: {dest/'verification_report.json'}")
    print("Original 00_original.py is an incomplete course template; it is preserved, not executed.")
    print("A pass is evidence for explicit checks, not teacher acceptance or proof of a global optimum.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(verify())
