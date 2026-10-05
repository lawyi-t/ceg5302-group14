# -*- coding: utf-8 -*-
"""第 3 阶段：新增约束处理与 MW7。

这是包含此前功能的独立完整版本；只需要 NumPy 和 Matplotlib。
python 03_constraints_mw7.py             运行本阶段新增实验
python 03_constraints_mw7.py --quick     小规模流程检查，不用于报告结论
python 03_constraints_mw7.py --all       运行本文件已实现的全部实验
python 03_constraints_mw7.py --check-only  只运行明确的代码正确性检查
阅读时先看“阶段 03 新增”分段，再回看被复用的函数。
"""


# %% 阶段 01 公共运行工具（每个阶段文件均自包含）
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

# 默认保存图片而不弹窗，保证命令行运行和自动验证不会停在窗口上。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS = {}
CHECKS = []
COLORS = ["#156082", "#E97132", "#196B24", "#A02B93", "#B8860B"]
ARTIFACT_DIR = None


def require(condition, message):
    """不能用可被 python -O 关闭的 assert 代替交付检查。"""
    if not condition:
        raise AssertionError(message)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(name, data):
    (ARTIFACT_DIR / name).write_text(
        json.dumps(json_ready(data), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n", encoding="utf-8")


def finish_figure(fig, name):
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def display(text):
    """将 Notebook 的文字分析改为终端输出及 UTF-8 文本文件。"""
    print(text, flush=True)
    with (ARTIFACT_DIR / "interpretation.txt").open("a", encoding="utf-8") as stream:
        stream.write(str(text) + "\n\n")


def Markdown(text):
    return text


def show_table(headers, rows):
    def fmt(value):
        return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
    display(" | ".join(headers) + "\n" + "\n".join(
        " | ".join(fmt(v) for v in row) for row in rows))


def mean_sd(values):
    # 单次实验没有样本标准差，不能将其伪装成零。
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def run_case(problem, optimizer_type, config, seed):
    """执行一次优化，重新计算最终种群，检查 X/F/C 对应关系。"""
    arguments = dict(config)
    generations = arguments.pop("max_gen")
    optimizer = optimizer_type(**arguments, seed=seed, record_every=max(1, generations // 10))
    started = time.perf_counter()
    X, F = optimizer.run(problem, max_gen=generations)
    C, CV = optimizer.final_constraints, optimizer.final_cv
    require(X.shape == (arguments["pop_size"], problem.n_var), "Invalid X shape")
    require(F.shape == (arguments["pop_size"], problem.n_obj), "Invalid F shape")
    require(np.isfinite(X).all() and np.isfinite(F).all(), "Nonfinite X or F")
    require(np.all(X >= problem.lower) and np.all(X <= problem.upper), "Bounds violated")
    reevaluated = problem.evaluate(X)
    check_F, check_C = (reevaluated if isinstance(reevaluated, tuple)
                        else (reevaluated, np.zeros((len(X), 0))))
    np.testing.assert_allclose(F, check_F, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(C, check_C, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(CV, C.sum(axis=1), rtol=0, atol=0)
    require(np.isfinite(C).all() and np.all(C >= 0), "Invalid constraint violations")
    require(optimizer.n_eval == arguments["pop_size"] * (generations + 1),
            "Wrong objective evaluation count")
    if hasattr(problem, "repair"):
        np.testing.assert_array_equal(X, problem.repair(X))
    return dict(seed=seed, config=dict(config), X=X, F=F, C=C, CV=CV,
                history=optimizer.history, seconds=time.perf_counter() - started,
                n_eval=optimizer.n_eval)


def feasible_mask(result):
    """同一个掩码用于 X、F 和 C，避免设计行与目标行错配。"""
    feasible = result["CV"] == 0
    mask = np.zeros(len(feasible), dtype=bool)
    mask[feasible] = non_dominated(result["F"][feasible])
    return mask


def feasible_front(result):
    return result["F"][feasible_mask(result)]


def reference_segments(reference):
    ordered = reference[np.argsort(reference[:, 0])]
    jumps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    return np.split(ordered, np.flatnonzero(jumps > .035) + 1)


def segment_coverage(F, reference, lo, hi, tolerance=.02):
    """距离阈值 .02 只作覆盖诊断，不是老师给出的评分线。"""
    if not len(F):
        return 0
    B = normalize_objectives(F, lo, hi)
    return sum(int(np.min(_nearest_distances(normalize_objectives(s, lo, hi), B))
                   <= tolerance) for s in reference_segments(reference))


def draw_reference(ax, reference, label="Reference front"):
    for i, piece in enumerate(reference_segments(reference)):
        ax.plot(piece[:, 0], piece[:, 1], color="#B52A35", lw=1.5,
                label=label if i == 0 else None)


def basic_config(args, n_var):
    return dict(pop_size=40 if args.quick else 100, pc=.9, pm=1/n_var,
                max_gen=60 if args.quick else 500)


def export_run(result, problem, stats):
    """完整最终种群和可行非支配子集分别保存。"""
    mask = feasible_mask(result)
    headers = [*[f"x{i+1}" for i in range(problem.n_var)],
               *[f"f{i+1}" for i in range(problem.n_obj)],
               *[f"C{i+1}" for i in range(result["C"].shape[1])], "CV", "feasible_nd"]
    matrix = np.column_stack((result["X"], result["F"], result["C"],
                              result["CV"], mask.astype(int)))
    for name, rows in [("population.csv", matrix), ("feasible_front.csv", matrix[mask])]:
        np.savetxt(ARTIFACT_DIR/name, rows, delimiter=",", header=",".join(headers), comments="")
    if hasattr(problem, "constraint_values"):
        G = problem.constraint_values(result["X"])
        np.savetxt(ARTIFACT_DIR/"signed_constraints.csv", G, delimiter=",",
                   header=",".join(f"g{i+1}" for i in range(G.shape[1])), comments="")
    stats.update(seed=result["seed"], config=result["config"], n_eval=result["n_eval"],
                 seconds=result["seconds"], feasible_ratio=float(np.mean(result["CV"] == 0)),
                 n_feasible_nd=int(mask.sum()), n_unique_nd=len(np.unique(result["F"][mask], axis=0)),
                 max_cv=float(np.max(result["CV"])), structure_checks="PASS")
    save_json("summary.json", stats)
    show_table(list(stats), [[stats[k] for k in stats]])
    if not mask.any():
        display("[WARN] No feasible nondominated solution. This is NOT a successful engineering result.")
    return stats


def main():
    global ARTIFACT_DIR
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(EXPERIMENTS), default=list(EXPERIMENTS)[-1],
                        help="Run an implemented task; default: newest task")
    parser.add_argument("--all", action="store_true", help="Run all tasks implemented in this file")
    parser.add_argument("--quick", action="store_true", help="Smoke experiment, NOT report-quality evidence")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--check-only", action="store_true", help="Deterministic checks without experiments")
    parser.add_argument("--out", type=Path, help="Output root; default: outputs/<script>/<full or quick>")
    args = parser.parse_args()
    for check in CHECKS:
        check()
        print(f"[PASS] {check.__name__}", flush=True)
    if args.check_only:
        return
    root = args.out or (Path(__file__).resolve().parent / "outputs" / Path(__file__).stem
                        / ("quick" if args.quick else "full"))
    for name in list(EXPERIMENTS) if args.all else [args.task]:
        ARTIFACT_DIR = root.resolve() / name
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR/"interpretation.txt").write_text("", encoding="utf-8")
        save_json("run_config.json", dict(task=name, quick=args.quick, seed=args.seed,
                  python=sys.version.split()[0], numpy=np.__version__, matplotlib=matplotlib.__version__))
        print(f"\n[RUN] {name} | {'QUICK smoke run' if args.quick else 'FULL experiment'}", flush=True)
        EXPERIMENTS[name](args)
        print(f"[DONE] Results: {ARTIFACT_DIR}", flush=True)
    print("\nPASS means explicit code checks passed; it does not mean a guaranteed course grade.")


# %% 阶段 01 老师提供的 ZDT3 模型（源代码保留）
class ZDT3():

    def __init__(self):
        self.name = 'ZDT3'
        self.n_obj = 2  # number of objectives
        self.n_var = 30  # number of decision variables

        self.lower = np.zeros(self.n_var)  # lower bound of decision variables
        self.upper = np.ones(self.n_var)  # upper bound of decision variables

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj)) # for each chromosom, we have 2 objective function evals.
        f[:, 0] = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        h = 1 - np.power(
            f[:, 0] * 1.0 / g,
            0.5) - (f[:, 0] * 1.0 / g) * np.sin(10 * np.pi * f[:, 0])
        f[:, 1] = g * h

        return f


# %% 阶段 01 本次补全的 NSGA-II 算法
"""Pure NumPy NSGA-II and constraint-dominance NSGA-II.

The problem interface is n_var, n_obj, lower, upper, evaluate(X), with
optional repair(X). evaluate returns F or (F, C); C contains nonnegative
constraint violation magnitudes, never raw signed constraint functions.
"""

import numpy as np


class NSGA2:
    """Elitist NSGA-II for minimization, with bounded real-coded variation.

    pc is the probability of SBX for a mating pair. pm is the independent
    mutation probability per variable (None means 1 / problem.n_var).
    max_gen counts offspring generations, so evaluations are N * (G + 1).
    The base class uses objective dominance; CNSGA2 also uses constraints.
    Every run resets the generator to seed, enabling reproducible reruns.
    """

    # 记录种群规模、交叉/变异概率和随机数种子。
    def __init__(self, pop_size=100, pc=0.9, pm=None, eta_c=20,
                 eta_m=20, seed=0, record_every=50):
        if not isinstance(pop_size, (int, np.integer)) or pop_size < 2:
            raise ValueError("pop_size must be an integer of at least 2")
        if not 0 <= pc <= 1 or (pm is not None and not 0 <= pm <= 1):
            raise ValueError("pc and pm must be probabilities in [0, 1]")
        if eta_c < 0 or eta_m < 0:
            raise ValueError("distribution indices must be nonnegative")
        if not isinstance(record_every, (int, np.integer)) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        self.pop_size = int(pop_size)
        self.pc = float(pc)
        self.pm = None if pm is None else float(pm)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.seed = seed
        self.record_every = int(record_every)
        self.rng = np.random.default_rng(seed)
        self.n_eval = 0
        self.history = []
        self.final_cv = None
        self.final_constraints = None

    @staticmethod
    # 逐目标比较：全部不差，且至少一个更好，才叫支配。
    def _objective_dominance(F):
        """D[i,j] means i dominates j; equal objective vectors do not dominate."""
        F = np.asarray(F, dtype=float)
        n = len(F)
        no_worse = np.ones((n, n), dtype=bool)
        strictly_better = np.zeros((n, n), dtype=bool)
        for column in F.T:
            no_worse &= column[:, None] <= column[None, :]
            strictly_better |= column[:, None] < column[None, :]
        return no_worse & strictly_better

    def _dominance_matrix(self, F, CV=None):
        return self._objective_dominance(F)

    # 一层一层取出尚未被支配的个体，得到 rank。
    def non_dominated_sort(self, F, CV=None):
        """Return zero-based ranks and a list of fronts of row indices."""
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or not np.isfinite(F).all():
            raise ValueError("F must be a finite two-dimensional array")
        dominates = self._dominance_matrix(F, CV)
        domination_count = dominates.sum(axis=0)
        rank = np.full(len(F), -1, dtype=int)
        front = np.flatnonzero(domination_count == 0)
        fronts = []
        while front.size:
            rank[front] = len(fronts)
            fronts.append(front)
            domination_count -= dominates[front].sum(axis=0)
            front = np.flatnonzero((domination_count == 0) & (rank < 0))
        if np.any(rank < 0):
            raise RuntimeError("Dominance sorting did not assign every row")
        return rank, fronts

    @staticmethod
    # 每个前沿内部估计稀疏程度，保留边界并归一化量纲。
    def crowding_distance(F, fronts):
        """Standard front-wise, objective-range-normalized crowding distance.

        An objective constant across a front contributes zero; it cannot
        distinguish any member and should not create arbitrary boundaries.
        """
        F = np.asarray(F, dtype=float)
        distance = np.zeros(len(F), dtype=float)
        for front in fronts:
            front = np.asarray(front, dtype=int)
            if len(front) <= 2:
                distance[front] = np.inf
                continue
            for j in range(F.shape[1]):
                ordered = front[np.argsort(F[front, j], kind="mergesort")]
                span = F[ordered[-1], j] - F[ordered[0], j]
                if span <= 0:
                    continue
                distance[ordered[[0, -1]]] = np.inf
                distance[ordered[1:-1]] += (
                    F[ordered[2:], j] - F[ordered[:-2], j]
                ) / span
        return distance

    def fitness_assignment(self, F, CV=None):
        ranks, fronts = self.non_dominated_sort(F, CV)
        return ranks, self.crowding_distance(F, fronts)

    # 二元锦标赛：先看 rank，再看拥挤距离。
    def tournament_selection(self, population_x, ranks, crowding):
        """Binary tournaments: lower rank, then larger crowding, random ties."""
        n = len(population_x)
        contestants = self.rng.integers(n, size=(self.pop_size, 2))
        a, b = contestants.T
        same_rank = ranks[a] == ranks[b]
        same_distance = crowding[a] == crowding[b]
        a_wins = (ranks[a] < ranks[b]) | (
            same_rank & ((crowding[a] > crowding[b]) |
                         (same_distance & (self.rng.random(len(a)) < 0.5)))
        )
        return np.asarray(population_x)[np.where(a_wins, a, b)].copy()

    def _repair(self, X, prob):
        original_shape = np.asarray(X).shape
        X = np.clip(X, prob.lower, prob.upper)
        if hasattr(prob, "repair"):
            repaired = prob.repair(X.copy())
            X = np.asarray(repaired, dtype=float)
        if X.ndim != 2 or X.shape != original_shape or X.shape[1] != prob.n_var:
            raise ValueError("repair must preserve the (population, n_var) shape")
        if (not np.isfinite(X).all() or np.any(X < prob.lower)
                or np.any(X > prob.upper)):
            raise ValueError("repair produced nonfinite or out-of-bounds variables")
        return X

    def initialize(self, prob):
        X = prob.lower + (prob.upper - prob.lower) * self.rng.random(
            (self.pop_size, prob.n_var))
        return self._repair(X, prob)

    # SBX 交叉：用两个父代产生两个有界实数子代。
    def crossover(self, parents, prob):
        """Deb's bounded simulated binary crossover, including odd sizes."""
        parents = np.asarray(parents, dtype=float)
        original_size = len(parents)
        if original_size % 2:
            parents = np.vstack((parents, parents[:1]))
        p1, p2 = parents[0::2], parents[1::2]
        q1, q2 = p1.copy(), p2.copy()
        y1, y2 = np.minimum(p1, p2), np.maximum(p1, p2)
        delta = y2 - y1
        active = ((self.rng.random((len(p1), 1)) < self.pc)
                  & (self.rng.random(p1.shape) < 0.5)
                  & (delta > 1e-14)
                  & (np.asarray(prob.upper) > np.asarray(prob.lower)))
        # Inactive coordinates use a safe denominator and are never replaced.
        denominator = np.where(active, delta, 1.0)
        random_value = self.rng.random(p1.shape)
        power = 1.0 / (self.eta_c + 1.0)

        beta_lower = 1.0 + 2.0 * (y1 - prob.lower) / denominator
        alpha_lower = 2.0 - beta_lower ** (-(self.eta_c + 1.0))
        beta_q_lower = np.where(
            random_value <= 1.0 / alpha_lower,
            (random_value * alpha_lower) ** power,
            (1.0 / (2.0 - random_value * alpha_lower)) ** power)
        beta_upper = 1.0 + 2.0 * (prob.upper - y2) / denominator
        alpha_upper = 2.0 - beta_upper ** (-(self.eta_c + 1.0))
        beta_q_upper = np.where(
            random_value <= 1.0 / alpha_upper,
            (random_value * alpha_upper) ** power,
            (1.0 / (2.0 - random_value * alpha_upper)) ** power)
        c1 = 0.5 * (y1 + y2 - beta_q_lower * delta)
        c2 = 0.5 * (y1 + y2 + beta_q_upper * delta)
        swap = self.rng.random(p1.shape) < 0.5
        q1[active] = np.where(swap, c2, c1)[active]
        q2[active] = np.where(swap, c1, c2)[active]
        offspring = np.empty_like(parents)
        offspring[0::2], offspring[1::2] = q1, q2
        return np.clip(offspring[:original_size], prob.lower, prob.upper)

    # 多项式变异：每个变量独立按 pm 概率改变，然后修复边界。
    def mutation(self, offspring, prob):
        """Bounded polynomial mutation; pm is a per-variable probability."""
        X = np.array(offspring, dtype=float, copy=True)
        probability = 1.0 / prob.n_var if self.pm is None else self.pm
        width = np.asarray(prob.upper) - np.asarray(prob.lower)
        active = (self.rng.random(X.shape) < probability) & (width > 0)
        safe_width = np.where(width > 0, width, 1.0)
        delta1 = (X - prob.lower) / safe_width
        delta2 = (prob.upper - X) / safe_width
        random_value = self.rng.random(X.shape)
        power = 1.0 / (self.eta_m + 1.0)
        lower_value = (2.0 * random_value + (1.0 - 2.0 * random_value)
                       * (1.0 - delta1) ** (self.eta_m + 1.0))
        upper_value = (2.0 * (1.0 - random_value)
                       + 2.0 * (random_value - 0.5)
                       * (1.0 - delta2) ** (self.eta_m + 1.0))
        # Each branch is meaningful only on its own half of [0,1].
        delta_q = np.where(random_value <= 0.5,
                           np.maximum(lower_value, 0.0) ** power - 1.0,
                           1.0 - np.maximum(upper_value, 0.0) ** power)
        X += np.where(active, delta_q * safe_width, 0.0)
        return self._repair(X, prob)

    # 将父代与子代合并，在 2N 个候选中选回 N 个。
    def environmental_selection(self, X, F, C=None):
        """Select N elites from the combined parents and offspring."""
        X, F = np.asarray(X), np.asarray(F)
        C = (np.zeros((len(X), 0)) if C is None
             else np.asarray(C, dtype=float))
        if C.ndim == 1:
            C = C[:, None]
        CV = C.sum(axis=1)
        ranks, fronts = self.non_dominated_sort(F, CV)
        distance = self.crowding_distance(F, fronts)
        selected = []
        for front in fronts:
            remaining = self.pop_size - len(selected)
            if len(front) <= remaining:
                selected.extend(front.tolist())
            else:
                # Random ordering breaks equal-distance ties reproducibly.
                shuffled = self.rng.permutation(front)
                best = shuffled[np.argsort(-distance[shuffled], kind="mergesort")]
                selected.extend(best[:remaining].tolist())
                break
            if len(selected) == self.pop_size:
                break
        selected = np.asarray(selected, dtype=int)
        return X[selected].copy(), F[selected].copy(), C[selected].copy()

    # 统一问题接口，并检查目标值和非负约束违反量。
    def _evaluate(self, prob, X):
        result = prob.evaluate(X)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("evaluate must return F or (F, C)")
            F, C = result
            C = np.asarray(C, dtype=float)
            if C.ndim == 1:
                C = C[:, None]
        else:
            F = result
            C = np.zeros((len(X), 0))
        F = np.asarray(F, dtype=float)
        if F.shape != (len(X), prob.n_obj) or not np.isfinite(F).all():
            raise ValueError("evaluate returned invalid objective values or shape")
        if (C.ndim != 2 or C.shape[0] != len(X)
                or not np.isfinite(C).all() or np.any(C < 0)):
            raise ValueError("C must contain finite nonnegative violation magnitudes")
        self.n_eval += len(X)
        return F, C

    def _record(self, generation, F, C):
        CV = C.sum(axis=1)
        self.history.append({
            "gen": int(generation), "n_eval": int(self.n_eval),
            "feasible_ratio": float(np.mean(CV == 0.0)),
            "cv_min": float(np.min(CV)),
            "F": F.copy(), "CV": CV.copy(),
        })

    # 初始化后循环执行选择、交叉、变异、评价和精英保留。
    def run(self, prob, max_gen=500):
        if not isinstance(max_gen, (int, np.integer)) or max_gen < 0:
            raise ValueError("max_gen must be a nonnegative integer")
        lower, upper = np.asarray(prob.lower), np.asarray(prob.upper)
        if (lower.shape != (prob.n_var,) or upper.shape != (prob.n_var,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(lower > upper)):
            raise ValueError("problem bounds must be finite valid n_var-vectors")
        self.rng = np.random.default_rng(self.seed)
        self.n_eval, self.history = 0, []
        population_x = self.initialize(prob)
        population_fx, population_c = self._evaluate(prob, population_x)
        self._record(0, population_fx, population_c)
        for generation in range(1, max_gen + 1):
            ranks, distance = self.fitness_assignment(
                population_fx, population_c.sum(axis=1))
            parents = self.tournament_selection(population_x, ranks, distance)
            offspring_x = self.mutation(self.crossover(parents, prob), prob)
            offspring_fx, offspring_c = self._evaluate(prob, offspring_x)
            population_x, population_fx, population_c = self.environmental_selection(
                np.vstack((population_x, offspring_x)),
                np.vstack((population_fx, offspring_fx)),
                np.vstack((population_c, offspring_c)))
            if generation % self.record_every == 0 or generation == max_gen:
                self._record(generation, population_fx, population_c)
        self.final_cv = population_c.sum(axis=1).copy()
        self.final_constraints = population_c.copy()
        return population_x, population_fx


# %% 阶段 01 公共指标与 ZDT3 参考前沿
def _matrix(values, n_obj=None, name="points"):
    values = np.asarray(values, dtype=float)
    if values.size == 0 and values.ndim == 1:
        values = values.reshape(0, 0 if n_obj is None else n_obj)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if n_obj is not None and values.shape[1] != n_obj:
        raise ValueError(f"{name} must have {n_obj} objectives")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def non_dominated(F):
    """Boolean mask of Pareto-nondominated rows; minimize every objective.

    Equal duplicates do not dominate one another and receive the same mask.
    The 2-objective implementation sorts in O(N log N), permitting dense MW7
    reference construction; other dimensions use an O(N**2) comparison loop.
    """
    F = _matrix(F)
    n, m = F.shape
    if n == 0:
        return np.zeros(0, dtype=bool)
    if m == 0:
        raise ValueError("At least one objective is required")
    if m == 1:
        return F[:, 0] == np.min(F[:, 0])
    if m == 2:
        unique, inverse = np.unique(F, axis=0, return_inverse=True)
        previous_best = np.r_[np.inf, np.minimum.accumulate(unique[:-1, 1])]
        return (unique[:, 1] < previous_best)[inverse]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = not np.any(np.all(F <= F[i], axis=1)
                             & np.any(F < F[i], axis=1))
    return keep


def normalize_objectives(F, lo, hi):
    """Apply fixed affine objective scaling; never silently clip values."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if lo.ndim != 1 or hi.shape != lo.shape or lo.size == 0:
        raise ValueError("lo and hi must be matching nonempty vectors")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Bounds must be finite")
    if np.any(hi <= lo):
        raise ValueError("Every hi must be strictly greater than lo")
    return (_matrix(F, lo.size) - lo) / (hi - lo)


def _nearest_distances(A, B, chunk_size=256):
    if len(B) == 0:
        return np.full(len(A), np.inf)
    result = np.empty(len(A))
    for start in range(0, len(A), chunk_size):
        delta = A[start:start + chunk_size, None, :] - B[None, :, :]
        result[start:start + chunk_size] = np.sqrt(
            np.min(np.sum(delta * delta, axis=2), axis=1))
    return result


def normalized_igd(ref, F, lo, hi):
    """Mean normalized Euclidean distance from reference front to approximation.

    Lower is better relative to the SAME reference sample and normalization.
    This is mean distance (IGD with p=1), not RMS distance. Empty F returns inf;
    an empty reference raises ValueError rather than inventing an ideal front.
    """
    R = normalize_objectives(ref, lo, hi)
    A = normalize_objectives(F, lo, hi)
    if not len(R):
        raise ValueError("IGD requires a nonempty reference front")
    return float(np.mean(_nearest_distances(R, A)))


def normalized_gd(ref, F, lo, hi):
    """Mean normalized Euclidean distance from approximation to reference.

    This is mean distance (GD with p=1). It measures proximity only: one good
    point can have excellent GD while covering almost none of the front.
    Empty F returns inf; an empty reference raises ValueError.
    """
    R = normalize_objectives(ref, lo, hi)
    A = normalize_objectives(F, lo, hi)
    if not len(R):
        raise ValueError("GD requires a nonempty reference front")
    if not len(A):
        return float("inf")
    return float(np.mean(_nearest_distances(A, R)))


def normalized_spacing(F, lo, hi):
    """Return mean_nn, cv_nn and n using normalized Euclidean NN distances.

    cv_nn is population standard deviation divided by mean NN distance.
    Neither statistic alone establishes better diversity: clustering may give
    a tiny CV, while an isolated outlier may raise mean_nn. Read them together
    with coverage, feasible nondominated count and HV/IGD. Duplicates remain
    visible as zero NN distances. For <2 points both statistics are NaN; if all
    NN distances are zero, mean_nn=0 and cv_nn=NaN (division is undefined).
    """
    A = normalize_objectives(F, lo, hi)
    n = len(A)
    if n < 2:
        return {"mean_nn": float("nan"), "cv_nn": float("nan"), "n": n}
    d = np.sqrt(np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    nearest = np.min(d, axis=1)
    mean = float(np.mean(nearest))
    return {"mean_nn": mean,
            "cv_nn": float(np.std(nearest) / mean) if mean > 0 else float("nan"),
            "n": n}


ZDT3_INTERVALS = np.array([
    [0.0, 0.0830015349],
    [0.1822287280, 0.2577623634],
    [0.4093136748, 0.4538821041],
    [0.6183967944, 0.6525117038],
    [0.8233317983, 0.8518328654],
])


def zdt3_reference_front(n_points=2000):
    """Sample the five analytic ZDT3 front segments (g=1).

    Endpoints are standard roots rounded to 10 decimals. Samples are allocated
    approximately uniformly by objective-space arc length, avoiding excessive
    weight near steep/short segments. Arc length uses a dense deterministic
    numerical table; front coordinates themselves use the exact g=1 formula.
    Segment starts at equal-f2 joins are interpreted as the front's closure.
    """
    if int(n_points) != n_points or n_points < 10:
        raise ValueError("n_points must be an integer >=10")
    n_points = int(n_points)
    tables, lengths = [], []
    for left, right in ZDT3_INTERVALS:
        t = np.linspace(0.0, 1.0, 4001)
        x = left + (right - left) * (t ** 2 if left == 0 else t)
        y = 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
        distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y))) ]
        tables.append((x, distance))
        lengths.append(distance[-1])
    allocation = (n_points - 10) * np.array(lengths) / np.sum(lengths)
    counts = np.floor(allocation).astype(int) + 2
    remainder = n_points - counts.sum()
    if remainder:
        counts[np.argsort(-(allocation - np.floor(allocation)))[:remainder]] += 1
    fronts = []
    for count, (x, distance) in zip(counts, tables):
        sampled_x = np.interp(np.linspace(0, distance[-1], count), distance, x)
        sampled_y = 1.0 - np.sqrt(sampled_x) - sampled_x * np.sin(10*np.pi*sampled_x)
        fronts.append(np.column_stack((sampled_x, sampled_y)))
    return np.vstack(fronts)


# %% 阶段 01 手算验证与 ZDT3 实验
def check_nsga2_core():
    """手算例子：用独立答案检查排序、距离与精英保留。"""
    F = np.array([[0, 4], [1, 3], [2, 2], [4, 0], [3, 4], [5, 5]], dtype=float)
    opt = NSGA2(pop_size=3, seed=14)
    ranks, fronts = opt.non_dominated_sort(F)
    np.testing.assert_array_equal(ranks, [0, 0, 0, 0, 1, 2])
    distance = opt.crowding_distance(F, fronts)
    np.testing.assert_allclose(distance[1:3], [1, 1.5])
    require(np.isinf(distance[[0, 3]]).all(), "Front endpoints must be protected")
    X, selected_F, _ = opt.environmental_selection(np.arange(6)[:, None], F)
    require(set(X[:, 0]) == {0, 2, 3}, "Wrong elitist truncation")
    np.testing.assert_array_equal(selected_F, F[X[:, 0]])
    require(not np.diag(opt._objective_dominance(F)).any(), "Self dominance")
    a = NSGA2(pop_size=7, seed=14)
    X1, F1 = a.run(ZDT3(), max_gen=3)
    X2, F2 = a.run(ZDT3(), max_gen=3)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(F1, F2)
    require(a.n_eval == 28, "Initialization must count toward evaluations")


CHECKS.append(check_nsga2_core)


def run_zdt3(args):
    problem = ZDT3()
    result = run_case(problem, NSGA2, basic_config(args, problem.n_var), args.seed)
    reference = zdt3_reference_front(2000)
    lo, hi = reference.min(axis=0), reference.max(axis=0)
    front = feasible_front(result)
    stats = dict(problem="ZDT3", gd=normalized_gd(reference, front, lo, hi),
                 igd=normalized_igd(reference, front, lo, hi),
                 segments_hit=segment_coverage(front, reference, lo, hi), total_segments=5,
                 median_g=float(np.median(1 + 9*np.mean(result["X"][:, 1:], axis=1))))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    draw_reference(axes[0], reference, "Analytic ZDT3 front")
    axes[0].scatter(*result["F"].T, s=17, label=f"Final population, seed {args.seed}")
    axes[0].set(xlabel="f1", ylabel="f2", title="ZDT3: final population")
    axes[0].legend(fontsize=8)
    curve = [normalized_igd(reference, h["F"], lo, hi) for h in result["history"]]
    axes[1].plot([h["n_eval"] for h in result["history"]], curve, marker=".")
    axes[1].set(xlabel="Objective evaluations", ylabel="Normalized IGD", yscale="log",
                title="Convergence (lower is better)")
    finish_figure(fig, "zdt3_results")
    export_run(result, problem, stats)
    display("ZDT3: g approaches 1 on the true front. Check IGD AND all five segments. "
            "The .02 segment-hit tolerance is diagnostic, not an official grading threshold.")


EXPERIMENTS["zdt3"] = run_zdt3


# %% 阶段 02 新增三目标 HV 工具
def _hypervolume_2d(F, reference):
    """Area of the union of minimization rectangles, with valid corners only."""
    if len(F) == 0:
        return 0.0
    order = np.argsort(F[:, 0], kind="stable")
    sorted_F = F[order]
    best_y = np.minimum.accumulate(sorted_F[:, 1])
    widths = np.diff(np.r_[sorted_F[:, 0], reference[0]])
    return float(np.dot(widths, reference[1] - best_y))


def hypervolume_3d(F, reference=1.1):
    """EXACT 3D dominated hypervolume by an x sweep and exact 2D slices.

    F must contain normalized, feasible objective vectors, all minimized.
    Feasibility cannot be inferred here: the caller must select feasible rows.
    The function defensively removes dominated rows and equal duplicates.
    reference is a fixed scalar or length-3 vector, normally [1.1]*3 shared by
    ALL configurations. Points at or beyond reference in ANY coordinate cannot
    dominate a positive-volume box and are excluded, not projected inwards.
    Negative coordinates are allowed and not clipped; they mean an observation
    improved on the fixed lower normalization bound. The returned volume is in
    normalized objective units and is NOT divided by 1.1**3. This is not a Monte
    Carlo estimate. Larger is better with a common scale and reference point.
    """
    F = _matrix(F, 3)
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 0:
        reference = np.full(3, float(reference))
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference must be a finite scalar or length-3 vector")
    F = F[np.all(F < reference, axis=1)]
    if not len(F):
        return 0.0
    F = np.unique(F[non_dominated(F)], axis=0)
    x_values = np.unique(F[:, 0])
    widths = np.diff(np.r_[x_values, reference[0]])
    volume = 0.0
    for x, width in zip(x_values, widths):
        volume += width * _hypervolume_2d(F[F[:, 0] <= x, 1:], reference[1:])
    return float(volume)


def normalized_hypervolume_3d(F, lo, hi, reference=1.1):
    """Convenience wrapper retaining a fixed shared normalization and reference."""
    return hypervolume_3d(normalize_objectives(F, lo, hi), reference)


# %% 阶段 02 老师提供的汽车模型（源代码保留）
class CrashworthinessDesign():

    def __init__(self):
        self.name = 'Crashworthiness design of vehicles'
        self.n_obj = 3
        self.n_var = 5

        self.lower = np.full(self.n_var, 1.0)
        self.upper = np.full(self.n_var, 3.0)

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj))

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        f[:, 0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (
            4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f[:, 1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (
            0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (
                0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3) + (
                    0.1764 * x4 * x4)
        f[:, 2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (
            0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (
                0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (
                    0.0109 * x4 * x4)

        return f


# %% 阶段 02 新增汽车设计实验
def run_crash(args):
    problem = CrashworthinessDesign()
    result = run_case(problem, NSGA2, basic_config(args, problem.n_var), args.seed)
    front = feasible_front(result)
    # 本阶段只有一次实验；这一尺度只用于该图，不跨运行比较 HV。
    lo, hi = result["F"].min(axis=0), result["F"].max(axis=0)
    require(np.all(hi > lo), "Degenerate objectives: cannot normalize")
    spacing = normalized_spacing(front, lo, hi)
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(*result["F"].T, s=18, c=result["F"][:, 0], cmap="viridis")
    ax.set(xlabel="Weight", ylabel="Acceleration", zlabel="Intrusion", title="Crashworthiness")
    ax.view_init(elev=23, azim=-52)
    ax2 = fig.add_subplot(122)
    for row in normalize_objectives(front, lo, hi):
        ax2.plot(range(3), row, alpha=.25, color=COLORS[0])
    ax2.set(xticks=range(3), xticklabels=["Weight", "Acceleration", "Intrusion"],
            ylabel="Normalized value (this run only)", title="Trade-offs")
    finish_figure(fig, "crash_results")
    export_run(result, problem, dict(problem="Crashworthiness", nn_cv=spacing["cv_nn"],
               normalization_scope="this run only; do not compare to other runs", lo=lo, hi=hi))
    display("Uneven spacing can remain on a curved three-objective surface: crowding distance "
            "sums one-dimensional neighbour gaps and protects boundaries. Raw objective scale "
            "alone is not the cause because crowding is range-normalized. Larger populations "
            "or alternative diversity rules are suggestions, not improvements tested here.")


EXPERIMENTS["crash"] = run_crash


# %% 阶段 03 新增可行性优先的 CNSGA2
class CNSGA2(NSGA2):
    """NSGA-II with parameter-less constraint dominance.

    Feasible solutions dominate infeasible ones. Between infeasible solutions,
    lower total violation wins. Between feasible solutions, objective Pareto
    dominance applies. Equal positive violation does not break ties by F.
    """

    def _dominance_matrix(self, F, CV=None):
        if CV is None:
            return self._objective_dominance(F)
        CV = np.asarray(CV, dtype=float).reshape(-1)
        if CV.shape != (len(F),) or np.any(CV < 0) or not np.isfinite(CV).all():
            raise ValueError("CV must be a finite nonnegative vector")
        feasible = CV == 0.0
        objective_dominance = self._objective_dominance(F)
        both_feasible = feasible[:, None] & feasible[None, :]
        feasible_over_infeasible = feasible[:, None] & ~feasible[None, :]
        both_infeasible = ~feasible[:, None] & ~feasible[None, :]
        lower_violation = CV[:, None] < CV[None, :]
        return ((both_feasible & objective_dominance)
                | feasible_over_infeasible | (both_infeasible & lower_violation))


# %% 阶段 03 老师提供的 MW7 模型（源代码保留）
class MW7():

    def __init__(self):
        self.name = 'MW7'
        self.n_obj = 2  # number of objectives
        self.n_var = 15  # number of decision variables
        self.n_con = 2  # number of constraints

        self.lower = np.zeros(self.n_var)
        self.upper = np.ones(self.n_var)

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj))
        c = np.zeros((pop_size, self.n_con))

        g3 = 1 + 2.0 * np.sum((x[:, self.n_obj - 1:] +
                               (x[:, self.n_obj - 2:-1] - 0.5)**2 - 1.0)**2,
                              axis=1)

        f[:, 0] = g3 * x[:, 0]
        f[:, 1] = g3 * (1 - (f[:, 0] / g3)**2)**0.5

        with np.errstate(divide='ignore'):
            l = np.arctan(f[:, 1] / f[:, 0])

        c[:, 0] = (1.2 + 0.4 * (np.sin(4 * l))**16)**2 - f[:, 0]**2 - f[:, 1]**2
        c[:, 1] = f[:, 0]**2 + f[:, 1]**2 - (1.15 - 0.2 * (np.sin(4 * l))**8)**2
        c = np.where(c < 0, -c, 0)  # the degree of constraint violation of x at each constraint

        return f, c


# %% 阶段 03 MW7 数值参考前沿
def mw7_reference_front(n_angles=100001, n_points=None):
    """Dense NUMERICAL reference front derived from the supplied MW7 equations.

    Set theta=atan2(f2,f1) in [0,pi/2]. The radial objectives have radius g3>=1.
    Feasibility requires 1.15-0.2*sin(4*theta)**8 <= g3 <=
    1.2+0.4*sin(4*theta)**16. Thus the feasible radial lower boundary has
    r=max(1, 1.15-0.2*sin(4*theta)**8). Some points on that boundary are globally
    dominated, so they MUST be filtered. We sample the boundary and retain its
    globally nondominated subset. This is a dense numerical approximation to
    the constrained front, not a claim that every radial-boundary point is PF.

    The bound g3=1 is attainable by x[i]=1-(x[i-1]-.5)**2. Required radii up to
    1.15 are attainable by lowering only the last variable by sqrt((r-1)/2),
    which remains inside [0,1]. Optional n_points uniformly subsamples the
    retained index sequence (not a curve interpolated across front gaps).
    """
    if int(n_angles) != n_angles or n_angles < 1001:
        raise ValueError("n_angles must be an integer >=1001")
    theta = np.linspace(0.0, np.pi / 2, int(n_angles))
    wave = np.sin(4 * theta) ** 8
    radius = np.maximum(1.0, 1.15 - 0.2 * wave)
    F = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    # Exact endpoints avoid floating residuals from cos(pi/2).
    F[0, 1] = 0.0
    F[-1, 0] = 0.0
    F = F[non_dominated(F)]
    F = F[np.argsort(F[:, 0])]
    if n_points is not None:
        if int(n_points) != n_points or n_points < 2:
            raise ValueError("n_points must be an integer >=2")
        if n_points < len(F):
            F = F[np.round(np.linspace(0, len(F)-1, int(n_points))).astype(int)]
    return F


# %% 阶段 03 约束规则验证与 MW7 实验
def check_constraint_rules():
    F = np.array([[100, 100], [0, 0], [-1, -1], [200, 200], [101, 101]])
    CV = np.array([0., 1., 2., 1., 0.])
    opt = CNSGA2(pop_size=2)
    ranks, _ = opt.non_dominated_sort(F, CV)
    np.testing.assert_array_equal(ranks, [0, 2, 3, 2, 1])
    dominance = opt._dominance_matrix(F, CV)
    require(not dominance[1, 3] and not dominance[3, 1], "Equal positive CV must tie")
    X, _, C = opt.environmental_selection(np.arange(5)[:, None], F, CV[:, None])
    require(set(X[:, 0]) == {0, 4}, "Feasible members must beat infeasible members")
    require(np.all(C == 0), "Wrong constrained elites")
    _, violations = MW7().evaluate(np.zeros((2, 15)))
    require(np.all(violations >= 0), "MW7 already returns nonnegative violations")


CHECKS.append(check_constraint_rules)


def run_mw7(args):
    problem = MW7()
    result = run_case(problem, CNSGA2, basic_config(args, problem.n_var), args.seed)
    reference = mw7_reference_front(n_points=2000)
    lo, hi = reference.min(axis=0), reference.max(axis=0)
    front = feasible_front(result)
    stats = dict(problem="MW7", gd=normalized_gd(reference, front, lo, hi),
                 igd=normalized_igd(reference, front, lo, hi),
                 segments_hit=segment_coverage(front, reference, lo, hi), total_segments=3)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    draw_reference(axes[0], reference, "Numerical constrained front")
    feasible = result["CV"] == 0
    axes[0].scatter(*result["F"][feasible].T, s=17, label="Feasible final members")
    if np.any(~feasible):
        axes[0].scatter(*result["F"][~feasible].T, s=20, marker="x", label="Infeasible members")
    axes[0].set(xlabel="f1", ylabel="f2", title="MW7: feasibility and coverage")
    axes[0].legend(fontsize=8)
    curve = [normalized_igd(reference, h["F"][h["CV"] == 0], lo, hi)
             for h in result["history"]]
    axes[1].plot([h["n_eval"] for h in result["history"]],
                 [v if np.isfinite(v) else np.nan for v in curve], marker=".")
    axes[1].set(xlabel="Objective evaluations", ylabel="Feasible-set IGD", yscale="log",
                title="Missing feasible sets leave gaps")
    finish_figure(fig, "mw7_results")
    export_run(result, problem, stats)
    if stats["segments_hit"] < 3:
        display("[WARN] Not all three front segments were reached. Feasibility does not prove coverage.")
    display("Disconnected feasible regions and finite-population search can lose segments. "
            "A low GD with high IGD is consistent with partial coverage; more generations do not "
            "guarantee recovery. Compare multiple seeds before drawing general conclusions.")


EXPERIMENTS["mw7"] = run_mw7


if __name__ == "__main__":
    main()
