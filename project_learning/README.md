# CEG5302 第14组逐阶段 Python 学习项目

这个文件夹把课程作业整理成“保留上一版，再增加下一阶段功能”的 Python 文件。每个完成阶段都是独立完整版本，不导入其它阶段文件，不需要打开 Notebook，也不需要读取课程 PDF 才能运行。

先阅读 `01_nsga2_zdt3.py`。所有完成阶段都可直接运行，默认执行该阶段新增加的实验；`--all` 执行该文件已经实现的全部实验。图表会自动保存，不会弹出窗口阻塞程序。

## 文件与四道题的对应关系

| 文件 | 在上一阶段上增加什么 | 对应题目 | 文件行数 |
|---|---|---|---:|
| `00_original.py` | 从老师原始 Notebook 逐代码单元原样导出，保留空方法及 `??` | 原始题目模板 | 见源文件 |
| `01_nsga2_zdt3.py` | 非支配排序、拥挤距离、锦标赛、SBX、变异、精英保留和 ZDT3 实验 | 第1题，30分 | 744 |
| `02_crashworthiness.py` | 汽车模型、三目标图和平行坐标图 | 第2题，10分 | 859 |
| `03_constraints_mw7.py` | 可行性优先规则、约束 NSGA-II、MW7 实验 | 第3题，30分 | 1008 |
| `04_rcm13.py` | 第14组齿轮箱的目标、约束、齿数修复、优化和设计表 | 第4题的模型与求解 | 1186 |
| `05_parameter_experiments.py` | 重复运行、四因素比较、等计算预算对照、独立种子验证 | 第4题的参数实验与分析 | 1373 |
| `verify_all.py` | 独立标准答案测试、阶段递增检查、独立进程运行和导出数据复算 | 全部阶段 | 见源文件 |

表中行数包含空行、注释和说明。后一份文件包含前一份的公共代码，所以不能把五个文件的行数相加，当成互不重复的新代码量。最终阶段文件为1373行。

`00_original.py` **不是可运行的完成版**：老师原模板本来就有缺失方法体和 `NSGA2(pop_size=100, ??)`。为便于对照，这里没有悄悄修复这些内容，直接运行可能出现 `IndentationError` 或 `SyntaxError`。验证器会确认它忠实保留模板，并将其记为 `PRESERVED_INCOMPLETE`；不会把它当作完成阶段执行。

辅助文件：

- `changes/00_to_01.diff` 至 `changes/04_to_05.diff`：每阶段相对上一版的逐行差异，`+` 表示增加，`-` 表示删除。
- `source_manifest.json`：原始来源、导出指纹和各阶段记录，用于核对来源。
- `requirements.txt`：运行依赖。只需要 NumPy 和 Matplotlib；验证使用 Python 自带的 unittest。
- `verification/`：运行 `verify_all.py` 后生成的日志、数据、图片和 JSON 报告；不随源码上传。

## 本次实际验证结果

2026年10月4日已运行 `python verify_all.py` 和 `python verify_all.py --full`。32项单元/结构检查全部通过，五个阶段均已在只含自身源码的临时目录中成功运行。

| 验证或实验 | 本次记录 |
|---|---|
| 快速完整流程 | 约9.2秒；小预算下的前沿覆盖警告已保留 |
| 完整规模流程 | 约57.1秒；五阶段运行、表格复算和输出检查通过 |
| ZDT3，固定种子14 | 命中5段；IGD约0.003506；median(g)约1.000037 |
| MW7，固定种子14 | 可行率100%；命中3段；IGD约0.005009 |
| RCM参数实验 | 70次优化；本次按平均HV选出 N=200 |
| RCM独立验证 | 基准平均HV约1.249929；所选配置约1.254792；可行率均为100% |

耗时仅是本机记录。固定种子14的 MW7 覆盖良好，不代表换种子也一定覆盖完整。这是上传前的本地验证记录。仓库只包含源码；下载后运行 `python verify_all.py --full` 会生成新的 `verification/full/verification_report.json` 和实验 `summary.json`，供核对本机结果。

## 准备环境

已经验证的版本是 Python 3.12.14、NumPy 2.5.3、Matplotlib 3.11.2。建议在 Python 3.12 环境中安装依赖。先执行 `python --version`，确认它不是 Python 2。

先将仓库克隆或下载到本地。在 PowerShell 中从仓库的上一级目录进入此文件夹，然后安装依赖：

```powershell
cd ceg5302-group14/project_learning
python -m pip install -r requirements.txt
```

也可以自行创建虚拟环境，再安装同一个依赖文件。这些源码本身没有写死本机路径，把单个完成阶段复制到另一台已安装依赖的电脑仍可运行。

## 按顺序运行

```powershell
python 01_nsga2_zdt3.py
python 02_crashworthiness.py
python 03_constraints_mw7.py
python 04_rcm13.py
python 05_parameter_experiments.py
```

前四阶段默认 `N=100`、`G=500`、`pc=0.9`、`pm=1/n_var`，种子为14。每次默认只做一次该问题的实验，以便逐步学习。第5阶段的 RCM 参数实验默认重复五个种子，完整矩阵共70次优化。

初次尝试可以先运行快速模式：

```powershell
python 01_nsga2_zdt3.py --quick
python 05_parameter_experiments.py --quick
```

快速模式前四题使用 `N=40`、`G=60`；第5阶段使用较小配置、两个调参种子、两个验证种子，共28次 RCM 优化。它用于确认流程和输出，**不应拿来声称已经收敛，或替代完整报告实验**。出现“前沿没有覆盖完整”的警告，可能正是小预算的实际表现。

后面的文件仍然能运行前面的题：

```powershell
python 04_rcm13.py --task zdt3
python 04_rcm13.py --task crash
python 04_rcm13.py --task mw7
python 05_parameter_experiments.py --all
```

可用任务名按阶段逐步增加：`zdt3`、`crash`、`mw7`、`rcm13`、`parameters`。`--all` 表示运行本文件已有的所有任务；默认只运行最新任务。

其它参数：

```powershell
python 03_constraints_mw7.py --seed 114
python 04_rcm13.py --check-only
python 01_nsga2_zdt3.py --out './my_results'
python 05_parameter_experiments.py --help
```

改变种子可以观察随机搜索的差异。第5阶段以 `--seed` 为起点，每次加100构造调参种子，验证种子在此基础上加1000。默认正好是14、114、214、314、414，以及1014、1114、1214、1314、1414。

## 运行后去哪里看结果

正常运行的结果默认在：

```text
outputs/
  01_nsga2_zdt3/full/zdt3/
    zdt3_results.png      最终种群和收敛曲线
    population.csv        完整最终种群及其目标和约束违反量
    feasible_front.csv    可行非支配子集，设计行和目标行严格对应
    summary.json          参数、种子、指标、可行率、耗时等
    run_config.json       模式和软件版本
    interpretation.txt   数据表和结果解释
```

其它阶段遵循同样结构。齿轮箱另有 `signed_constraints.csv`，保存原始不等式值 `g`；非负违反量是 `C=max(g,0)`，两者不要混淆。

第5阶段的 `parameters/` 中包含参数比较图、最终解与收敛图、60次调参/对照运行的 `rcm_experiments.csv`、10次独立验证的 `rcm_holdout.csv`、最终设计的 `rcm_final_designs.csv`，以及记录统一尺度、经验参考集和选参依据的 `summary.json`。快速模式相应为24次加4次。

再次运行同一文件、模式和任务会覆盖该目录中同名结果。若需要保留旧实验，使用不同的 `--out` 路径。快速结果和完整结果默认分开存放。

## 一条命令验证什么

```powershell
python verify_all.py             # 32项检查 + 五阶段小规模运行
python verify_all.py --full      # 32项检查 + 五阶段完整规模运行
python verify_all.py --unit-only # 只检查逻辑、公式和结构，不运行完整实验
```

验证器会把每个阶段文件单独复制到临时目录，并从该目录执行，以确认没有偷偷依赖相邻文件、Notebook 或构建脚本。第5阶段会使用 `--all`，同时检查它仍保留了前四题的能力。

验证包含：

1. 支配排序、重复点、归一化拥挤距离、精英保留、随机种子复现、奇数种群、边界与固定变量。
2. 可行解优先、较小违反量优先、相等正违反量不按目标强行分胜负。
3. RCM13 独立高精度参考点、11个约束、离散修复和不同来源模式的区别。
4. GD、IGD、HV 的小规模独立标准答案；HV 还与容斥法对照。
5. 老师三个模型保持原样、后阶段保留前阶段实现、导入模块不会自动跑实验。
6. 子进程退出状态、图像是否生成、输出表格的 X/F/C 重新计算是否一致、等预算评价次数相等、调参与验证种子不重叠。
7. 完整模式中，对预先固定的种子进行可行率及 ZDT3 收敛回归检查。这些阈值是本项目的回归检查，不是老师的评分标准。

日志写入 `verification/quick/` 或 `verification/full/`。总结果是 `verification_report.json`，每个阶段另有 `.log` 文件和输出子目录。

- `[PASS]`：明确写出的检查通过。
- `[WARN]`：实验有需要解释的表现，例如漏掉一段前沿。
- `[FAIL]`：检查失败，查看对应日志，程序返回非零退出码。
- `PRESERVED_INCOMPLETE`：原始模板已保留，不表示原始模板可以完成实验。

“检查通过”不等于证明全局最优或保证拿满分。GD 很低仍可能覆盖不全；100%可行率也不等于收敛。参数比较需要共同尺度，并同时关注计算成本。

## 读代码和对比阶段的方法

每份阶段代码以 `# %% 阶段 XX` 分段，可在 VS Code 中折叠阅读，也可搜索“阶段 03 新增”等标记。第2阶段以后是在前一阶段正文之后继续添加，公共实现保持一致。

第1阶段建议先找 `class NSGA2` 中的 `run()`，沿着调用顺序阅读：

```text
initialize → evaluate
  → fitness_assignment（排序和拥挤距离）
  → tournament_selection
  → crossover
  → mutation
  → evaluate
  → environmental_selection
  → 下一代
```

再读 `run_zdt3()`，理解“算法”和“实验入口”如何连接。公共文件输出、JSON 转换等工具可以最后看。第3阶段重点比较 `NSGA2._dominance_matrix` 与 `CNSGA2._dominance_matrix`；第4阶段先看 `RCM13.evaluate`、`constraint_values` 和 `repair`。

`changes/` 的差异文件可直接打开，也可以在编辑器中选两份 `.py` 文件进行比较。不要把累积版本误解成五个必须一起导入的模块。

## 参数实验和模型口径

第5阶段使用围绕同一基准的一次改变一个因素方法，共9个唯一配置：种群规模50/100/200，交叉概率0.6/0.9/1.0，变异概率0.5/7、1/7、2/7，代数200/500/1000。每个配置五个种子，45次运行。另有3种种群各5次等预算对照，每次50,000次评价（包括初始化）；最后用5个新种子分别验证基准和所选设置，共10次。因此总数为70。

所有配置共享从调参及对照运行中建立的固定归一化尺度与经验参考集，验证种子不参与选参或建立尺度。HV参考点固定为 `(1.1,1.1,1.1)`。这里的IGD参考是经验近似，不能称作已知真实齿轮箱前沿。

RCM13 默认按课程 PDF 的字面定义：齿数取 `{17,28}`，重量项包含 `14.9334/x3`。公开代码对公式、齿数范围或应力上限有不同定义，源码里保留了明确分开的模式。课程老师尚未确认哪种解释最终适用。改变模型定义后必须重新运行实验，不能混用不同模式的指标或旧报告。

这些 `.py` 文件是学习与验证版本。最终课程提交仍按要求使用 `.ipynb` 和2–3页PDF报告，不能直接把本文件夹当成提交ZIP。
