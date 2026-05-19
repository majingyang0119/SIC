# 小脑神经元仿真实验数据

本文件夹包含三组突触整合实验的仿真数据，涵盖七种小脑神经元类型。

## 目录结构

```
EE/          兴奋性 + 兴奋性 突触实验
EI/          兴奋性 + 抑制性 突触实验
II/          抑制性 + 抑制性 突触实验

morphology_complexity.json   各细胞类型的树突复杂度与通道复杂度数据
```

每组实验目录（`EE/`、`EI/`、`II/`）按细胞类型组织：

```
EE/            EI/             II/
├── BC/        ├── BC/         ├── BC/
├── DCN/       ├── DCN/        ├── DCN/
├── GoC/       ├── GoC/        ├── GoC/
├── GrC/       ├── GrC/        ├── GrC/
├── IO/        ├── IO/         ├── IO/
├── PC/        ├── PC/         ├── PC/
└── SC/        └── SC/         └── SC/
```

## 各实验组数据量说明

### EE（兴奋性 + 兴奋性）

- 各细胞类型默认 **50 组**数据（GoC 和 GrC 除外）
- **GoC**：仅 **32 组**（预筛时剔除了无效位点）

### EI（兴奋性 + 抑制性）和 II（抑制性 + 抑制性）

- 各细胞类型默认 **100 组**数据（GoC 和 GrC 除外）
- **GoC**：数据量超过 100 组。因无法提前预知无效位点，实验时多跑了一些组，事后根据预实验结果筛选。有效位点记录在 `selected_pairs.json` 中

## 文件格式说明

每个细胞目录下包含以下文件：

| 文件/目录                                             | 说明                                                                                                                          |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `data/`                                           | 原始仿真输出（CSV 格式）。列名：`exc_w`, `inh_w`, `kappa`, `time`, `non_stim_value`, `e_value`, `i_value`, `ei_value`, `lin_sum_value`    |
| `{细胞}_{实验}_dur{时间窗}ms_tau{时间常数}ms.json`           | 各刺激位点的VE，按拟合阶数分别记录。`dur` = 观察时间窗口（毫秒），`tau` = 时间常数（毫秒）。（其中EI文件夹中使用tau4&5的含义是兴奋性tau=4ms、抑制性tau=5ms，并不是像之前一样分别把tau设成4和5之后取平均） |
| `{细胞}_{实验}_dur{时间窗}ms_tau{时间常数}ms_top_50pct.json` | 同上，但只保留 VE 前 50% 的位点                                                                                                        |
| `selected_pairs.json`（仅 GoC）                      | 预筛选后的有效刺激位点索引                                                                                                               |

### 命名示例

- `bc_ee_dur25_tau4.json` — BC 细胞在 EE 实验中，观察窗口 25 ms，τ = 4 ms
- `pc_ei_dur50_tau4&5.json` — PC 细胞在 EI 实验中（同时涉及兴奋性和抑制性突触），观察窗口 50 ms，兴奋性 τ = 4 ms，抑制性 τ = 5 ms

### VE JSON 数据格式（数据太多，我就没有把二次和四次全都改成Quadratic和Quartic）

```json
{
    "position_1": {
        "Linear": 0.9776,
        "Bilinear": 0.9962,
        "Cubic": 0.9996,
        "4th": 0.99997
    },
    ...
}
```

## 细胞类型对照

| 缩写  | 全称                              |
| --- | ------------------------------- |
| BC  | Basket cell（篮状细胞）               |
| DCN | Deep cerebellar nucleus（小脑深部核团） |
| GoC | Golgi cell（高尔基细胞）               |
| GrC | Granule cell（颗粒细胞）              |
| IO  | Inferior olive（下橄榄核）            |
| PC  | Purkinje cell（浦肯野细胞）            |
| SC  | Stellate cell（星形细胞）             |

## 形态复杂度与通道复杂度

`morphology_complexity.json` 记录了各细胞类型的结构和生物物理指标：

- **形态指标**：胞体/树突节段数、分支点数、树突末梢数、树突长度、分支级数
- **通道指标**：离子通道类型及其在各节段的分布
- **复杂度评分**：`Structure Complexity`（结构复杂度）和 `Channel Complexity`（通道复杂度），用于论文 Fig.5

## 论文图表数据来源

| 图表    | 数据来源                                                         |
| ----- | ------------------------------------------------------------ |
| Fig.1 | `EE/PC/data/`                                                |
| Fig.2 | `EE/PC/extra_PC_experiments.zip` |
| Fig.3 | `EE/PC/data/`                                                |
| Fig.4 | `EE/**/data/`（各细胞类型）                                         |
| Fig.5 | `EE/**/data/`（各细胞类型）+ `morphology_complexity.json`           |
| Fig.6 | `EI/**/data/` + `II/**/data/`（各细胞类型）                         |

## 补充说明

- `EE/PC/extra_PC_experiments.zip` 为浦肯野细胞的补充实验数据，用于 Fig.2
- GrC 的数据量与其他细胞类型不同，具体数量以各细胞文件夹中的实际文件数为准
