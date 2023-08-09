# Tools README

## Prompt Viewer

> \[!warning\] 有Bug
> 支持的retriever和inferencer种类很少

本工具允许你在不启动完整训练流程的情况下，直接查看生成的 prompt。如果传入的配置仅为数据集配置（如 configs/datasets/nq/nq_gen_3dcea1.py），则展示数据集配置中定义的原始 prompt。若为完整的评测配置（包含模型和数据集），则会展示所选模型运行时实际接收到的 prompt。

```bash
python tools/prompt_viewer.py CONFIG_PATH [-n] [-a] [-p PATTERN]
```

- `-n`：不进入交互模式，默认选择第一个model和dataset
- `-a`：查看所有模型和数据集组合所接收到的prompt
- `-p PATTERN`：不进入交互模式，选择所有与正则表达式PATTERN匹配的数据集

example:

```bash
python tools/prompt_viewer.py configs/eval_opsqa.py

Number of tokens:  91
----------------------------------------------------------------------------------------------------
Sample prompt:
----------------------------------------------------------------------------------------------------

这里有一个关于运维的选择题，用A, B, C或D来回答。
Q: 什么是CI/CD的主要优点？
A. 提高安全性
B. 减少硬件成本
C. 提高开发速度和效率
D. 增加系统复杂性
答案:
----------------------------------------------------------------------------------------------------
```

## Case Analyzer

> \[!warning\] 也tm有bug

本工具在已有评测结果的基础上，产出推理错误样本以及带有标注信息的全量样本。

运行方式：

```
python tools/case_analyzer.py CONFIG_PATH [-w WORK_DIR]
```

- `-w`：工作路径，默认为 './outputs/default'。

## API Model Tester

本工具可以快速测试 API 模型的功能是否正常。

运行方式：

```
python tools/test_api_model.py [CONFIG_PATH] -n
```

## Prediction Merger

本工具可以合并由于 partitioner 而产生的分片推理结果。

运行方式：

```
python tools/prediction_merger.py CONFIG_PATH [-w WORK_DIR]
```

- `-w`：工作路径，默认为 './outputs/default'。
