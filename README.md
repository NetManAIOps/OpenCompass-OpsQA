[OpenCompass中文说明文档](https://opencompass.readthedocs.io/zh_CN)

更多踩坑记录或者使用笔记，欢迎添加至netman-doc文件夹

# 环境配置

```bash
cd /mnt/mfs/opsgpt/opencompass
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
pip install -e .
```

# 运行评测

```bash
python run.py configs/eval_demo.py -w outputs/test [--debug]

-w: workdir，输出文件夹
--debug: 关闭任务并行，更多调试输出
```

# 主要目录说明

configs: 配置文件夹，这里面的配置脚本不包含类的定义，而是导入对应的类并制定类的参数、评估方式等
datasets: 数据集配置文件夹
models: 模型配置文件夹
local_models: 我新建的，专门放/mnt/mfs/opsgpt/models里的模型的配置文件
eval_xxx.py: 运行评测任务的文件，参考eval_oreilly.py, eval_opsqa.py等
opencompass: 本框架的主要源码文件夹
datasets: 数据集的类定义
models: 定义了几个基础模型类包括huggingface, openai_api等，我加了个chatanywhere_api(peiqi)
openicl:
icl_evaluator: icl_hf_evaluator.py中定义了几个我们最常用的Evaluator（Acc, BLEU, Rouge等，后两个都依赖hf的库，需要安装依赖）
icl_inferencer: 包括gen, ppl, sc等，涉及模型推理答案的方式
icl_retriever: ZeroRetriever以及其他few-shot需要用到的retrieve方式

其他地方我也不太熟悉

# 关于添加数据集的说明

在opencompass框架中添加数据集需要如下几个步骤：(以opencompass项目目录为根目录）

1. 在`opencompass/datasets`下添加py脚本描述数据集类，主要完成加载raw数据并返回huggingface的dataset类的工作

   参考`opencompass/datasets/opsqa.py`、`opencompass/datasets/mmlu.py`

   然后在`opencompass/datasets/__init__.py`中import对应类，方便后续导入

2. 在`configs/datasets`中添加对应数据集配置文件

   参考`configs/datasets/OpsQA/opsqa_gen.py`，以及文档：[数据集配置文件示例](https://opencompass.readthedocs.io/zh_CN)

   在这个配置文件中需要配置三个cfg：

   1. `reader_cfg`：读取配置，包括`input_columns`，`output_column`，`test_split`/`train_split`，指导如何从dataset中提取输入列和输出列

   2. `infer_cfg`：推理配置，配置Prompt生成`prompt_template`，上下文样本`retriever`，推理方式`inferencer`

   3. `eval_cfg`：评估配置，指定使用什么作为评估指标`evaluator`

## 踩坑记录

- API模型用不了PPLInferencer，因为获得不了模型推理的loss
