falcon-7b-hf: einops

bleu: sacrebleu

rouge: rouge_score

qwen: transformers_stream_generator

llama:

```
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -e .
```

qwen需要flash-attention来进行加速，但是flash-attention需要安装，我目前懒得装
