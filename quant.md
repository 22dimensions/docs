
原始模型 
---> 模型量化工具（modelslim/gptqmodel/llmcompressor/gguf/autoawq）
---> 量化后的模型 
---> vLLM加载运行
（Qwen3常用格式 GPTQ gguf AWQ MLX）

vLLM 加载GPTQ算法量化后的模型
```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

# Create an LLM.
llm = LLM(model="/root/.cache/modelscope/hub/models/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
print("-"*50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-"*50)
```


量化的运行流程 
- debug代码演示
- 加载量化配置 **普通层替换为量化层**
- 非量化layer ---> 为量化计算layer
- init 的时候 create_weights  forward调用apply

新增量化算法
- 继承 QuantizationConfig
- 继承 QuantizeMethodBase

涉及的量化layer
- BaseKVCacheMethod
- FusedMoEMethodBase
- UnquantizedEmbeddingMethod
- LinearMethodBase

vLLM-Ascend量化流程
- 注册了AscendQuantConfig
