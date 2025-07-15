
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

import os 

os.environ["SAVE_PATH"] = "/root/dump_data/new_fa3"

llm = LLM(model="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct-fa3",
          max_model_len=2048,
          trust_remote_code=True,
          enforce_eager=True,
          quantization="ascend")
          # Enable quantization by specifing `quantization="ascend"`)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
