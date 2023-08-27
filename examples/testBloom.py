from smoothquant.bloom import Int8BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
import torch
from datasets import load_dataset

model = BloomForCausalLM.from_pretrained('bigscience/bloom-3b', torch_dtype=torch.float16)
#model = Int8BloomForCausalLM.from_pretrained("./int8_models/bloom-3b-smoothquant.pt", torch_dtype=torch.float16)
#model = Int8BloomForCausalLM.from_pretrained("./int8_models/bloom-3b-smoothquant.pt", ignore_mismatched_sizes=True)
model.to('cuda')
model.eval()
af = 0
for i in range(100000000):
    af += 1

data = load_dataset("json", data_files='alpaca_data_cleaned.json')
val_data = data["train"]

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


#if torch.__version__ >= "2":
#    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        max_new_tokens=512, repetition_penalty=3.5, temperature=0.50, top_k=50, top_p=1,
        #temperature=temperature,
        #top_p=top_p,
        #top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            #max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()
