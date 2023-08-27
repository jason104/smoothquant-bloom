from datasets import load_dataset
from transformers import BloomForCausalLM, BloomTokenizerFast, GenerationConfig
import time
import torch
from smoothquant.bloom import Int8BloomForCausalLM


CUTOFF_LEN = 256
device = 'cuda'

#model = BloomForCausalLM.from_pretrained('bigscience/bloom-560m', torch_dtype=torch.float16)
model = Int8BloomForCausalLM.from_pretrained("./int8_models/bloom-3b-smoothquant.pt", torch_dtype=torch.float16)
tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom')
#print(tokenizer.pad_token_id)
#tokenizer.pad_token_id = 0
model.to(device)
model.eval()

#def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        #result = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN + 1, padding="max_length",)
        #return {"input_ids": result["input_ids"][:-1], "attention_mask": result["attention_mask"][:-1],}


def generate_and_tokenize_prompt(data_point):
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        user_prompt = (
                        (
                            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                            ### Instruction:
                            {data_point["instruction"]}

                            ### Input:
                            {data_point["input"]}

                            ### Response:
                            """
                        )
                    if data_point["input"]
                    else (
                            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                            ### Instruction:
                            {data_point["instruction"]}

                            ### Response:
                            """
                        )
                    )

        return {'prompt': user_prompt}


def evaluate(data_point, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128, **kwargs,):
    #inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = data_point["input_ids"].to(device)
    generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs,)
    with torch.no_grad():
        generation_output = model.generate(input_ids=input_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=max_new_tokens,)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()


data = load_dataset("json", data_files='alpaca_data_cleaned.json')

train_data = data["train"].map(generate_and_tokenize_prompt)
print(train_data)
start_time = time.time()
total_tokens = 0

for i, data in enumerate(train_data):
    if i > 10:
        break
    #tokens = tokenizer(data['prompt'], truncation=True, max_length=CUTOFF_LEN + 1, padding="max_length", return_tensors='pt')
    tokens = tokenizer(data['prompt'], return_tensors='pt')
    output = evaluate(tokens)
    total_tokens += len(output)
time_span = time.time() - start_time
print('total generated tokens:', total_tokens, '\ntotal time span:', time_span)


