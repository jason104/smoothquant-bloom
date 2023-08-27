import gc
import os
import sys
import threading

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
    Trainer, TrainingArguments,
    BloomForCausalLM, BloomTokenizerFast,
)

from peft import LoraConfig, TaskType, get_peft_model
import transformers



def main():
    accelerator = Accelerator()
    model_name = "bigscience/bloom-7b1"
    MICRO_BATCH_SIZE = 1  # this could actually be 5 but i like powers of 2
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 3  # we don't always need 3 tbh
    LEARNING_RATE = 3e-4  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_SET_SIZE = 2000
    DATA_PATH = "alpaca_data_cleaned.json"
    do_test = False
    OUTPUT_DIR = 'bloom-7b1-alpaca-full'
    seed = 42
    set_seed(seed)
    accelerator.wait_for_everyone()

    model = BloomForCausalLM.from_pretrained( 
        model_name,
        torch_dtype=torch.float16,
        #device_map='auto',
        #load_in_8bit=True,
    )
    tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom')

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=DATA_PATH)

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
        len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=CUTOFF_LEN + 1,
                    padding="max_length",
                )["input_ids"]
            )
            - 1
        )  # no eos token
        full_tokens = tokenizer(
            user_prompt + data_point["output"],
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }


    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if VAL_SET_SIZE > 0 else None,
            save_steps=200,
            output_dir=OUTPUT_DIR, #output_dir=repository_id,
            save_total_limit=1,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            #ddp_find_unused_parameters=False if ddp else None,
            torch_compile=True, # optimizations
            #optim="adamw_torch_fused", # improved optimizer
            optim="adamw_torch", # improved optimizer
            # push to hub parameters
            report_to='none',
            #push_to_hub=True,
            #hub_strategy="every_save",
            #hub_model_id=repository_id,
            #hub_token=HfFolder.get_token(),
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False


    #dataset = load_dataset("ought/raft", dataset_name)
    #classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    #dataset = dataset.map(
    #    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    #    batched=True,
    #    num_proc=1,
    #)

    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)



    # creating model
    #model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    #model = get_peft_model(model, peft_config)
    #model.print_trainable_parameters()

    # optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    #accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    #for epoch in range(num_epochs):
    #    with TorchTracemalloc() as tracemalloc:
    #        model.train()
    #        total_loss = 0
    #        for step, batch in enumerate(tqdm(train_dataloader)):
    #            outputs = model(**batch)
    #            loss = outputs.loss
    #            total_loss += loss.detach().float()
    #            accelerator.backward(loss)
    #            optimizer.step()
    #            lr_scheduler.step()
    #            optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    #    accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
    #    accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
    #    accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
    #    accelerator.print(
    #        "GPU Total Peak Memory consumed during the train (max): {}".format(
    #            tracemalloc.peaked + b2mb(tracemalloc.begin)
    #        )
    #    )

    #    accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
    #    accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
    #    accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
    #    accelerator.print(
    #        "CPU Total Peak Memory consumed during the train (max): {}".format(
    #            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
    #        )
    #    )
    #    train_epoch_loss = total_loss / len(train_dataloader)
    #    train_ppl = torch.exp(train_epoch_loss)
    #    accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    #    model.eval()
    #    eval_preds = []
    #    with TorchTracemalloc() as tracemalloc:
    #        for _, batch in enumerate(tqdm(eval_dataloader)):
    #            batch = {k: v for k, v in batch.items() if k != "labels"}
    #            with torch.no_grad():
    #                outputs = accelerator.unwrap_model(model).generate(
    #                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
    #                )  # synced_gpus=True for DS-stage 3
    #            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
    #            preds = accelerator.gather_for_metrics(outputs)
    #            preds = preds[:, max_length:].detach().cpu().numpy()
    #            eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    #    accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
    #    accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
    #    accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
    #    accelerator.print(
    #        "GPU Total Peak Memory consumed during the eval (max): {}".format(
    #            tracemalloc.peaked + b2mb(tracemalloc.begin)
    #        )
    #    )

    #    accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
    #    accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
    #    accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
    #    accelerator.print(
    #        "CPU Total Peak Memory consumed during the eval (max): {}".format(
    #            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
    #        )
    #    )

    #    correct = 0
    #    total = 0
    #    assert len(eval_preds) == len(
    #        dataset["train"][label_column]
    #    ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
    #    for pred, true in zip(eval_preds, dataset["train"][label_column]):
    #        if pred.strip() == true.strip():
    #            correct += 1
    #        total += 1
    #    accuracy = correct / total * 100
    #    accelerator.print(f"{accuracy=}")
    #    accelerator.print(f"{eval_preds[:10]=}")
    #    accelerator.print(f"{dataset['train'][label_column][:10]=}")

    if do_test:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        test_preds_cleaned = []
        for _, pred in enumerate(test_preds):
            test_preds_cleaned.append(get_closest_label(pred, classes))

        test_df = dataset["test"].to_pandas()
        assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
        test_df[label_column] = test_preds_cleaned
        test_df["text_labels_orig"] = test_preds
        accelerator.print(test_df[[text_column, label_column]].sample(20))

        pred_df = test_df[["ID", label_column]]
        pred_df.columns = ["ID", "Label"]

        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)

    trainer.train(resume_from_checkpoint = False)
    accelerator.wait_for_everyone()
    model.save_pretrained(OUTPUT_DIR)
    #torch.save(model.state_dict(), OUTPUT_DIR + '/model.pt')
    print(OUTPUT_DIR)
    #model.push_to_hub(
    #    "smangrul/"
    #    + f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
    #    state_dict=accelerator.get_state_dict(model),
    #    use_auth_token=True,
    #)
    #accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
