import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear



def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model



class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc


from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-13b')
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer, 'cuda')


model_fp16 = OPTForCausalLM.from_pretrained('facebook/opt-13b', torch_dtype=torch.float16, device_map='auto')
acc_fp16 = evaluator.evaluate(model_fp16)
print(f'Original model (fp16) accuracy: {acc_fp16}')


model_w8a8 = quantize_model(model_fp16)
print(model_w8a8)
acc_w8a8 = evaluator.evaluate(model_w8a8)
print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')



model = OPTForCausalLM.from_pretrained('facebook/opt-13b', torch_dtype=torch.float16, device_map='auto')
act_scales = torch.load('../act_scales/opt-13b.pt')
smooth_lm(model, act_scales, 0.5)
model_smoothquant_w8a8 = quantize_model(model)
print(model_smoothquant_w8a8)


acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')

