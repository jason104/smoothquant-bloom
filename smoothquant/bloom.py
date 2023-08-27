import torch
import math

from torch import nn
from torch.nn import functional as F

from transformers.models.bloom.modeling_bloom import (
    BloomConfig,
    BloomForCausalLM,
    BloomModel,
    BloomPreTrainedModel,
    BloomAttention,
    BloomMLP,
    BloomGelu,
    BloomBlock
)

from typing import Optional, Tuple, List, Union
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearGELU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)


class Int8BloomAttention(nn.Module):

    _split_heads = BloomAttention._split_heads
    _merge_heads = BloomAttention._merge_heads

    def __init__(self, config: BloomConfig):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.query_key_value = W8A8B8O8Linear(
            self.hidden_size, 3 * self.hidden_size)
        self.dense = W8A8BFP32OFP32Linear(self.hidden_size, self.hidden_size)

    @staticmethod
    @torch.no_grad()
    def from_float(module: BloomAttention,
                   input_scale: float,
                   qkv_output_scale: float,
                   dense_input_scale: float):
        config = BloomConfig(hidden_size=module.hidden_size, n_head=module.num_heads, pretraining_tp=module.pretraining_tp, slow_but_exact=module.slow_but_exact)
        int8_module = BloomAttention(config)
        #int8_module = BloomAttention(module.embed_dim, module.num_heads)

        int8_module.query_key_value = W8A8B8O8Linear.from_float(
            module.query_key_value, input_scale, qkv_output_scale)
        int8_module.dense = W8A8BFP32OFP32Linear.from_float(
            module.dense, dense_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            qkv_output_scale, qkv_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, qkv_output_scale, dense_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = self.query_key_value(hidden_states)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        # matmul_result = alibi.baddbmm(
        #     batch1=query_layer,
        #     batch2=key_layer,
        #     beta=self.beta,
        #     alpha=self.inv_norm_factor,
        # )
        matmul_result = alibi * self.beta + self.qk_bmm(
            query_layer, key_layer) * self.inv_norm_factor

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(
            batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(
            attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        output_tensor = output_tensor + residual

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class Int8BloomMLP(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = W8A8B8O8LinearGELU(hidden_size, hidden_size * 4)
        self.dense_4h_to_h = W8A8BFP32OFP32Linear(hidden_size * 4, hidden_size)
        #self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        #self.gelu_impl = BloomGelu()
        #self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        #hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))
        hidden_states = self.dense_h_to_4h(hidden_states)

        intermediate_output = self.dense_4h_to_h(hidden_states)

        output = intermediate_output + residual

        return output

    @staticmethod
    def from_float(module: BloomMLP, fc1_input_scale: float, fc2_input_scale: float):
       config = BloomConfig(hidden_size=module.dense_h_to_4h.in_features, pretraining_tp=module.pretraining_tp, slow_but_exact=module.slow_but_exact)
       int8_module = Int8BloomMLP(config)
       #int8_module.dense_h_to_4h = W8A8BFP32OFP32Linear.from_float(module.dense_h_to_4h, fc1_input_scale)
       #gelu_dim = module.dense_h_to_4h.out_features
       int8_module.dense_h_to_4h = W8A8B8O8LinearGELU.from_float(module.dense_h_to_4h, fc1_input_scale, fc2_input_scale)
       int8_module.dense_4h_to_h = W8A8BFP32OFP32Linear.from_float(module.dense_4h_to_h, fc2_input_scale)
       return int8_module


class Int8BloomBlock(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNormQ(
            hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = Int8BloomAttention(config)
        self.post_attention_layernorm = LayerNormQ(
            hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = Int8BloomMLP(config)

        if config.apply_residual_connection_post_layernorm:
            raise NotImplementedError('Not implemented yet.')

    @staticmethod
    def from_float(module: BloomBlock,
                   attn_input_scale: float,
                   #q_output_scale: float,
                   #k_output_scale: float,
                   #v_output_scale: float,
                   qkv_output_scale: float,
                   out_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        config = BloomConfig(hidden_size=module.self_attention.hidden_size, layer_norm_epsilon=module.input_layernorm.eps, n_head=module.num_heads, pretraining_tp=module.mlp.pretraining_tp, slow_but_exact=module.mlp.slow_but_exact)
        int8_module = Int8BloomBlock(
            config
        )
        int8_module.input_layernorm = LayerNormQ.from_float(
            module.input_layernorm, attn_input_scale)
        int8_module.self_attention = Int8BloomAttention.from_float(
            module.self_attention, attn_input_scale, qkv_output_scale, out_input_scale)
        int8_module.post_attention_layernorm = LayerNormQ.from_float(
            module.post_attention_layernorm, fc1_input_scale)
        int8_module.mlp = Int8BloomMLP.from_float(
            module.mlp, fc1_input_scale, fc2_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class Int8BloomModel(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNormQ(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([Int8BloomBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNormQ(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = BloomModel.get_input_embeddings
    set_input_embeddings = BloomModel.set_input_embeddings
    _prepare_attn_mask = BloomModel._prepare_attn_mask
    build_alibi_tensor = BloomModel.build_alibi_tensor
    old_forward = BloomModel.forward


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Tuple[torch.Tensor, ...]:
    #) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:

        input_len = input_ids.shape[1]
        from torch.nn.functional import pad
        if input_len % 16 != 0:
            # <pad> is 1
            padding_len = 16 - input_len % 16
            input_ids = pad(input_ids, (0, padding_len), value=1)
            if attention_mask is not None:
                attention_mask = pad(attention_mask, (0, padding_len), value=0)

        output = self.old_forward(input_ids, past_key_values, attention_mask, head_mask, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict, **deprecated_arguments)

        if input_len % 16 != 0:
            output.last_hidden_state = output.last_hidden_state[:, :input_len, :]
        return output


    #@add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    #@add_code_sample_docstrings(
    #    checkpoint=_CHECKPOINT_FOR_DOC,
    #    output_type=BaseModelOutputWithPastAndCrossAttentions,
    #    config_class=_CONFIG_FOR_DOC,
    #)

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8BloomModel(module.config)

        int8_module.word_embeddings = module.word_embeddings
        int8_module.word_embeddings_layernorm = module.word_embeddings_layernorm
        int8_module.ln_f = module.ln_f

        for i, layer in enumerate(module.h):
            int8_module.h[i] = Int8BloomBlock.from_float(layer, **decoder_layer_scales[i])        
        return int8_module



class Int8BloomForCausalLM(BloomPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = Int8BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    #@add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    #@add_code_sample_docstrings(
    #    checkpoint=_CHECKPOINT_FOR_DOC,
    #    output_type=CausalLMOutputWithCrossAttentions,
    #    config_class=_CONFIG_FOR_DOC,
    #)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Tuple[torch.Tensor]:
    #) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8BloomForCausalLM(module.config)
        int8_module.transformer = Int8BloomModel.from_float(
            module.transformer, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module

    _reorder_cache = BloomForCausalLM._reorder_cache
    get_output_embeddings = BloomForCausalLM.get_output_embeddings
    set_output_embeddings = BloomForCausalLM.set_output_embeddings
    prepare_inputs_for_generation = BloomForCausalLM.prepare_inputs_for_generation

