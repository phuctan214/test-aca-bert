

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import MSELoss
from torch.nn.parameter import Parameter
from torch.utils import checkpoint
from transformers import BertConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from gpytorch.kernels.kernel import Distance    

logger = logging.getLogger(__name__)

from pretraining.modeling import f_gelu, bias_gelu, bias_relu, bias_tanh, gelu, swish, get_deepspeed_config

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish, "tanh": F.tanh}
from pretraining.modeling import LinearActivation, RegularLinearActivation, get_apex_layer_norm, RMSNorm

LAYER_NORM_TYPES = {"pytorch": nn.LayerNorm, "apex": get_apex_layer_norm(), "rms_norm": RMSNorm}


from pretraining.modeling import BertEmbeddings, get_layer_norm_type

######### bert_mgk8_scale3_pi_square_final
class MGKBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(MGKBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)//2
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dist = Distance()
        self.pi = nn.Parameter(torch.ones(2, self.num_attention_heads)/2., requires_grad = True)
        self.scaling = self.attention_head_size**-0.5
        # self.softmax = nn.Softmax(dim=-1)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer1 = self.key1(hidden_states)
        mixed_key_layer2 = self.key2(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        QK1_distance = (-self.scaling/2)*self.dist._sq_dist(query_layer, key_layer1, postprocess = False) + attention_mask
        QK2_distance = (-1.5*self.scaling/2)*self.dist._sq_dist(query_layer, key_layer2, postprocess = False) + attention_mask
        QK1_distance = QK1_distance - QK1_distance.max(dim = -1, keepdim = True)[0]
        QK2_distance = QK2_distance - QK2_distance.max(dim = -1, keepdim = True)[0]

        
        # assert 1==2
        attention_scores = torch.exp(QK1_distance)*torch.clamp(self.pi[0][None, :, None, None], min = 1e-6, max = 2.) + torch.exp(QK2_distance)*torch.clamp(self.pi[1][None, :, None, None], min = 1e-6, max = 2.)
        # Normalize the attention scores to probabilities.
        attention_probs = attention_scores/(attention_scores.sum(dim = -1, keepdim = True) + 1e-6)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs



############# bert_mgk8_scale1_5_final
# class MGKBertSelfAttention(nn.Module):
#     def __init__(self, config):
#         super(MGKBertSelfAttention, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads)
#             )
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)//2
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key1 = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.dist = Distance()
#         self.pi = nn.Parameter(torch.ones(2, self.num_attention_heads)/2., requires_grad = True)
#         self.scaling = self.attention_head_size**-0.5
#         # self.softmax = nn.Softmax(dim=-1)


#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def transpose_key_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 3, 1)

#     def forward(self, hidden_states, attention_mask):
#         # print(hidden_states.shape, 'hidden_states')
#         # print(attention_mask.shape, 'attention_mask')

#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer1 = self.key1(hidden_states)
#         mixed_key_layer2 = self.key2(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer1 = self.transpose_for_scores(mixed_key_layer1)
#         key_layer2 = self.transpose_for_scores(mixed_key_layer2)
#         # key_layer1 = self.transpose_key_for_scores(mixed_key_layer1)
#         # key_layer2 = self.transpose_key_for_scores(mixed_key_layer2)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#         # print(query_layer.shape, 'query_layer')
#         # print(key_layer1.shape, 'key_layer1')


#         QK1_distance = (-self.scaling/2)*self.dist._sq_dist(query_layer, key_layer1, postprocess = False) + attention_mask
#         QK2_distance = (-1.5*self.scaling/2)*self.dist._sq_dist(query_layer, key_layer2, postprocess = False) + attention_mask
#         QK1_distance = QK1_distance - QK1_distance.max(dim = -1, keepdim = True)[0]
#         QK2_distance = QK2_distance - QK2_distance.max(dim = -1, keepdim = True)[0]
#         # QK1_distance = torch.matmul(query_layer, key_layer1) / math.sqrt(self.attention_head_size) + attention_mask
#         # QK2_distance = torch.matmul(query_layer, key_layer2) / math.sqrt(self.attention_head_size) + attention_mask
#         # print(QK1_distance.shape, 'QK1_distance_shape')
#         # print(QK1_distance, 'QK1_distance')
#         # Take the dot product between "query" and "key" to get the raw attention scores.
        
#         # assert 1==2
#         attention_scores = torch.exp(QK1_distance)*torch.clamp(self.pi[0][None, :, None, None], min = 1e-6, max = 2.) + torch.exp(QK2_distance)*torch.clamp(self.pi[1][None, :, None, None], min = 1e-6, max = 2.)
#         # Normalize the attention scores to probabilities.
#         attention_probs = attention_scores/(attention_scores.sum(dim = -1, keepdim = True) + 1e-6)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         return context_layer, attention_probs

class MGKBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(MGKBertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size//2, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MGKBertAttention(nn.Module):
    def __init__(self, config):
        super(MGKBertAttention, self).__init__()
        self.self = MGKBertSelfAttention(config)
        self.output = MGKBertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):

        context_layer, attention_probs = self.self(input_tensor, attention_mask)
        # print(context_layer.shape, attention_probs.shape)
        # assert 1==2
        attention_output = self.output(context_layer, input_tensor)
        output = (
            attention_output,
            attention_probs,
        )
        return output


from pretraining.modeling import BertIntermediate, BertOutput
class MGKBertLayer(nn.Module):
    def __init__(self, config):
        super(MGKBertLayer, self).__init__()
        self.attention = MGKBertAttention(config)
        self.config = config

        BertLayerNorm = get_layer_norm_type(config)

        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def maybe_layer_norm(self, hidden_states, layer_norm, current_ln_mode):
        if self.config.useLN and self.config.encoder_ln_mode in current_ln_mode:
            return layer_norm(hidden_states)
        else:
            return hidden_states

    def forward(self, hidden_states, attention_mask, action=1, keep_prob=1.0):
        attention_probs = None
        intermediate_input = None

        if action == 0:
            intermediate_input = hidden_states
        else:
            pre_attn_input = self.maybe_layer_norm(
                hidden_states, self.PreAttentionLayerNorm, "pre-ln"
            )
            self_attn_out = self.attention(pre_attn_input, attention_mask)

            attention_output, attention_probs = self_attn_out
            attention_output = attention_output * 1 / keep_prob

            intermediate_input = hidden_states + attention_output
            intermediate_input = self.maybe_layer_norm(
                intermediate_input, self.PreAttentionLayerNorm, "post-ln"
            )

        if action == 0:
            layer_output = intermediate_input
        else:
            intermediate_pre_ffn = self.maybe_layer_norm(
                intermediate_input, self.PostAttentionLayerNorm, "pre-ln"
            )
            intermediate_output = self.intermediate(intermediate_pre_ffn)

            layer_output = self.output(intermediate_output)
            layer_output = layer_output * 1 / keep_prob

            layer_output = layer_output + intermediate_input
            layer_output = self.maybe_layer_norm(
                layer_output, self.PostAttentionLayerNorm, "post-ln"
            )

        output = (
            layer_output,
            attention_probs,
        )
        return output


class MGKBertEncoder(nn.Module):
    def __init__(self, config, args):
        super(MGKBertEncoder, self).__init__()
        self.config = config
        BertLayerNorm = get_layer_norm_type(config)
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.is_transformer_kernel = (
            hasattr(args, "deepspeed_transformer_kernel") and args.deepspeed_transformer_kernel
        )

        if hasattr(args, "deepspeed_transformer_kernel") and args.deepspeed_transformer_kernel:
            from deepspeed import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer

            ds_config = get_deepspeed_config(args)
            has_huggingface = hasattr(args, "huggingface")
            ds_transformer_config = DeepSpeedTransformerConfig(
                batch_size=ds_config.train_micro_batch_size_per_gpu,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                local_rank=args.local_rank if hasattr(args, "local_rank") else -1,
                seed=args.seed,
                fp16=ds_config.fp16_enabled,
                pre_layer_norm=True if "pre-ln" in config.encoder_ln_mode else False,
                normalize_invertible=args.normalize_invertible,
                gelu_checkpoint=args.gelu_checkpoint,
                adjust_init_range=True,
                attn_dropout_checkpoint=args.attention_dropout_checkpoint,
                stochastic_mode=args.stochastic_mode,
                huggingface=has_huggingface,
                training=self.training,
            )

            self.layer = nn.ModuleList(
                [
                    copy.deepcopy(DeepSpeedTransformerLayer(ds_transformer_config))
                    for _ in range(config.num_hidden_layers)
                ]
            )
        else:
            layer = MGKBertLayer(config)
            self.layer = nn.ModuleList(
                [copy.deepcopy(layer) for _ in range(self.config.num_hidden_layers)]
            )

    def add_attention(self, all_attentions, attention_probs):
        if attention_probs is not None:
            all_attentions.append(attention_probs)

        return all_attentions

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        all_encoder_layers = []
        all_attentions = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(
                    custom(l, l + chunk_length), hidden_states, attention_mask * 1
                )
                l += chunk_length
            # decoder layers
        else:
            for layer_module in self.layer:
                if self.is_transformer_kernel:
                    # using Deepspeed Transformer kernel
                    hidden_states = layer_module(hidden_states, attention_mask)
                else:
                    layer_out = layer_module(
                        hidden_states,
                        attention_mask,
                    )
                    hidden_states, attention_probs = layer_out
                    # get all attention_probs from layers
                    if output_attentions:
                        all_attentions = self.add_attention(all_attentions, attention_probs)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if self.config.useLN and self.config.encoder_ln_mode in "pre-ln":
                hidden_states = self.FinalLayerNorm(hidden_states)

            all_encoder_layers.append(hidden_states)
        outputs = (all_encoder_layers,)
        if output_attentions:
            outputs += (all_attentions,)
        return outputs


from pretraining.modeling import BertLMPredictionHead, BertOnlyMLMHead, BertPredictionHeadTransform, BertPooler, BertPreTrainedModel

class MGKBertModel(BertPreTrainedModel):

    def __init__(self, config, args=None):
        super(MGKBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = MGKBertEncoder(config, args)
        self.pooler = BertPooler(config)

        logger.info("Init BERT pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)


        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
            output_attentions=output_attentions,
        )
        encoded_layers = encoder_output[0]
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        output = (
            encoded_layers,
            pooled_output,
        )
        if output_attentions:
            output += (encoder_output[-1],)
        return output

from pretraining.modeling import BertPreTrainingHeads
class MGKBertForPreTraining(BertPreTrainedModel):

    def __init__(self, config, args):
        super(MGKBertForPreTraining, self).__init__(config)
        self.bert = MGKBertModel(config, args)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False

        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        if masked_lm_labels is not None and next_sentence_label is not None:
            # filter out all masked labels.
            masked_token_indexes = torch.nonzero(
                (masked_lm_labels + 1).view(-1), as_tuple=False
            ).view(-1)
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_token_indexes
            )
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
            return prediction_scores, seq_relationship_score


class MGKBertLMHeadModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super(MGKBertLMHeadModel, self).__init__(config)
        self.bert = MGKBertModel(config, args)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, batch, output_attentions=False):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[4]
        checkpoint_activations = False

        bert_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = bert_output[0]

        if masked_lm_labels is None:
            prediction_scores = self.cls(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.cls(sequence_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            if output_attentions:
                outputs += (bert_output[-1],)
            return outputs
        else:
            return prediction_scores

from pretraining.modeling import BertOnlyNSPHead
class MGKBertForNextSentencePrediction(BertPreTrainedModel):

    def __init__(self, config, args):
        super(MGKBertForNextSentencePrediction, self).__init__(config)
        self.bert = MGKBertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        next_sentence_label=None,
        checkpoint_activations=False,
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            return next_sentence_loss
        else:
            return seq_relationship_score


class MGKBertForSequenceClassification(BertPreTrainedModel):


    def __init__(self, config, args=None):
        super(MGKBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = MGKBertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
