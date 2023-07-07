""" PyTorch ViT model."""


import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput

from transformers.utils import logging
from transformers import ViTConfig
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


logger = logging.get_logger(__name__)


class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_ones=True):
        super().__init__()
        if init_ones:
            self.alpha = nn.Parameter(torch.ones(hidden_size) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.move = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = x * self.alpha + self.move
        return out


class ViTEmbeddings(nn.Module):
    """
    Construct position and patch embeddings.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()

        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size[0]))
        trunc_normal_(self.position_embeddings, std=.02)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.norm = config.norm_layer(config.hidden_size[0], eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size[0]
        patch_size = 4

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        if config.hidden_size[config.stages[layer_num]] % config.num_attention_heads[config.stages[layer_num]] != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size[config.stages[layer_num]],} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads[config.stages[layer_num]]}."
            )

        self.num_attention_heads = config.num_attention_heads[config.stages[layer_num]]
        self.attention_head_size = int(config.hidden_size[config.stages[layer_num]] / config.num_attention_heads[config.stages[layer_num]])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.moveq = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
        self.movek = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
        self.movev = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))

        self.query = nn.Linear(config.hidden_size[config.stages[layer_num]], self.all_head_size)
        self.key = nn.Linear(config.hidden_size[config.stages[layer_num]], self.all_head_size)
        self.value = nn.Linear(config.hidden_size[config.stages[layer_num]], self.all_head_size)

        self.reduction_ratio = config.reduction_ratio[config.stages[layer_num]]
        if self.reduction_ratio > 1:
            self.pool = nn.AvgPool2d(config.reduction_ratio[config.stages[layer_num]], stride=config.reduction_ratio[config.stages[layer_num]])
            self.mover = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
            self.reduction = nn.Linear(config.hidden_size[config.stages[layer_num]], self.all_head_size)
            self.norm_r = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
            self.rprelur = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.normq = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.normk = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.normv = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.rpreluq = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.rpreluk = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.rpreluv = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.moveq2 = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
        self.movek2 = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
        self.movev2 = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))

        self.norm_context = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.rprelu_context = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        mixed_query_layer = self.normq(self.query(hidden_states + self.moveq)) + hidden_states

        if self.reduction_ratio > 1:
            B, N, C = hidden_states.shape
            image_height = int(math.sqrt(N))
            image_width = int(math.sqrt(N))
            hidden_states = hidden_states.permute(0, 2, 1).reshape(B, C, image_height, image_width)
            hidden_states = self.pool(hidden_states).reshape(B, C, -1).permute(0, 2, 1)
            hidden_states = self.norm_r(self.reduction(hidden_states + self.mover)) + hidden_states
            hidden_states = self.rprelur(hidden_states)

        mixed_key_layer = self.normk(self.key(hidden_states + self.movek)) + hidden_states
        mixed_value_layer = self.normv(self.value(hidden_states + self.movev)) + hidden_states

        mixed_query_layer = self.rpreluq(mixed_query_layer)
        mixed_key_layer = self.rpreluk(mixed_key_layer)
        mixed_value_layer = self.rpreluv(mixed_value_layer)

        query_layer = mixed_query_layer + self.moveq2
        key_layer = mixed_key_layer + self.movek2
        value_layer = mixed_value_layer + self.movev2

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.reduction_ratio > 1:
            mixed_key_layer = mixed_key_layer.permute(0, 2, 1).reshape(B, C, image_height // self.reduction_ratio, 
                                                                       image_width // self.reduction_ratio)
            mixed_key_layer = torch.nn.functional.interpolate(mixed_key_layer, size=image_height, mode='nearest').reshape(B, C, -1).permute(0, 2, 1)

            mixed_value_layer = mixed_value_layer.permute(0, 2, 1).reshape(B, C, image_height // self.reduction_ratio, 
                                                                           image_width // self.reduction_ratio)
            mixed_value_layer = torch.nn.functional.interpolate(mixed_value_layer, size=image_height, mode='nearest').reshape(B, C, -1).permute(0, 2, 1)

        context_layer = self.norm_context(context_layer) + mixed_query_layer + mixed_key_layer + mixed_value_layer
        context_layer = self.rprelu_context(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size[config.stages[layer_num]], config.hidden_size[config.stages[layer_num]])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.move = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
        self.norm = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.rprelu = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.layerscale = LayerScale(config.hidden_size[config.stages[layer_num]])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        out = self.norm(self.dense(hidden_states + self.move)) + hidden_states
        out = self.rprelu(out)
        out = self.dropout(out)

        out = self.layerscale(out)

        return out


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config, layer_num)
        self.output = ViTSelfOutput(config, layer_num)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size[config.stages[layer_num]], config.intermediate_size[config.stages[layer_num]])
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

        self.move = nn.Parameter(torch.zeros(config.hidden_size[config.stages[layer_num]]))
        self.norm = config.norm_layer(config.intermediate_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.rprelu = RPReLU(config.intermediate_size[config.stages[layer_num]])
        self.expansion_ratio = config.intermediate_size[config.stages[layer_num]] // config.hidden_size[config.stages[layer_num]]


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        out = self.norm(self.dense(hidden_states + self.move)) + torch.concat([hidden_states for _ in range(self.expansion_ratio)], dim=-1)
        out = self.rprelu(out)
        # out = self.intermediate_act_fn(out)

        return out


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig, layer_num, drop_path=0.0) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size[config.stages[layer_num]], config.hidden_size[config.stages[layer_num]])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.move = nn.Parameter(torch.zeros(config.intermediate_size[config.stages[layer_num]]))
        self.norm = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.rprelu = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.pooling = nn.AvgPool1d(config.intermediate_size[config.stages[layer_num]] // config.hidden_size[config.stages[layer_num]])
        self.layerscale = LayerScale(config.hidden_size[config.stages[layer_num]])


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.dense(hidden_states + self.move)) + self.pooling(hidden_states)
        out = self.rprelu(out)
        out = self.dropout(out)

        out = self.layerscale(out)

        out = self.drop_path(out)

        return out



class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, layer_num, drop_path=0.0) -> None:
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config, layer_num)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.intermediate = ViTIntermediate(config, layer_num)
        self.output = ViTOutput(config, layer_num, drop_path=drop_path)
        self.layernorm_before = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.layernorm_after = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.avg_res3 = config.avg_res3
        self.avg_res5 = config.avg_res5

        if self.avg_res5:
            print("Using Avg-Pooling 5 Residual")
            kernel_size = 5
            self.avg_res_w5 = nn.AvgPool2d((1, kernel_size), stride=1, padding=(0, int((kernel_size-1)/2)))
            self.layerscale_w5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.avg_res_h5 = nn.AvgPool2d((kernel_size, 1), stride=1, padding=(int((kernel_size-1)/2), 0))
            self.layerscale_h5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)

        if self.avg_res3:
            print("Using Avg-Pooling 3 Residual")
            kernel_size = 3
            self.avg_res_w3 = nn.AvgPool2d((1, kernel_size), stride=1, padding=(0, int((kernel_size-1)/2)))
            self.layerscale_w3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.avg_res_h3 = nn.AvgPool2d((kernel_size, 1), stride=1, padding=(int((kernel_size-1)/2), 0))
            self.layerscale_h3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        hidden_states_norm = self.layernorm_before(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states_norm,  # in ViT, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in ViT, layernorm is also applied after self-attention
        hidden_states_norm = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(hidden_states_norm)

        # second residual connection is done here
        layer_output = self.output(layer_output) + hidden_states
        B, N, C = hidden_states_norm.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))
        if self.avg_res3:
            layer_output += self.layerscale_h3(self.avg_res_h3(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous())
            layer_output += self.layerscale_w3(self.avg_res_w3(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous())
        if self.avg_res5:
            layer_output += self.layerscale_h5(self.avg_res_h5(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous())
            layer_output += self.layerscale_w5(self.avg_res_w5(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous())

        outputs = (layer_output,) + outputs

        return outputs



class BinaryPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=2, in_dim=3, out_dim=64, config=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0], img_size[1]
        self.num_patches = self.H * self.W

        self.norm0 = config.norm_layer(in_dim)
        
        self.move = nn.Parameter(torch.zeros(1, in_dim, 1, 1))
        self.proj = nn.Conv2d(in_dim, out_dim, self.patch_size, self.patch_size, bias=False)
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.norm = config.norm_layer(out_dim)
        self.rprelu = RPReLU(out_dim)

        self.position_embeddings = nn.Parameter(torch.zeros(1, img_size[0]//2 * img_size[0]//2, out_dim))
        trunc_normal_(self.position_embeddings, std=.02)


    def forward(self, hidden_states):
        B1, N1, C1 = hidden_states.shape
        hidden_states = self.norm0(hidden_states)

        residual = hidden_states.permute(0, 2, 1).reshape(B1, C1, self.H, self.W)
        residual = self.pool(residual).reshape(B1, C1, -1).permute(0, 2, 1).contiguous()

        hidden_states = hidden_states.permute(0, 2, 1).reshape(B1, C1, self.H, self.W)
        hidden_states = self.proj(hidden_states + self.move.expand_as(hidden_states))
        B2, C2, H2, W2 = hidden_states.shape
        hidden_states = hidden_states.reshape(B2, C2, -1).permute(0, 2, 1).contiguous()

        residual = torch.concat([residual for _ in range(C2 // C1)], dim=-1)

        hidden_states = self.norm(hidden_states) + residual
        hidden_states = self.rprelu(hidden_states)

        return hidden_states + self.position_embeddings



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=2, in_dim=3, out_dim=64, config=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0], img_size[1]
        self.num_patches = self.H * self.W
        
        self.norm0 = config.norm_layer(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.norm = config.norm_layer(out_dim)

        self.position_embeddings = nn.Parameter(torch.zeros(1, img_size[0]//2 * img_size[0]//2, out_dim))
        trunc_normal_(self.position_embeddings, std=.02)


    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        hidden_states = self.norm0(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(B, C, self.H, self.W)
        hidden_states = self.proj(hidden_states).flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)

        return hidden_states + self.position_embeddings


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.num_hidden_layers)]
        self.layer = nn.ModuleList([ViTLayer(config, i, drop_path=dpr[i]) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.patch_embed1 = PatchEmbed(56, in_dim=config.hidden_size[0], out_dim=config.hidden_size[1], config=config)
        self.patch_embed2 = PatchEmbed(28, in_dim=config.hidden_size[1], out_dim=config.hidden_size[2], config=config)
        self.patch_embed3 = PatchEmbed(14, in_dim=config.hidden_size[2], out_dim=config.hidden_size[3], config=config)
            
        self.depths = config.depths

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if i == self.depths[0] - 1:
                hidden_states = self.patch_embed1(hidden_states)
            elif i == self.depths[0] + self.depths[1] - 1:
                hidden_states = self.patch_embed2(hidden_states)
            elif i == self.depths[0] + self.depths[1] + self.depths[2] - 1:
                hidden_states = self.patch_embed3(hidden_states)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = config.norm_layer(config.hidden_size[3], eps=config.layer_norm_eps)

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def generating_stage_per_depth(depths):
    i = 0
    stage_per_depth = []
    current_stage_depth = depths[i]
    while True:
        current_stage_depth -= 1
        stage_per_depth.append(i)
        if current_stage_depth == 0:
            i += 1
            if i == len(depths):
                break
            current_stage_depth = depths[i]
    return stage_per_depth


class ViTForImageClassification(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.num_labels = config.num_labels
        config.num_hidden_layers = sum(config.depths)
        config.stages = generating_stage_per_depth(config.depths)

        self.vit = ViTModel(config)
        self.config = config

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size[3], config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.apply(self.init_weights)


    @torch.no_grad()
    def init_weights(module: nn.Module, name: str = ''):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(torch.mean(sequence_output, dim=1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embeddings', 'cls_token', 'dist_token'}