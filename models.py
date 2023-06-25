from transformer.modeling_vit import ViTForImageClassification
from transformer.modeling_vit_extra_res import ViTForImageClassification as ViTForImageClassificationExtraRes
from transformer.modeling_qvit_extra_res import ViTForImageClassification as QViTForImageClassificationExtraRes
from transformer.modeling_vit_extra_res_pyramid import ViTForImageClassification as PyramidViTForImageClassification
from transformer.modeling_qvit_extra_res_pyramid import ViTForImageClassification as QPyramidViTForImageClassification
from transformers import ViTConfig
import torch.nn as nn


class BatchNormT(nn.BatchNorm1d):
    def __init__(self, *kargs, eps=1e-5):
        super(BatchNormT, self).__init__(*kargs, eps=eps)
    
    def forward(self, input):
        # Need to transpose unlike original nn.BatchNorm1d
        input = input.transpose(-1, -2)

        self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Output needs to be de-transposed unlike original nn.BatchNorm1d
        return nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        ).transpose(-1, -2)


def get_model(args, model_config, model_type, weight_bits, input_bits):
    config = ViTConfig.from_pretrained(model_config)
    config.drop_path = args.drop_path
    config.layer_norm_eps = 1e-5
    config.num_labels = args.nb_classes

    config.avg_res3 = args.avg_res3
    config.avg_res5 = args.avg_res5
    config.norm_layer = BatchNormT if args.replace_ln_bn else nn.LayerNorm
    config.disable_layerscale = args.disable_layerscale
    config.enable_cls_token = args.enable_cls_token

    config.weight_bits = weight_bits
    config.input_bits = input_bits
    config.att_prob_quantizer_type = args.att_prob_quantizer_type
    config.some_fp = args.some_fp

    if config.weight_bits == 32 and config.input_bits == 32:
        if model_type == "extra-res-pyramid":
            model = PyramidViTForImageClassification(config=config)
        elif model_type == "extra-res":
            model = ViTForImageClassificationExtraRes(config=config)
        else:    
            model = ViTForImageClassification(config=config)
    else:
        if model_type == "extra-res-pyramid":
            model = QPyramidViTForImageClassification(config=config)
        else:
            model = QViTForImageClassificationExtraRes(config=config)
    
    return model