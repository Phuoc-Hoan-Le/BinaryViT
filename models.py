from transformer.modeling_vit import ViTForImageClassification
from transformer.modeling_vit_extra_res import ViTForImageClassification as ViTForImageClassificationExtraRes
from transformer.modeling_vit_extra_res_pyramid import ViTForImageClassification as PyramidViTForImageClassification
from transformers import ViTConfig
import torch
import torch.nn as nn
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm



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
    


class SyncBatchNormT(nn.SyncBatchNorm):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group = None,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SyncBatchNormT, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.process_group = process_group

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError("SyncBatchNorm expected input tensor to be on GPU")
        # Need to transpose unlike original nn.SyncBatchNorm
        input = input.transpose(-1, -2)
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
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

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (bn_training and self.training)
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return nn.functional.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            ).transpose(-1, -2)
        else:
            assert bn_training
            return sync_batch_norm.apply(
                input,
                self.weight,
                self.bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,
                world_size,
            ).transpose(-1, -2)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        module_output = module
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNormT(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output



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
    config.some_fp = args.some_fp

    if model_type == "extra-res-pyramid":
        model = PyramidViTForImageClassification(config=config)
    elif model_type == "extra-res":
        model = ViTForImageClassificationExtraRes(config=config)
    elif model_type == "deit":    
        model = ViTForImageClassification(config=config)
    else:
        raise NotImplementedError("Need to specify a supported model type.")
    
    return model