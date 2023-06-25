import torch
import torch.nn as nn


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        # indicate_small = (input < -1).float()
        # indicate_big = (input > 1).float()
        indicate_leftmid = ((input >= -1) & (input <= 0)).float()
        indicate_rightmid = ((input > 0) & (input <= 1)).float()

        grad_input = (indicate_leftmid * (2 + 2*input) + indicate_rightmid * (2 - 2*input)) * grad_output.clone()
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input - 0.35 * torch.max(input, dim=-1, keepdim=True)[0])
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input
    

class BiTHeadBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val):
        input = input.view(-1, 6, 196*196).transpose(-1, -2)
        ctx.save_for_backward(input, clip_val)
        out = torch.round(input / clip_val).clamp(0.0, 1.0) * clip_val
        return out.transpose(-1, -2).view(-1, 6, 196, 196)

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        q_w = input / clip_val
        indicate_small = (q_w < 0.0).float()
        indicate_big = (q_w > 1.0).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        grad_clip_val = ((indicate_middle * (q_w.round() - q_w) + indicate_big) 
                            * grad_output.view(-1, 6, 196*196).transpose(-1, -2)).sum(dim=1).sum(dim=0).unsqueeze(dim=0)
        grad_input = indicate_middle.transpose(-1, -2).view(-1, 6, 196, 196) * grad_output.clone()
        return grad_input, grad_clip_val


class BiTHeadMaxBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val):
        # B, 6, 196, 196
        input = input.permute(0, 2, 3, 1)
        # B, 196, 196, 6
        scale = torch.max(torch.abs(input), dim=-2, keepdim=True)[0].expand_as(input).detach() * clip_val
        ctx.save_for_backward(input, scale) 
        out = torch.round(input / scale).clamp(0.0, 1.0) * scale
        return out.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors
        q_w = input / scale
        indicate_small = (q_w < 0.0).float()
        indicate_big = (q_w > 1.0).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        M = torch.max(torch.abs(input), dim=-2, keepdim=True)[0].expand_as(input).detach()

        grad_clip_val = ((indicate_middle * (M * q_w.round() - M * q_w) + indicate_big) 
                            * grad_output.permute(0, 2, 3, 1)).sum(dim=2).sum(dim=1).sum(dim=0).unsqueeze(dim=0)
        grad_input = indicate_middle.permute(0, 3, 1, 2) * grad_output.clone()
        return grad_input, grad_clip_val


class BiTBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val):
        ctx.save_for_backward(input, clip_val)
        out = torch.round(input / clip_val).clamp(0.0, 1.0) * clip_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        q_w = input / clip_val
        indicate_small = (q_w < 0.0).float()
        indicate_big = (q_w > 1.0).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        grad_clip_val = ((indicate_middle * (q_w.round() - q_w) + indicate_big) * grad_output).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output.clone()
        return grad_input, grad_clip_val


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(self,  *kargs, bias=True, config=None):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
        if self.input_bits < 32:
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            elif self.input_bits == 2:
                self.act_quantizer = TwnQuantizer
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            else:
                self.act_quantizer = SymQuantizer
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
 

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.input_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            ba = self.act_quantizer.apply(input)
        
        out = nn.functional.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out


class QuantizeConv2d(nn.Conv2d):
    def __init__(self,  *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
        if self.input_bits < 32:
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            elif self.input_bits == 2:
                self.act_quantizer = TwnQuantizer
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            else:
                self.act_quantizer = SymQuantizer
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
 

    def forward(self, input):
        # This forward pass is meant for only binary weights and activations
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)

        ba = self.act_quantizer.apply(input)
        
        out = torch.nn.functional.conv2d(ba, binary_weights, stride=self.stride, padding=self.padding)
        
        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self,  *kargs,padding_idx=None, config=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out