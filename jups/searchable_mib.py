from collections import OrderedDict
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from enot.utils.latency import LatencyMixin
from enot.models.operations.common_operations import ConvBN
from enot.models.operations.common_operations import DepthwiseConvBN
from enot.models.operations.common_operations import calculate_hidden_channels
from enot.models.operations.common_operations import same_padding
from enot.models.operations.operations_registry import GLOBAL_ACTIVATION_FUNCTION_REGISTRY
from enot.models.operations.operations_registry import register_searchable_op
from enot.utils.latency import mib_mac_count
from enot.utils.pruning.split_prune_utils import compute_channels_to_keep
from enot.utils.pruning.utils import no_prune
from enot.utils.pruning.utils import prune_axis_0
from enot.utils.pruning.utils import prune_axis_1
from enot.utils.common_utils import input_hw
import logging

__all__ = [
    'SearchableMobileInvertedBottleneck',
]

_LOGGER = logging.getLogger('enot_debug_logger')


@register_searchable_op('MIB', {
    'k': ('kernel_size', int),
    't': ('expand_ratio', float),
    'activation': ('activation', str),
})
class SearchableMobileInvertedBottleneck(nn.Module, LatencyMixin):
    """
    Searchable block from MobileNetV2.
    .. _original paper:
    https://arxiv.org/pdf/1801.04381.pdf
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dw_channels: Optional[int] = None,
            expand_ratio: Optional[float] = None,
            kernel_size: int = 3,
            stride: int = 1,
            
            ###
            dilation: int = 1,
            ###
        
            # padding='SAME',
            padding: Optional[int] = None,
            affine: bool = True,
            track: bool = True,
            activation: Optional[str] = 'relu6',
            use_skip_connection: bool = True,
    ):
        """
        Parameters
        ----------
        in_channels: int
            Number of channels in the input tensor.
        out_channels: int
            Number of channels of output tensor.
        dw_channels: int, optional
            Number of channels in depthwise convolution.
            Only one of dw_channels and expand_ratio con be set.
            If both is None there is no expand operation and only depthwise and squeeze blocks will be compute.
        expand_ratio: float, optional
            Used for computing dw_channels.

            dw_channels = round(in_channels * expand_ratio).
        kernel_size: int
            Size of the convolving kernel in the depthwise convolution. Default value: 3
        stride: int
            Stride of the depthwise convolution. Default value: 1
        padding: int, optional
            Padding added to both sides of the input. Default value: None (that means 'same')
        affine: bool
            Flag for using affine in all BatchNorm-s of MIB.
        track: bool
            Flag for using track_running_stats in all BatchNorm-s of MIB.
        activation: str, optional
            Name for used activation function.

            This function must be registered in GLOBAL_ACTIVATION_FUNCTION_REGISTRY
            from enot.operations.operations_registry.

            If activation is None - there is no activation_function.

            Default value: 'relu6'
        use_skip_connection: bool
            Add skip connection (y+=x) if this flag is True
            and output produced by convolution has the same shape as input.
            Default value: True
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip_connection = use_skip_connection

        if padding is None:
            padding = same_padding(kernel_size)

        dw_channels = calculate_hidden_channels(in_channels, dw_channels, expand_ratio)

        # Save arguments that may be used for latency calculation
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.padding = padding
        self.dw_channels = dw_channels
        self.affine = affine
        self.track = track
        self.activation_function_name = activation
        self.activation_function = None
        if self.activation_function_name is not None:
            self.activation_function = GLOBAL_ACTIVATION_FUNCTION_REGISTRY[activation]()

        self.expand_op = None

        if dw_channels is not None:
            self.expand_op = nn.Sequential(
                ConvBN(
                    in_channels=in_channels,
                    out_channels=dw_channels,
                    kernel_size=1,
                    stride=1,
                    affine=affine,
                    track=track,
                ),
                self.activation_function,
            )
        else:
            dw_channels = in_channels

        self.depthwise_op = nn.Sequential(
            DepthwiseConvBN(
                in_channels=dw_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                affine=affine,
                track=track,
                
                ###
                dilation=dilation,
                ###
            ),
            self.activation_function,
        )

        self.squeeze_op = ConvBN(
            dw_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            affine=affine,
            track=track,
        )

    def prune(
            self,
            criteria: Callable[[nn.Module], torch.Tensor],
            keep_proportion: float,
    ) -> OrderedDict:
        """
        Prune mobilenet_v2 block with criteria and sparsity.
        Parameters
        ----------
        criteria:
            Selected pruning criteria.
        keep_proportion:
            Sparsity amount for criteria.

        Returns
        -------
        Pruned weights: OrdetedDict
        """

        important_channel_indices = compute_channels_to_keep(self, keep_proportion, criteria)
        weights = self._get_important_weights(important_channel_indices)
        return weights

    def _get_important_weights(
            self,
            channel_indices: List[int],
    ) -> OrderedDict:
        weights = OrderedDict()
        weight_transforms = {
            'expand_op.0.conv.weight': prune_axis_0,
            'expand_op.0.conv.bias': prune_axis_0,
            'expand_op.0.bn.weight': prune_axis_0,
            'expand_op.0.bn.bias': prune_axis_0,
            'expand_op.0.bn.running_mean': prune_axis_0,
            'expand_op.0.bn.running_var': prune_axis_0,
            'expand_op.0.bn.num_batches_tracked': no_prune,
            'depthwise_op.0.conv.weight': prune_axis_0,
            'depthwise_op.0.conv.bias': prune_axis_0,
            'depthwise_op.0.bn.weight': prune_axis_0,
            'depthwise_op.0.bn.bias': prune_axis_0,
            'depthwise_op.0.bn.running_mean': prune_axis_0,
            'depthwise_op.0.bn.running_var': prune_axis_0,
            'depthwise_op.0.bn.num_batches_tracked': no_prune,
            'squeeze_op.conv.weight': prune_axis_1,
            'squeeze_op.conv.bias': no_prune,
            'squeeze_op.bn.weight': no_prune,
            'squeeze_op.bn.bias': no_prune,
            'squeeze_op.bn.running_mean': no_prune,
            'squeeze_op.bn.running_var': no_prune,
            'squeeze_op.bn.num_batches_tracked': no_prune,
        }
        for weight_name, weight in self.state_dict().items():
            if weight_name not in weight_transforms.keys():
                raise RuntimeError(f"Unexpected key in state dict: {weight_name}")
            weights[weight_name] = weight_transforms[weight_name](weight, channel_indices)
        return weights

    def forward(self, x):
        y = x

        if self.expand_op is not None:
            y = self.expand_op(y)

        y = self.depthwise_op(y)
        y = self.squeeze_op(y)

        if self.use_skip_connection and x.shape == y.shape:
            _LOGGER.debug(f'[{self.__class__.__name__}] apply skip connection')
            y += x

        return y

    def latency_mmac(self, inputs: Tuple[torch.Tensor, ...]):

        expand_ratio = self.expand_ratio
        if expand_ratio is None and self.dw_channels:
            expand_ratio = round(self.dw_channels / self.in_channels)

        return mib_mac_count(
            spatial_size=input_hw(inputs),
            kernel_size=self.kernel_size,
            expand_ratio=expand_ratio,
            stride=self.stride,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            padding=self.padding,
            calculate_in_millions=True
        )
