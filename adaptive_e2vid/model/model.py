import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .unet import UNet, UNetFlow, WNet, UNetFlowNoRecur, UNetRecurrent, UNetRecurrent_Sparse, UNetRecurrent_Sparse_GU
from .submodules import ResidualBlock, ConvGRU, ConvLayer

def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)


class FlowNet(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlow(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetflow.states)

    @states.setter
    def states(self, states):
        self.unetflow.states = states

    def reset_states(self):
        self.unetflow.states = [None] * self.unetflow.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict


class FlowNetNoRecur(BaseModel):
    """
    UNet-like architecture without recurrent units
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlowNoRecur(unet_kwargs)

    def reset_states(self):
        pass

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict

class ColorE2VID(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlow(unet_kwargs, img_3c=True)

    @property
    def states(self):
        return copy_states(self.unetflow.states)

    @states.setter
    def states(self, states):
        self.unetflow.states = states

    def reset_states(self):
        self.unetflow.states = [None] * self.unetflow.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict

class E2VIDRecurrent(BaseModel):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict

class E2VIDSparse(BaseModel):
    """
    Parameter structure same as E2VIDRecurrent, but support sparse update.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent_Sparse(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self, B, H, W, device): # H, W are the full frame size
        self.unetrecurrent.reset_states(B, H, W, device)

    def forward(self, event_tensor, ey, ex, eh, ew):
        """
        The whole frame has shape [H, W].
        The incoming event tensor corresponds to [ey:ey+eh, ex:ex+ew] in the whole frame.

        :param event_tensor: N x num_bins x eh x eW
        :return: output dict with the image patch [ey:ey+eh, ex:ex+ew] taking values in [0,1].
        """
        output_dict = self.unetrecurrent.forward(event_tensor, ey, ex, eh, ew)
        return output_dict

class E2VIDSparse2(BaseModel):
    """
    Parameter structure same as E2VIDRecurrent, but support sparse update. Also add global updating in deepest layer.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent_Sparse_GU(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self, B, H, W, device): # H, W are the full frame size
        self.unetrecurrent.reset_states(B, H, W, device)

    def forward(self, event_tensor, ey, ex, eh, ew, begin_t, end_t):
        """
        The whole frame has shape [H, W].
        The incoming event tensor corresponds to [ey:ey+eh, ex:ex+ew] in the whole frame. The event tensor covers time span [begin_t, end_t] (unit is seconds).

        :param event_tensor: N x num_bins x eh x eW
        :return: output dict with the image patch [ey:ey+eh, ex:ex+ew] taking values in [0,1].
        """
        output_dict = self.unetrecurrent.forward(event_tensor, ey, ex, eh, ew, begin_t, end_t)
        return output_dict


class EVFlowNet(BaseModel):
    """
    Model from the paper: "EV-FlowNet: Self-Supervised Optical Flow for Event-based Cameras", Zhu et al. 2018.
    Pytorch adaptation of https://github.com/daniilidis-group/EV-FlowNet/blob/master/src/model.py (may differ slightly)
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        # put 'hardcoded' EVFlowNet parameters here
        EVFlowNet_kwargs = {
            'base_num_channels': 32, # written as '64' in EVFlowNet tf code
            'num_encoders': 4,
            'num_residual_blocks': 2,  # transition
            'num_output_channels': 2,  # (x, y) displacement
            'skip_type': 'concat',
            'norm': None,
            'use_upsample_conv': True,
            'kernel_size': 3,
            'channel_multiplier': 2
            }
        unet_kwargs.update(EVFlowNet_kwargs)

        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unet = UNet(unet_kwargs)

    def reset_states(self):
        pass

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with N x 2 X H X W (x, y) displacement within event_tensor.
        """
        flow = self.unet.forward(event_tensor)
        # to make compatible with our training/inference code that expects an image, make a dummy image.
        return {'flow': flow, 'image': 0 * flow[..., 0:1, :, :]}


class FireNet(BaseModel):
    """
    Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H x W image
        """
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {'image': self.pred(x)}
