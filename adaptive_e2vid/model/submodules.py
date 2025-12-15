import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 BN_momentum=0.1):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # 可选的激活函数:
        # 'relu': nn.ReLU(inplace=True)
        # 'tanh': nn.Tanh()
        # 'leaky_relu': nn.LeakyReLU(0.1, inplace=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation is None:
            self.activation = None
        else:
            raise NotImplementedError(f'activation function [{activation}] is not found')

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state
    
class RecurrentConvLayer_Sparse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1, axis_scale=1):
        super(RecurrentConvLayer_Sparse, self).__init__()
        # For E2VID, kernel_size=5, stride=2, padding=2

        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            raise NotImplementedError("Only ConvLSTM is implemented for sparse version.")

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm, BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3, external_padding=True) # Do padding manually (when cropped area is in center, directly take existing outer pixels as padding)
        self.axis_scale = axis_scale

    def forward(self, x, prev_state, ey, ex, eh, ew):
        # prev_state corresponds to the whole frame with shape [H, W]. It has shape [B, self.hidden_size, H//axis_scale//2, W//axis_scale//2].
        # x is the feature corresponding to frame[ey:ey+eh, ex:ex+ew].
        # x has shape [B, in_channels, eh//axis_scale, ew//axis_scale].
        x_conv = self.conv(x)
        # x_conv has shape [B, in_channels, eh//axis_scale//2, ew//axis_scale//2] due to stride=2 & padding in self.conv.
        partial_prev_state_padded = (
            crop_with_padding(
                prev_state[0], ey//self.axis_scale//2, ex//self.axis_scale//2, eh//self.axis_scale//2, ew//self.axis_scale//2, padding=self.recurrent_block.kernel_size//2),
            crop_with_padding(
                prev_state[1], ey//self.axis_scale//2, ex//self.axis_scale//2, eh//self.axis_scale//2, ew//self.axis_scale//2, padding=self.recurrent_block.kernel_size//2)
        )
        pad = self.recurrent_block.kernel_size//2
        x_conv_padded = f.pad(x_conv, (pad, pad, pad, pad), mode='constant')
        
        partial_new_state = self.recurrent_block(x_conv_padded, partial_prev_state_padded)
        feat = partial_new_state[0]

        # update the prev_state
        new_state_0 = prev_state[0].clone()
        new_state_1 = prev_state[1].clone()
        new_state_0[..., ey//self.axis_scale//2:(ey+eh)//self.axis_scale//2, ex//self.axis_scale//2:(ex+ew)//self.axis_scale//2] = partial_new_state[0]
        new_state_1[..., ey//self.axis_scale//2:(ey+eh)//self.axis_scale//2, ex//self.axis_scale//2:(ex+ew)//self.axis_scale//2] = partial_new_state[1]
        
        return feat, (new_state_0, new_state_1)

def crop_with_padding(arr, ey, ex, eh, ew, padding):
    # arr has shape [B, C, H, W]
    # crop arr[..., ey:ey+eh, ex:ex+ew] with padding
    B, C, H, W = arr.shape
    ey_pad = max(0, padding - ey)
    ex_pad = max(0, padding - ex)
    eh_pad = max(0, ey + eh + padding - H)
    ew_pad = max(0, ex + ew + padding - W)
    arr_cropped = arr[..., max(0, ey - padding):min(H, ey + eh + padding), max(0, ex - padding):min(W, ex + ew + padding)]
    if ey_pad > 0 or ex_pad > 0 or eh_pad > 0 or ew_pad > 0:
        arr_cropped = f.pad(arr_cropped, (ex_pad, ew_pad, ey_pad, eh_pad), mode='constant')
    return arr_cropped


class RecurrentConvLayer_Sparse_GU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1, axis_scale=1, global_feature_channels=128):
        super(RecurrentConvLayer_Sparse_GU, self).__init__()
        # For E2VID, kernel_size=5, stride=2, padding=2

        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            raise NotImplementedError("Only ConvLSTM is implemented for sparse version.")

        self.conv = ConvLayer(in_channels+global_feature_channels, out_channels+global_feature_channels, kernel_size, stride, padding, activation="tanh", norm=None, BN_momentum=BN_momentum)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3, external_padding=True) # Do padding manually (when cropped area is in center, directly take existing outer pixels as padding)
        self.axis_scale = axis_scale
        self.in_channels = in_channels
        self.global_feature_channels = global_feature_channels
        self.out_channels = out_channels
        self.global_activation = nn.Tanh()  # 推荐使用Tanh

    def forward(self, x, prev_state, ey, ex, eh, ew, global_feature):
        # prev_state corresponds to the whole frame with shape [H, W]. It has shape [B, self.hidden_size, H//axis_scale//2, W//axis_scale//2].
        # x is the feature corresponding to frame[ey:ey+eh, ex:ex+ew].
        # x has shape [B, in_channels, eh//axis_scale, ew//axis_scale].
        # global_feature is [B, F]. Broadcast it to [B, F, eh//axis_scale, ew//axis_scale] and concatenate it with x.
        gf_broadcast = global_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, eh//self.axis_scale, ew//self.axis_scale)
        x_with_gf = torch.cat([x, gf_broadcast], dim=1)
        B, F = global_feature.shape
        B, C, H, W = x.shape

        x_conv = self.conv(x_with_gf) # [B, in_channels+F, eh//axis_scale//2, ew//axis_scale//2]
        # The first in_channels channels are used later. The later F channels are average pooled to update the global feature.
        gf_update = self.avg_pool(x_conv[:, self.out_channels:, ...]).squeeze(-1).squeeze(-1) # [B, F]
        #gf_update = torch.zeros_like(gf_update)
        #print("Min, max and avg of gf_update:", gf_update.min().item(), gf_update.max().item(), gf_update.mean().item())
        x_conv_feat = x_conv[:, :self.out_channels, ...]
        # x_conv has shape [B, in_channels, eh//axis_scale//2, ew//axis_scale//2] due to stride=2 & padding in self.conv.

        partial_prev_state_padded = (
            crop_with_padding(
                prev_state[0], ey//self.axis_scale//2, ex//self.axis_scale//2, eh//self.axis_scale//2, ew//self.axis_scale//2, padding=self.recurrent_block.kernel_size//2),
            crop_with_padding(
                prev_state[1], ey//self.axis_scale//2, ex//self.axis_scale//2, eh//self.axis_scale//2, ew//self.axis_scale//2, padding=self.recurrent_block.kernel_size//2)
        )
        x_conv_feat_padded = f.pad(x_conv_feat, (self.recurrent_block.kernel_size//2, self.recurrent_block.kernel_size//2, self.recurrent_block.kernel_size//2, self.recurrent_block.kernel_size//2), mode='constant')
        
        partial_new_state = self.recurrent_block(x_conv_feat_padded, partial_prev_state_padded)
        feat = partial_new_state[0]

        # update the prev_state
        new_state_0 = prev_state[0].clone()
        new_state_1 = prev_state[1].clone()
        new_state_0[..., ey//self.axis_scale//2:(ey+eh)//self.axis_scale//2, ex//self.axis_scale//2:(ex+ew)//self.axis_scale//2] = partial_new_state[0]
        new_state_1[..., ey//self.axis_scale//2:(ey+eh)//self.axis_scale//2, ex//self.axis_scale//2:(ex+ew)//self.axis_scale//2] = partial_new_state[1]
        
        new_global_feature =self.global_activation(global_feature + gf_update) # Residual update
        return feat, (new_state_0, new_state_1), new_global_feature


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation)

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size, external_padding=False):
        # External padding: The passed in input_ and prev_state are already padded by kernel_size//2.
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.external_padding = external_padding
        if not external_padding:
            pad = kernel_size // 2
        else:
            pad = 0

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype, device=input_.device),
                    torch.zeros(state_size, dtype=input_.dtype, device=input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state
        if self.external_padding:
            # Unpad prev_cell
            pad = self.kernel_size // 2
            prev_cell = prev_cell[..., pad:-pad, pad:-pad]

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype, device=input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class RecurrentResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 recurrent_block_type='convlstm', norm=None, BN_momentum=0.1):
        super(RecurrentResidualLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  norm=norm,
                                  BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels,
                                              hidden_size=out_channels,
                                              kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state
