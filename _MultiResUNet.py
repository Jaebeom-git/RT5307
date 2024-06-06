import torch
import torch.nn as nn

def Conv_Block(in_channels, out_channels, kernel_size, multiplier):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels * multiplier, kernel_size, padding='same'),  # [Batch, in_channels, Length] -> [Batch, out_channels * multiplier, Length]
        nn.BatchNorm1d(out_channels * multiplier),  # [Batch, out_channels * multiplier, Length]
        nn.ReLU()  # [Batch, out_channels * multiplier, Length]
    )

def trans_conv1D(in_channels, out_channels, multiplier):
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels * multiplier, kernel_size=2, stride=2, padding=0),  # [Batch, in_channels, Length] -> [Batch, out_channels * multiplier, 2 * Length]
        nn.BatchNorm1d(out_channels * multiplier),  # [Batch, out_channels * multiplier, 2 * Length]
        nn.ReLU()  # [Batch, out_channels * multiplier, 2 * Length]
    )

def Concat_Block(input1, *argv):
    return torch.cat([input1] + list(argv), dim=1)  # Concatenate along the channel dimension [Batch, Channels, Length]

def upConv_Block():
    return nn.Upsample(scale_factor=2, mode='nearest')  # [Batch, in_channels, Length] -> [Batch, in_channels, 2 * Length]

def Feature_Extraction_Block(in_channels, seq_len, feature_number):
    return nn.Sequential(
        nn.Flatten(),  # [Batch, Channels, Length] -> [Batch, Channels*Length]
        nn.Linear(in_channels * seq_len, feature_number),  # [Batch, Channels*Length] -> [Batch, feature_number]
        nn.Linear(feature_number, in_channels * seq_len),  # [Batch, feature_number] -> [Batch, Channels*Length]
        nn.Unflatten(1, (in_channels, seq_len))  # [Batch, Channels*Length] -> [Batch, Channels, Length]
    )

class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier):
        super(MultiResBlock, self).__init__()

        # Make out_channels a multiple of 18
        if out_channels % 18 != 0:
            w = (out_channels // 18 + 1) * 18
        else:
            w = out_channels

        self.conv3x3 = Conv_Block(in_channels, int(w * 0.167), kernel_size, multiplier)  # [Batch, in_channels, Length] -> [Batch, int(w * 0.167), Length]
        self.conv5x5 = Conv_Block(int(w * 0.167), int(w * 0.333), kernel_size, multiplier)  # [Batch, int(w * 0.167), Length] -> [Batch, int(w * 0.333), Length]
        self.conv7x7 = Conv_Block(int(w * 0.333), int(w * 0.5), kernel_size, multiplier)  # [Batch, int(w * 0.333), Length] -> [Batch, int(w * 0.5), Length]

        total_out_channels = int(w * 0.167) + int(w * 0.333) + int(w * 0.5)
        self.shortcut = nn.Conv1d(in_channels, total_out_channels, kernel_size=1)  # [Batch, in_channels, Length] -> [Batch, total_out_channels, Length]

        self.bn = nn.BatchNorm1d(total_out_channels)  # [Batch, total_out_channels, Length]
        self.relu = nn.ReLU()  # [Batch, total_out_channels, Length]

        # Final adjustment layer to match the desired out_channels
        self.adjust_channels = nn.Conv1d(total_out_channels, out_channels, kernel_size=1)  # [Batch, total_out_channels, Length] -> [Batch, out_channels, Length]

    def forward(self, x):
        shortcut = self.shortcut(x)  # [Batch, in_channels, Length] -> [Batch, total_out_channels, Length]
        conv3x3 = self.conv3x3(x)  # [Batch, in_channels, Length] -> [Batch, int(w * 0.167), Length]
        conv5x5 = self.conv5x5(conv3x3)  # [Batch, int(w * 0.167), Length] -> [Batch, int(w * 0.333), Length]
        conv7x7 = self.conv7x7(conv5x5)  # [Batch, int(w * 0.333), Length] -> [Batch, int(w * 0.5), Length]
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)  # Concatenate along the channel dimension [Batch, total_out_channels, Length]
        out = self.bn(out)  # [Batch, total_out_channels, Length]
        out = out + shortcut  # Add the shortcut connection [Batch, total_out_channels, Length]
        out = self.relu(out)  # [Batch, total_out_channels, Length]

        # Adjust the output channels to match the desired out_channels
        out = self.adjust_channels(out)  # [Batch, total_out_channels, Length] -> [Batch, out_channels, Length]
        return out

class ResPath(nn.Module):
    def __init__(self, in_channels, depth, model_width, kernel_size, multiplier):
        super(ResPath, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size, padding='same'),  # [Batch, in_channels, Length] -> [Batch, in_channels, Length]
                nn.BatchNorm1d(in_channels),  # [Batch, in_channels, Length]
                nn.ReLU(),  # [Batch, in_channels, Length]
            ))

    def forward(self, x):
        shortcut = x  # [Batch, in_channels, Length]
        for block in self.blocks:
            x = block(x)  # [Batch, in_channels, Length] -> [Batch, in_channels, Length]
            x = x + shortcut  # Add the shortcut connection [Batch, in_channels, Length]
            shortcut = x  # [Batch, in_channels, Length]
        return x

class UNet(nn.Module):
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_channels=1, ds=0, ae=0, feature_number=1024, is_transconv=True):
        super(UNet, self).__init__()

        self.model_depth = model_depth
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_channels = output_channels
        self.D_S = ds
        self.A_E = ae
        self.feature_number = feature_number
        self.is_transconv = is_transconv
        self.input_length = length

        self.encoder = nn.ModuleList()
        self.respaths = nn.ModuleList()
        in_channels = num_channel  # [Batch, Length, Channels]
        for i in range(1, model_depth + 1):
            out_channels = model_width * (2 ** (i - 1))
            mresblock = MultiResBlock(in_channels, out_channels, kernel_size, 1)  # [Batch, in_channels, Length] -> [Batch, out_channels, Length]
            self.encoder.append(mresblock)  # [Batch, out_channels, Length]
            self.encoder.append(nn.MaxPool1d(2))  # [Batch, out_channels, Length] -> [Batch, out_channels, Length // 2]
            self.respaths.append(ResPath(out_channels, model_depth - i + 1, model_width, kernel_size, 1))
            in_channels = out_channels  # [Batch, out_channels, Length // 2]

        if ae == 1:
            self.feature_extraction = Feature_Extraction_Block(in_channels, length // (2 ** model_depth), feature_number)  # [Batch, out_channels, Length // 2 ** model_depth]

        self.bottleneck = MultiResBlock(in_channels, model_width * (2 ** model_depth), kernel_size, 1)  # [Batch, in_channels, Length // 2 ** model_depth] -> [Batch, model_width * (2 ** model_depth), Length // 2 ** model_depth]

        self.decoder = nn.ModuleList()
        self.deep_supervision = nn.ModuleList()
        self.upsample_to_input = nn.ModuleList()
        in_channels = model_width * (2 ** model_depth)  # [Batch, model_width * (2 ** model_depth), Length // 2 ** model_depth]
        for j in range(0, model_depth):
            out_channels = model_width * (2 ** (model_depth - j - 1))
            if self.is_transconv:
                upconv = trans_conv1D(in_channels, out_channels, 1)  # [Batch, in_channels, Length] -> [Batch, out_channels, 2 * Length]
            else:
                upconv = upConv_Block()  # [Batch, in_channels, Length] -> [Batch, in_channels, 2 * Length]
            self.decoder.append(upconv)  # [Batch, out_channels, 2 * Length] 또는 [Batch, in_channels, 2 * Length]
            self.decoder.append(MultiResBlock(in_channels, out_channels, kernel_size, 1))  # [Batch, in_channels + out_channels, 2 * Length] -> [Batch, out_channels, 2 * Length]
            in_channels = out_channels  # [Batch, out_channels, 2 * Length]
            
            if ds == 1:
                self.deep_supervision.append(nn.Conv1d(out_channels, output_channels, kernel_size=1))  # [Batch, out_channels, Length] -> [Batch, output_channels, Length]
                self.upsample_to_input.append(nn.Upsample(size=self.input_length, mode='nearest'))  # [Batch, out_channels, Length] -> [Batch, out_channels, input_length]

        if problem_type == 'Classification':
            self.final_conv = nn.Conv1d(in_channels, output_channels, kernel_size=1)  # [Batch, in_channels, Length] -> [Batch, output_channels, Length]
            self.final_activation = nn.Softmax(dim=1)  # [Batch, output_channels, Length]
        elif problem_type == 'Regression':
            self.final_conv = nn.Conv1d(in_channels, output_channels, kernel_size=1)  # [Batch, in_channels, Length] -> [Batch, output_channels, Length]
            self.final_activation = nn.Identity()  # [Batch, output_channels, Length]

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [Batch, Length, Channels] -> [Batch, Channels, Length]

        enc_outs = []
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)  # [Batch, Channels, Length] -> [Batch, Channels, Length]
            if self.D_S == 1:
                x = self.respaths[i // 2](x)  # Apply ResPath for deep supervision [Batch, Channels, Length]
            enc_outs.append(x)  # Encoder output 저장
            x = self.encoder[i+1](x)  # [Batch, Channels, Length] -> [Batch, Channels, Length // 2]

        if self.A_E == 1:
            x = self.feature_extraction(x)  # [Batch, Channels, Length // 2 ** model_depth]

        x = self.bottleneck(x)  # [Batch, Channels, Length // 2 ** model_depth]

        ds_outputs = []
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsample [Batch, Channels, Length] -> [Batch, Channels, 2 * Length]
            enc_out = enc_outs[-(i // 2 + 1)]  # [Batch, Channels, Length]
            x = Concat_Block(x, enc_out)  # Concatenate skip connection [Batch, Channels, 2 * Length]
            x = self.decoder[i+1](x)  # MultiResBlock 적용 [Batch, Channels, 2 * Length]
            if self.D_S == 1:
                ds_out = self.deep_supervision[i // 2](x)  # Deep supervision output [Batch, output_channels, Length]
                ds_out = self.upsample_to_input[i // 2](ds_out).permute(0, 2, 1)  # [Batch, output_channels, Length] -> [Batch, Length, output_channels]
                ds_outputs.append(ds_out)

        x = self.final_conv(x)  # Final convolution [Batch, Channels, Length]
        x = self.final_activation(x)  # Activation function [Batch, Channels, Length]
        x = x.permute(0, 2, 1)  # [Batch, Channels, Length] -> [Batch, Length, Channels]
        
        if self.D_S == 1:
            ds_outputs.append(x)  # Append final output for deep supervision
            return ds_outputs
        else:
            return x