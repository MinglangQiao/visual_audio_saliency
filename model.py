
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torchviz import make_dot, make_dot_from_trace
import torch.nn.functional as F
from torchvision import models

import numpy as np

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

class ConvLSTMCell(nn.Module):
    
    """
    0413 update by ml

    Parameters:
    -----------

    Refercence:
    -----------
    https://github.com/automan000/Convolution_LSTM_pytorch

    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size, convlstm_version="version_simple"):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.convlstm_version = convlstm_version

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):

        if self.convlstm_version == "version_simple":
            ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
            cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
            cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h)) ## the later part is g
            co = torch.sigmoid(self.Wxo(x) + self.Who(h))
            ch = co * torch.tanh(cc)
        else:
            ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
            cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
            cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
            co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
            ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if not self.convlstm_version == "version_simple":
            if self.Wci is None:
                # self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                # self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                # self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                
                #### refer to: https://github.com/automan000/Convolution_LSTM_PyTorch/issues/9
                ####  self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(cfg.GLOBAL.DEVICE)
                self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
                self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()

                # print(">>>>>>>>>> reset wci, wcf, wco")
            else:
                print(">>>> debug lstm: {}, {}, {}, {}".format(shape[0], self.Wci.size()[2], 
                    shape[1], self.Wci.size()[3]))

                assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
                assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        else:
            pass

        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class visual_audio_face_modal_combine_net(nn.Module):
    """
    0901 update

    """
    def __init__(self, train_batch_size=12, output_dim=32, Fusion_Type='concate', lstm_input_channels=128, lstm_hidden_channels=[128, 64], lstm_kernel_size=3, 
        batchNorm=False, Return_feature_map=False):
        super(visual_audio_face_modal_combine_net, self).__init__()

        ### visual subnet
        flow_channel_for_combine = 128
        visual_cnn_channel_for_combine = 512
        combine_chanel = flow_channel_for_combine + visual_cnn_channel_for_combine

        self.convlstm_version = "version_simple"
        self.fusion_type = Fusion_Type
        self.return_feature_map = Return_feature_map

        ## cnn
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, visual_cnn_channel_for_combine, kernel_size=3, padding=1)

        ## flow
        self.batchNorm = batchNorm
        self.flow_conv1 = conv(self.batchNorm,  6,   64, kernel_size=7, stride=2)
        self.flow_conv2  = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.flow_conv3  = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.flow_deconv5 = deconv(256, flow_channel_for_combine)

        ## combine cnn+flow
        self.combine_conv1 = nn.Conv2d(combine_chanel, 256, kernel_size=3, padding=1)
        self.combine_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.combine_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        ## lstm
        self.input_channels = [lstm_input_channels] + lstm_hidden_channels # [512, 128, 64, 64, 32, 32] = [512] + [128, 64, 64, 32, 32] 
        self.hidden_channels = lstm_hidden_channels
        self.kernel_size = lstm_kernel_size
        self.num_layers = len(lstm_hidden_channels)

        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, 
                convlstm_version=self.convlstm_version)
            setattr(self, name, cell)
            self._all_layers.append(cell)
        
        ### audio subnet
        self.train_bach_size = train_batch_size
        self.output_dim = output_dim

        ## 3d cnn part
        self.conv1_3d = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2a_3d = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2b_3d = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2_3d = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.conv3_3d = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3_3d = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.relu = nn.ReLU()

        #### combine visual and audio subnet
        if self.fusion_type == "concate":
            self.conv_normal_visual1 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # 1 is for combine(element-wise addtion) 3 modal feature     
            self.conv_normal_visual2 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 2 is for concate combined-modal-feature and each individal modal geature
            self.conv_normal_audio1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.conv_normal_audio2 = nn.Conv2d(192, 64, kernel_size=3, padding=1) ###
            self.conv_normal_face1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) ## for 3 modal combination
            self.conv_normal_face2 = nn.Conv2d(65, 64, kernel_size=3, padding=1)

            self.visual_hmap = nn.Conv2d(192, 1, kernel_size=1, padding=0) ## 64(visual) + 64(audio) = 128

    def forward(self, x1, x2, x3, x4, i_batch=0):
        """
        0820 update by ml

        x1: frame_1
        x2: frame_6
        x3: frame_6's audio melspectr, shape (B, C, T, H, W)= (12, 1, 16, 112, 112) in one GPU case
        x4: face subnet gmm result, one channel image, [12, 1, 32, 32]
        """

        #### visual subnet
        x = x2
        y = torch.cat((x1, x2), 1)

        ### object part
        x = F.relu(self.conv1_1(x))
        # save_feature_map_for_CR(x, i_batch, i_conv='1_1')
        x = F.relu(self.conv1_2(x))

        x = F.relu(self.conv2_1(self.pool(x)))
        x = F.relu(self.conv2_2(x))
        # for save_feature_map

        x = F.relu(self.conv3_1(self.pool(x)))
        # save_feature_map_for_CR(x, i_batch, i_conv='3_1')
        x = F.relu(self.conv4_1(self.pool(x))) 
        

        ### flow part
        y = self.flow_conv2(self.flow_conv1(y))
        # save_feature_map_for_CR(y, i_batch, i_conv='flow_conv2')
        y = self.flow_conv3(y)
        y = self.flow_deconv5(self.pool(y))

        #### combine part
        z =  torch.cat((y, x), 1)
        z = F.relu(self.combine_conv1(z))
        z = F.relu(self.combine_conv2(z))
        z = F.relu(self.combine_conv3(z))
        # print(">>>> dbeug: ", np.shape(z)) # [12, 128, 32, 32]
        # t

        #### transport to the lstm part
        input1 = z 
        internal_state = []
        outputs = []
        self.step = input1.size()[0]

        for step in range(self.step): ### the step is necessary, as in 
            x = input1[step]
            x = x[np.newaxis, ...]

            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
                ## ml add
                # print(">>>> step: {}, conv_lstm layer: {}, shape:{}".format(step, i, np.shape(internal_state)))

                if i == (self.num_layers - 1):
                    outputs.append(x)
                    # print(">>>> np.shape(x): before cat: ",  x.size()) ## for debug [1, 64, 32, 32]
        outputs = torch.cat(outputs) ## this is the convlstm output (batch, 64, 32, 32)
        # print(">>>> np.shape(outputs): after cat: ", np.shape(outputs)) ## for debug [12, 64, 32, 32]
        # t

        #### audio subnet, will downscale the input by 2
        h_3d = self.relu(self.conv1_3d(x3)) ## [12, 16, 16, 112, 112]
        h_3d = self.pool1_3d(h_3d) # [12, 16, 8, 56, 56]

        h_3d = self.relu(self.conv2a_3d(h_3d)) # [12, 32, 8, 56, 56]
        h_3d = self.relu(self.conv2b_3d(h_3d)) # [12, 32, 8, 56, 56]
        h_3d = self.pool2_3d(h_3d) # [12, 32, 4, 56, 56]

        h_3d = self.relu(self.conv3_3d(h_3d)) # [12, 64, 4, 56, 56]
        h_3d = self.pool3_3d(h_3d) # [12, 64, 2, 56, 56]
        ## commnet by ml in 1108
        h_3d_h, h_3d_w = np.shape(h_3d)[3], np.shape(h_3d)[4]
        h_3d = h_3d.view(self.train_bach_size, -1, h_3d_h, h_3d_w) # [12, 128, 56, 56]

        #### combine visual and audio subnet
        # print(">>>> np.shape(outputs)  ", np.shape(h_3d), np.shape(outputs)) ## for debug [12, 128, 32, 32] [12, 64, 32, 32]
        if self.fusion_type in ["concate", "addition", "remove_fusion", "multiple"]:
            normal1_visual_feature = F.relu(self.conv_normal_visual1(outputs))
            normal1_audio_feature = F.relu(self.conv_normal_audio1(h_3d))
            normal1_face_feature = F.relu(self.conv_normal_face1(x4))

            combine_feature = torch.cat([normal1_visual_feature, normal1_audio_feature, normal1_face_feature], dim=1) # [12, 192, 32, 32]
            n, c, h, w = combine_feature.size() # [12, 128, 32, 32]
            modal_num, interval = 3, 64

        if self.fusion_type == "concate":
            ## addition
            combine_feature = combine_feature.reshape(n, c//modal_num//interval, modal_num, interval, h, w) # [12, 1, 3, 64, 32, 32]
            combine_feature = combine_feature.sum(dim=2) # [12, 1, 64, 32, 32]
            combine_feature = combine_feature.view(n, c//modal_num, h, w) ## [12, 64, 32, 32]
            ## concate
            concate_visual_features = torch.cat([combine_feature, outputs], dim=1) ## size=(12, 128, 32, 32)
            concate_audio_freatures = torch.cat([combine_feature, h_3d], dim=1) # size = [12, 192, 32, 32]
            concate_face_freatures = torch.cat([combine_feature, x4], dim=1) # size = [12, 65, 32, 32]
            ## conv
            final_visual_features = self.relu(self.conv_normal_visual2(concate_visual_features))
            final_audio_features = self.relu(self.conv_normal_audio2(concate_audio_freatures))
            final_face_features = self.relu(self.conv_normal_face2(concate_face_freatures)) # [12, 64, 32, 32]

            ### concate and conv
            final_saliency_map = self.visual_hmap(torch.cat([final_visual_features, final_audio_features, final_face_features], dim=1))
            # print(">>>> final_salmap: {}".format(np.shape(final_saliency_map))) # [12, 1, 32, 32]
            # t
        
        return final_saliency_map