import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class FullyConvNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        motion = self.conv(x)
        return motion


class FullyConvResNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvResNet, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(2*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(num_hidden)
        self.conv1_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(num_hidden)
        self.conv2_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(num_hidden)
        self.conv2_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(num_hidden)
        self.conv3_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(num_hidden)
        self.conv3_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(num_hidden)
        self.conv4_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(num_hidden)
        self.conv4_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(num_hidden)
        self.conv5_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_1 = nn.BatchNorm2d(num_hidden)
        self.conv5_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_2 = nn.BatchNorm2d(num_hidden)
        self.conv6_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(num_hidden)
        self.conv6_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_2 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.bn0(self.conv0(x))
        y = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(y) + x))
        y = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(y) + x))
        y = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(y) + x))
        y = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(y) + x))
        y = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(y) + x))
        y = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(y) + x))
        motion = self.conv(x)
        return motion


class UNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_class):
        super(UNet, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion = self.conv(x)
        return motion


class UNetBidirection(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(UNetBidirection, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(im_channel * n_inputs, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv12 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_hidden)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(num_hidden*2, n_class, 3, 1, 1)
        self.conv_a = nn.Conv2d(2, 1, 1, 1, 0)
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, ones):
        x = self.bn0(self.conv0(im_input_f))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion_f = self.conv(x)

        x = self.bn0(self.conv0(im_input_b))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x7 = self.maxpool(x6)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x6), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x5), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x4), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x3), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        x11 = self.upsample(x11)
        x12 = torch.cat((x11, x2), 1)
        x12 = F.relu(self.bn12(self.conv12(x12)))
        x12 = self.upsample(x12)
        x = torch.cat((x12, x1), 1)
        motion_b = self.conv(x)

        m_mask_f = F.softmax(motion_f)
        m_mask_b = F.softmax(motion_b)

        seg_f = construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        pred_f = construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, self.m_kernel, self.m_range)
        pred_b = construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, self.m_kernel, self.m_range)

        seg = torch.cat((seg_f, seg_b), 1)
        attn = self.conv_a(seg)
        attn = F.sigmoid(attn)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, pred_f, m_mask_f, attn, pred_b, m_mask_b, 1 - attn


def construct_seg(ones, m_mask, m_kernel, m_range):
    ones_expand = ones.expand_as(m_mask) * m_mask
    seg = Variable(torch.Tensor(ones.size()))
    if torch.cuda.is_available():
        seg = seg.cuda()
    for i in range(ones.size(0)):
        seg[i, :, :, :] = F.conv2d(ones_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return seg


def construct_image(im, m_mask, m_kernel, m_range):
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
        for j in range(im.size(0)):
            pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
    return pred
