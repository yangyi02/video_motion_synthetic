import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class NetGT(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(NetGT, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input, im_output, ones, gt_motion):
        m_mask = gt_motion
        seg = self.construct_seg(ones, m_mask, self.m_kernel, self.m_range)
        disappear = F.relu(seg - 1)
        appear = F.relu(1 - disappear)
        pred = self.construct_image(im_input[:, -self.im_channel:, :, :], m_mask, appear, self.m_kernel, self.m_range)
        seg = 1 - F.relu(1 - seg)
        return pred, m_mask, 1 - appear

    def construct_seg(self, ones, m_mask, m_kernel, m_range):
        ones_expand = ones.expand_as(m_mask) * m_mask
        seg = Variable(torch.Tensor(ones.size()))
        if torch.cuda.is_available():
            seg = seg.cuda()
        for i in range(ones.size(0)):
            seg[i, :, :, :] = F.conv2d(ones_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return seg

    def construct_image(self, im, m_mask, appear, m_kernel, m_range):
        im = im * appear.expand_as(im)
        pred = Variable(torch.Tensor(im.size()))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im.size(1)):
            im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
            for j in range(im.size(0)):
                pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return pred


class BiNetGT(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BiNetGT, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel

    def forward(self, im_input_f, im_input_b, ones, gt_motion_f, gt_motion_b):
        m_mask_f = gt_motion_f
        m_mask_b = gt_motion_b

        seg_f = self.construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = self.construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        disappear_f = F.relu(seg_f - 1)
        appear_f = F.relu(1 - disappear_f)
        disappear_b = F.relu(seg_b - 1)
        appear_b = F.relu(1 - disappear_b)

        pred_f = self.construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, appear_f, self.m_kernel, self.m_range)
        pred_b = self.construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, appear_b, self.m_kernel, self.m_range)

        seg_f = 1 - F.relu(1 - seg_f)
        seg_b = 1 - F.relu(1 - seg_b)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, m_mask_f, 1 - appear_f, attn, m_mask_b, 1 - appear_b, 1 - attn

    def construct_seg(self, ones, m_mask, m_kernel, m_range):
        ones_expand = ones.expand_as(m_mask) * m_mask
        seg = Variable(torch.Tensor(ones.size()))
        if torch.cuda.is_available():
            seg = seg.cuda()
        for i in range(ones.size(0)):
            seg[i, :, :, :] = F.conv2d(ones_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return seg

    def construct_image(self, im, m_mask, appear, m_kernel, m_range):
        im = im * appear.expand_as(im)
        pred = Variable(torch.Tensor(im.size()))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im.size(1)):
            im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
            for j in range(im.size(0)):
                pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return pred


class BiNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BiNet, self).__init__()
        num_hidden = 64
        self.bn = nn.BatchNorm2d(im_channel)
        self.conv0 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
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
        self.conv7_1 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn7_1 = nn.BatchNorm2d(num_hidden)
        self.conv7_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn7_2 = nn.BatchNorm2d(num_hidden)
        self.conv8_1 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8_1 = nn.BatchNorm2d(num_hidden)
        self.conv8_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn8_2 = nn.BatchNorm2d(num_hidden)
        self.conv9_1 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9_1 = nn.BatchNorm2d(num_hidden)
        self.conv9_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn9_2 = nn.BatchNorm2d(num_hidden)
        self.conv10_1 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10_1 = nn.BatchNorm2d(num_hidden)
        self.conv10_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn10_2 = nn.BatchNorm2d(num_hidden)
        self.conv11_1 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11_1 = nn.BatchNorm2d(num_hidden)
        self.conv11_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn11_2 = nn.BatchNorm2d(num_hidden)

        self.conv1 = nn.Conv2d(n_inputs*num_hidden, n_inputs*num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(n_inputs*num_hidden)
        self.conv2 = nn.Conv2d(n_inputs*num_hidden, n_inputs*num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(n_inputs*num_hidden)
        self.conv3 = nn.Conv2d(n_inputs*num_hidden, n_inputs*num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(n_inputs*num_hidden)
        self.conv4 = nn.Conv2d(n_inputs*num_hidden, n_inputs*num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(n_inputs*num_hidden)

        self.conv = nn.Conv2d(n_inputs*num_hidden, n_class, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_inputs = n_inputs
        self.n_class = n_class
        self.m_range = m_range
        self.m_kernel = m_kernel
        self.num_hidden = num_hidden

    def forward(self, im_input_f, im_input_b, ones):
        x_f = Variable(torch.zeros((im_input_f.size(0), self.n_inputs*self.num_hidden, self.im_height, self.im_width)))
        if torch.cuda.is_available():
            x_f = x_f.cuda()
        for i in range(self.n_inputs):
            x = self.bn0(self.conv0(self.bn(im_input_f[:, i*self.im_channel:(i+1)*self.im_channel, :, :])))
            x1 = F.relu(self.bn1_1(self.conv1_1(x)))
            x2 = self.maxpool(x1)
            x2 = F.relu(self.bn2_1(self.conv2_1(x2)))
            x3 = self.maxpool(x2)
            x3 = F.relu(self.bn3_1(self.conv3_1(x3)))
            x4 = self.maxpool(x3)
            x4 = F.relu(self.bn4_1(self.conv4_1(x4)))
            x5 = self.maxpool(x4)
            x5 = F.relu(self.bn5_1(self.conv5_1(x5)))
            x6 = self.maxpool(x5)
            x6 = F.relu(self.bn6_1(self.conv6_1(x6)))
            x6 = self.upsample(x6)
            x7 = torch.cat((x6, x5), 1)
            x7 = F.relu(self.bn7_1(self.conv7_1(x7)))
            x7 = self.upsample(x7)
            x8 = torch.cat((x7, x4), 1)
            x8 = F.relu(self.bn8_1(self.conv8_1(x8)))
            x8 = self.upsample(x8)
            x9 = torch.cat((x8, x3), 1)
            x9 = F.relu(self.bn9_1(self.conv9_1(x9)))
            x9 = self.upsample(x9)
            x10 = torch.cat((x9, x2), 1)
            x10 = F.relu(self.bn10_1(self.conv10_1(x10)))
            x10 = self.upsample(x10)
            x11 = torch.cat((x10, x1), 1)
            x11 = F.relu(self.bn11_1(self.conv11_1(x11)))
            x_f[:, i*self.num_hidden:(i+1)*self.num_hidden, :, :] = x11

        x_b = Variable(torch.zeros((im_input_b.size(0), self.n_inputs*self.num_hidden, self.im_height, self.im_width)))
        if torch.cuda.is_available():
            x_b = x_b.cuda()
        for i in range(self.n_inputs):
            x = self.bn0(self.conv0(self.bn(im_input_b[:, i*self.im_channel:(i+1)*self.im_channel, :, :])))
            x1 = F.relu(self.bn1_1(self.conv1_1(x)))
            x2 = self.maxpool(x1)
            x2 = F.relu(self.bn2_1(self.conv2_1(x2)))
            x3 = self.maxpool(x2)
            x3 = F.relu(self.bn3_1(self.conv3_1(x3)))
            x4 = self.maxpool(x3)
            x4 = F.relu(self.bn4_1(self.conv4_1(x4)))
            x5 = self.maxpool(x4)
            x5 = F.relu(self.bn5_1(self.conv5_1(x5)))
            x6 = self.maxpool(x5)
            x6 = F.relu(self.bn6_1(self.conv6_1(x6)))
            x6 = self.upsample(x6)
            x7 = torch.cat((x6, x5), 1)
            x7 = F.relu(self.bn7_1(self.conv7_1(x7)))
            x7 = self.upsample(x7)
            x8 = torch.cat((x7, x4), 1)
            x8 = F.relu(self.bn8_1(self.conv8_1(x8)))
            x8 = self.upsample(x8)
            x9 = torch.cat((x8, x3), 1)
            x9 = F.relu(self.bn9_1(self.conv9_1(x9)))
            x9 = self.upsample(x9)
            x10 = torch.cat((x9, x2), 1)
            x10 = F.relu(self.bn10_1(self.conv10_1(x10)))
            x10 = self.upsample(x10)
            x11 = torch.cat((x10, x1), 1)
            x11 = F.relu(self.bn11_1(self.conv11_1(x11)))
            x_b[:, i*self.num_hidden:(i+1)*self.num_hidden, :, :] = x11

        x = x_f
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        motion_f = self.conv(x)
        m_mask_f = F.softmax(motion_f)

        x = x_b
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        motion_b = self.conv(x)
        m_mask_b = F.softmax(motion_b)

        seg_f = self.construct_seg(ones, m_mask_f, self.m_kernel, self.m_range)
        seg_b = self.construct_seg(ones, m_mask_b, self.m_kernel, self.m_range)

        disappear_f = F.relu(seg_f - 1)
        appear_f = F.relu(1 - disappear_f)
        disappear_b = F.relu(seg_b - 1)
        appear_b = F.relu(1 - disappear_b)

        pred_f = self.construct_image(im_input_f[:, -self.im_channel:, :, :], m_mask_f, appear_f, self.m_kernel, self.m_range)
        pred_b = self.construct_image(im_input_b[:, -self.im_channel:, :, :], m_mask_b, appear_b, self.m_kernel, self.m_range)

        seg_f = 1 - F.relu(1 - seg_f)
        seg_b = 1 - F.relu(1 - seg_b)

        attn = (seg_f + 1e-5) / (seg_f + seg_b + 2e-5)
        pred = attn.expand_as(pred_f) * pred_f + (1 - attn.expand_as(pred_b)) * pred_b
        return pred, m_mask_f, 1 - appear_f, attn, m_mask_b, 1 - appear_b, 1 - attn

    def construct_seg(self, ones, m_mask, m_kernel, m_range):
        ones_expand = ones.expand_as(m_mask) * m_mask
        seg = Variable(torch.Tensor(ones.size()))
        if torch.cuda.is_available():
            seg = seg.cuda()
        for i in range(ones.size(0)):
            seg[i, :, :, :] = F.conv2d(ones_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return seg

    def construct_image(self, im, m_mask, appear, m_kernel, m_range):
        im = im * appear.expand_as(im)
        pred = Variable(torch.Tensor(im.size()))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im.size(1)):
            im_expand = im[:, i, :, :].unsqueeze(1).expand_as(m_mask) * m_mask
            for j in range(im.size(0)):
                pred[j, i, :, :] = F.conv2d(im_expand[j, :, :, :].unsqueeze(0), m_kernel, None, 1, m_range)
        return pred

