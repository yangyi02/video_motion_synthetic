import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import pickle
from PIL import Image

import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def motion_dict(m_range):
    m_dict, reverse_m_dict = {}, {}
    x = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
    y = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
    m_x, m_y = numpy.meshgrid(x, y)
    m_x, m_y, = m_x.reshape(-1).astype(int), m_y.reshape(-1).astype(int)
    m_kernel = numpy.zeros((1, len(m_x), 2 * m_range + 1, 2 * m_range + 1))
    for i in range(len(m_x)):
        m_dict[(m_x[i], m_y[i])] = i
        reverse_m_dict[i] = (m_x[i], m_y[i])
        m_kernel[:, i, m_y[i] + m_range, m_x[i] + m_range] = 1
    return m_dict, reverse_m_dict, m_kernel


def generate_images(args, m_dict, reverse_m_dict):
    noise = 0.5
    im_size, m_range, batch_size, im_channel = args.image_size, args.motion_range, args.batch_size, args.num_channel
    im3 = generate_box(batch_size, im_channel, im_size)
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    m_f_x = numpy.zeros((batch_size)).astype(int)
    m_f_y = numpy.zeros((batch_size)).astype(int)
    m_b_x = numpy.zeros((batch_size)).astype(int)
    m_b_y = numpy.zeros((batch_size)).astype(int)
    for i in range(batch_size):
        (m_f_x[i], m_f_y[i]) = reverse_m_dict[m_label[i]]
        (m_b_x[i], m_b_y[i]) = (-m_f_x[i], -m_f_y[i])
    gt_motion_f = numpy.zeros((batch_size, 1, im_size, im_size))
    gt_motion_b = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        gt_motion_f[i, :, :, :] = m_label[i]
        gt_motion_b[i, :, :, :] = m_dict[(m_b_x[i], m_b_y[i])]
    bg = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise
    im2 = move_image(im3, m_b_x, m_b_y)
    im1 = move_image(im2, m_b_x, m_b_y)
    im4 = move_image(im3, m_f_x, m_f_y)
    im5 = move_image(im4, m_f_x, m_f_y)
    gt_motion_f[numpy.expand_dims(im2.sum(1), 1) == 0] = m_dict[(0, 0)]
    gt_motion_b[numpy.expand_dims(im4.sum(1), 1) == 0] = m_dict[(0, 0)]
    im1[im1 == 0] = bg[im1 == 0]
    im2[im2 == 0] = bg[im2 == 0]
    im3[im3 == 0] = bg[im3 == 0]
    im4[im4 == 0] = bg[im4 == 0]
    im5[im5 == 0] = bg[im5 == 0]
    im_input_f = numpy.concatenate((im1, im2), 1)
    im_input_b = numpy.concatenate((im5, im4), 1)
    im_output = im3
    return im_input_f, im_input_b, im_output, gt_motion_f.astype(int), gt_motion_b.astype(int)


def generate_box(batch_size, im_channel, im_size):
    noise = 0.5
    im = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        width = numpy.random.randint(im_size/8, im_size*3/4)
        height = numpy.random.randint(im_size/8, im_size*3/4)
        x = numpy.random.randint(0, im_size - width)
        y = numpy.random.randint(0, im_size - height)
        color = numpy.random.uniform(noise, 1, im_channel)
        for j in range(im_channel):
            im[i, j, y:y+height, x:x+width] = color[j]
    return im


def move_image(im, m_x, m_y):
    [batch_size, im_channel, _, im_size] = im.shape
    m_range_x = numpy.max(numpy.abs(m_x).reshape(-1))
    m_range_y = numpy.max(numpy.abs(m_y).reshape(-1))
    m_range = max(m_range_x, m_range_y).astype(int)
    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im
    im_new = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        im_new[i, :, :, :] = im_big[i, :, m_range + m_y[i]:m_range + m_y[i] + im_size,
                             m_range + m_x[i]:m_range + m_x[i] + im_size]
    return im_new


def display(images1, images2, images3, images4, images5):
    for i in range(images1.shape[0]):
        plt.figure(1)
        plt.subplot(2, 5, 1)
        if images1.shape[1] == 1:
            im1 = images1[i, :, :, :].squeeze()
            plt.imshow(im1, cmap='gray')
        else:
            im1 = images1[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im1)
        plt.subplot(2, 5, 2)
        if images2.shape[1] == 1:
            im2 = images2[i, :, :, :].squeeze()
            plt.imshow(im2, cmap='gray')
        else:
            im2 = images2[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im2)
        plt.subplot(2, 5, 3)
        if images3.shape[1] == 1:
            im3 = images3[i, :, :, :].squeeze()
            plt.imshow(im3, cmap='gray')
        else:
            im3 = images3[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im3)
        plt.subplot(2, 5, 4)
        if images4.shape[1] == 1:
            im4 = images4[i, :, :, :].squeeze()
            plt.imshow(im4, cmap='gray')
        else:
            im4 = images4[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im4)
        plt.subplot(2, 5, 5)
        if images5.shape[1] == 1:
            im5 = images5[i, :, :, :].squeeze()
            plt.imshow(im5, cmap='gray')
        else:
            im5 = images5[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im5)
        plt.subplot(2, 5, 6)
        im_diff1 = abs(im2 - im1)
        plt.imshow(im_diff1)
        plt.subplot(2, 5, 7)
        im_diff2 = abs(im3 - im2)
        plt.imshow(im_diff2)
        plt.subplot(2, 5, 9)
        im_diff3 = abs(im4 - im3)
        plt.imshow(im_diff3)
        plt.subplot(2, 5, 10)
        im_diff4 = abs(im5 - im4)
        plt.imshow(im_diff4)
        plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    im_input_f, im_input_b, im_output, gt_motion_f, gt_motion_b = generate_images(args, m_dict, reverse_m_dict)
    if True:
        im1 = im_input_f[:, -args.num_channel*2:-args.num_channel, :, :]
        im2 = im_input_f[:, -args.num_channel:, :, :]
        im3 = im_output
        im4 = im_input_b[:, -args.num_channel:, :, :]
        im5 = im_input_b[:, -args.num_channel*2:-args.num_channel, :, :]
        print im1.shape, im2.shape, im3.shape, im4.shape, im5.shape
        display(im1, im2, im3, im4, im5)

if __name__ == '__main__':
    unit_test()

