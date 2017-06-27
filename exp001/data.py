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


def get_meta(args, image_dir):
    meta = {}
    cnt = 0
    sub_dirs = os.listdir(image_dir)
    for sub_dir in sub_dirs:
        sub_sub_dirs = os.listdir(os.path.join(image_dir, sub_dir))
        for sub_sub_dir in sub_sub_dirs:
            image_files = os.listdir(os.path.join(image_dir, sub_dir, sub_sub_dir))
            image_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            image_names = [os.path.join(image_dir, sub_dir, sub_sub_dir, f) for f in image_files]
            image_names = image_names[::2]
            num_images = len(image_names)
            if num_images < args.num_inputs + 1:
                continue
            idx = range(0, num_images - args.num_inputs - 1)
            for i in range(len(idx)):
                start_idx = idx[i]
                end_idx = idx[i] + args.num_inputs + 1
                meta[cnt] = image_names[start_idx:end_idx]
                cnt += 1
    return meta


def generate_batch(args, meta):
    batch_size, n_inputs, height, width, im_channel = args.batch_size, args.num_inputs, args.image_size, args.image_size, args.num_channel
    idx = numpy.random.permutation(len(meta))[0:batch_size]
    im_input = numpy.zeros((batch_size, im_channel * n_inputs, height, width))
    im_output = numpy.zeros((batch_size, im_channel, height, width))
    for i in range(batch_size):
        image_names = meta[idx[i]]
        n_image = len(image_names)
        assert n_image == n_inputs + 1
        for j in range(n_image):
            image_name = image_names[j]
            if args.num_channel == 1:
                im = numpy.array(Image.open(image_name).convert('L')) / 255.0
            else:
                im = numpy.array(Image.open(image_name)) / 255.0
            if args.num_channel == 1:
                im = numpy.expand_dims(im, 0)
            else:
                im = im.transpose((2, 0, 1))
            if j == 0:
                _, im_height, im_width = im.shape
                idx_h = (im_height - height) / 2
                idx_w = (im_width - width) / 2
            if j < n_inputs:
                im_input[i, j*im_channel:(j+1)*im_channel, :, :] = im[:, idx_h:idx_h+height, idx_w:idx_w+width]
            else:
                im_output[i, :, :, :] = im[:, idx_h:idx_h+height, idx_w:idx_w+width]
    return im_input, im_output


def generate_images(args, m_dict, reverse_m_dict):
    im_size, m_range, batch_size, channel = args.image_size, args.motion_range, args.batch_size, args.num_channel
    im1 = generate_box(batch_size, channel, im_size)
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    m_x = numpy.zeros(batch_size).astype(int)
    m_y = numpy.zeros(batch_size).astype(int)
    for i in range(batch_size):
        (m_x[i], m_y[i]) = reverse_m_dict[m_label[i]]
        gt_motion[i, :, :, :] = m_label[i]
    im2 = move_image(im1, m_x, m_y)
    im3 = move_image(im2, m_x, m_y)
    gt_motion[numpy.expand_dims(im2.sum(1), 1) == 0] = m_dict[(0, 0)]
    im_input = numpy.concatenate((im1, im2), 1)
    im_output = im3
    return im_input, im_output, gt_motion.astype(int)


def generate_box(batch_size, channel, im_size):
    im = numpy.zeros((batch_size, channel, im_size, im_size))
    for i in range(batch_size):
        width = numpy.random.randint(im_size/4, im_size*3/4)
        height = numpy.random.randint(im_size/4, im_size*3/4)
        x = numpy.random.randint(0, im_size - width)
        y = numpy.random.randint(0, im_size - height)
        color = numpy.random.uniform(0.5, 1, channel)
        for j in range(channel):
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


def display(images1, images2, images3):
    for i in range(images1.shape[0]):
        plt.figure(1)
        plt.subplot(2, 3, 1)
        if images1.shape[1] == 1:
            im1 = images1[i, :, :, :].squeeze()
            plt.imshow(im1, cmap='gray')
        else:
            im1 = images1[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im1)
        plt.subplot(2, 3, 2)
        if images2.shape[1] == 1:
            im2 = images2[i, :, :, :].squeeze()
            plt.imshow(im2, cmap='gray')
        else:
            im2 = images2[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im2)
        plt.subplot(2, 3, 3)
        if images3.shape[1] == 1:
            im3 = images3[i, :, :, :].squeeze()
            plt.imshow(im3, cmap='gray')
        else:
            im3 = images3[i, :, :, :].squeeze().transpose(1, 2, 0)
            plt.imshow(im3)
        plt.subplot(2, 3, 4)
        im_diff1 = abs(im2 - im1)
        plt.imshow(im_diff1)
        plt.subplot(2, 3, 5)
        im_diff2 = abs(im3 - im2)
        plt.imshow(im_diff2)
        plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    im_input, im_output = generate_images(args, m_dict, reverse_m_dict)
    if True:
        im1 = im_input[:, -args.num_channel*2:-args.num_channel, :, :]
        im2 = im_input[:, -args.num_channel:, :, :]
        im3 = im_output
        print im1.shape, im2.shape, im3.shape
        display(im1, im2, im3)

if __name__ == '__main__':
    unit_test()

