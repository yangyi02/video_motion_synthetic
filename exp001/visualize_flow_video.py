import os
import sys
import numpy
from PIL import Image
from learning_args import parse_args
import flowlib
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def create_video(args):
    image_dir, flow_dir, flow_video_dir = args.test_dir, args.flow_dir, args.flow_video_dir
    if not os.path.exists(flow_video_dir):
        os.mkdir(flow_video_dir)
    for sub_dir in os.listdir(flow_dir):
        if not os.path.exists(os.path.join(flow_video_dir, sub_dir)):
            os.mkdir(os.path.join(flow_video_dir, sub_dir))
        for sub_sub_dir in os.listdir(os.path.join(flow_dir, sub_dir)):
            if not os.path.exists(os.path.join(flow_video_dir, sub_dir, sub_sub_dir)):
                os.mkdir(os.path.join(flow_video_dir, sub_dir, sub_sub_dir))
            image_files = os.listdir(os.path.join(image_dir, sub_dir, sub_sub_dir))
            image_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            flow_video_files = image_files
            image_files = [os.path.join(image_dir, sub_dir, sub_sub_dir, f) for f in image_files]
            flow_files = os.listdir(os.path.join(flow_dir, sub_dir, sub_sub_dir))
            flow_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            flow_files = [os.path.join(flow_dir, sub_dir, sub_sub_dir, f) for f in flow_files]
            flow_video_files = [os.path.join(flow_video_dir, sub_dir, sub_sub_dir, f) for f in flow_video_files]
            create_one_video(args, image_files, flow_files, flow_video_files)


def create_one_video(args, image_files, flow_files, flow_video_files):
    m_range = args.motion_range
    start_idx = args.num_inputs
    for i in range(len(image_files)):
        if i < start_idx:
            im = numpy.array(Image.open(image_files[i]))
            flow = numpy.zeros((im.shape[0], im.shape[1], 2))
        else:
            logging.info('%s, %s' % (image_files[i], flow_files[i-start_idx]))
            im = numpy.array(Image.open(image_files[i]))
            flow = flowlib.read_flow(flow_files[i-start_idx])

        im_width, im_height = im.shape[1], im.shape[0]
        width, height = get_img_size(1, 2, im_width, im_height)
        img = numpy.ones((height, width, 3))

        x1, y1, x2, y2 = get_img_coordinate(1, 1, im_width, im_height)
        img[y1:y2, x1:x2, :] = im / 255.0

        optical_flow = flowlib.visualize_flow(flow, m_range)
        x1, y1, x2, y2 = get_img_coordinate(1, 2, im_width, im_height)
        img[y1:y2, x1:x2, :] = optical_flow / 255.0

        img = img * 255.0
        img = img.astype(numpy.uint8)
        img = Image.fromarray(img)
        img.save(flow_video_files[i])


def get_img_size(n_row, n_col, im_width, im_height):
    height = n_row * im_height + (n_row - 1) * int(im_height/10)
    width = n_col * im_width + (n_col - 1) * int(im_width/10)
    return width, height


def get_img_coordinate(row, col, im_width, im_height):
    y1 = (row - 1) * im_height + (row - 1) * int(im_height/10)
    y2 = y1 + im_height
    x1 = (col - 1) * im_width + (col - 1) * int(im_width/10)
    x2 = x1 + im_width
    return x1, y1, x2, y2


def main():
    args = parse_args()
    logging.info(args)
    create_video(args)

if __name__ == '__main__':
    main()
