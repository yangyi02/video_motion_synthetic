import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='unsupervised motion', add_help=False)
    arg_parser.add_argument('--train', action='store_true')
    arg_parser.add_argument('--test', action='store_true')
    arg_parser.add_argument('--method', default='unsupervised')
    arg_parser.add_argument('--train_epoch', type=int, default=1000)
    arg_parser.add_argument('--test_epoch', type=int, default=100)
    arg_parser.add_argument('--test_interval', type=int, default=500)
    arg_parser.add_argument('--display', action='store_true')
    arg_parser.add_argument('--save_dir', default='./model')
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--init_model_path', default='')
    arg_parser.add_argument('--learning_rate', type=float, default=0.01)
    arg_parser.add_argument('--motion_range', type=int, default=1)
    arg_parser.add_argument('--image_size', type=int, default=64)
    arg_parser.add_argument('--num_channel', type=int, default=3)
    arg_parser.add_argument('--num_inputs', type=int, default=2)
    arg_parser.add_argument('--train_dir', default='train_images')
    arg_parser.add_argument('--test_dir', default='test_images')
    arg_parser.add_argument('--test_video', action='store_true')
    arg_parser.add_argument('--flow_dir', default='flow')
    arg_parser.add_argument('--flow_video_dir', default='flow-video')
    args = arg_parser.parse_args()
    return args

