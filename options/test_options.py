from .base_options import BaseOptions
import copy


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

class FakeTestOptions():
    def __init__(self, opt):
        # 深拷贝opt
        self.opt = copy.deepcopy(opt)
        # hard-code some parameters for test
        self.opt.num_threads = 1   # test code only supports num_threads = 1
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.opt.isTrain = False
        self.opt.dataset_mode = 'unaligned'
        # self.opt.dataroot = '/home/ubuntu/merge/test_set'
        self.opt.dataroot = 'E:/PythonProjects/test-image-gen/test_set'
        self.opt.phase = 'test'
        self.opt.direction = 'AtoB'
        self.opt.max_dataset_size = 500
        self.opt.input_nc = 3
        self.opt.output_nc = 1