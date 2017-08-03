import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BRATS Pipeline')

    parser.add_argument('gpu', type=str)

    args = parser.parse_args(sys.argv[2:])
    print 'Running Deep Learning Pipeline on gpu... %s' % args.gpu

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import train

    train.pipeline()