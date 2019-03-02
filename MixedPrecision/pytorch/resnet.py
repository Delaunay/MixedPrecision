"""
    Legacy

"""
from MixedPrecision.pytorch.convnet import generic_main as generic_main_conv


def generic_main(name_override):
    from MixedPrecision.tools.args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    args.model = name_override
    return generic_main_conv(args)


def resnet18_main():
    return generic_main('resnet18')


def resnet50_main():
    return generic_main('resnet50')


def resnet34_main():
    return generic_main('resnet34')


def resnet101_main():
    return generic_main('resnet101')


def densenet161_main():
    return generic_main('densenet161')
