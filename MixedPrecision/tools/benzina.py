

def make_data_loader(args):
    import benzina.torch

    dataset = benzina.torch.ImageNet(args.data + '/train')

    return benzina.torch.NvdecodeDataLoader(
        dataset,
        batch_size=args.batch_size,
        seed=0,
        shape=(256, 256),
        warp_transform=None,
        oob_transform=(0, 0, 0),
        scale_transform=1 / 255,
        bias_transform=-0.5
    )
