

def make_data_loader(args, size):
    import benzina.torch

    dataset = benzina.torch.ImageNet(args.data)

    return benzina.torch.DataLoader(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        shape=(size, size),
        warp_transform=benzina.torch.operations.SimilarityTransform(),
        norm_transform=1 / 255,
        bias_transform=-0.5
    )
