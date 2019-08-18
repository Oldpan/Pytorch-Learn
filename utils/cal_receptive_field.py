vgg_16 = [
    # 1
    [3, 1], [3, 1], [2, 2],
    # 2
    [3, 1], [3, 1], [2, 2],
    # 3
    [3, 1], [3, 1], [3, 1], [2, 2],
    # 4
    [3, 1], [3, 1], [3, 1], [2, 2],
    # 5
    [3, 1], [3, 1], [3, 1], [2, 2],
    # fc6, fake convolutional layer
    [7, 1]
]
vgg16_layers = [
    "3x3 conv 64", "3x3 conv 64", "pool1",
    "3x3 conv 128", "3x3 conv 128", "pool2",
    "3x3 conv 256", "3x3 conv 256", "3x3 conv 256", "pool3",
    "3x3 conv 512", "3x3 conv 512", "3x3 conv 512", "pool4",
    "3x3 conv 512", "3x3 conv 512", "3x3 conv 512", "pool5",
    "7x7 fc"
]


def cal_receptive_field(kspairs, layers=None):
    # K: composed kernel, also the receptive field
    # S: composed stride
    K, S = 1, 1
    # H = 224
    if not layers:
        layers = range(len(kspairs))
    for layer, kspair in zip(layers, kspairs):
        k, s = kspair
        K = (k - 1) * S + K
        S = S * s
        # H = H//s
        # iamge size {0}'.format(H)

        print('layer {:<15}: {} [{:3},{:2}]'.format(layer, kspair, K, S))


cal_receptive_field(vgg_16, vgg16_layers)