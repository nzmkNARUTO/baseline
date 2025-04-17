linear_mnist = [
    [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.8, 0.2, 0.8],
    [
        92.41,
        95.76,
        84.85,
        92.46,
        92.68,
        95.88,
        84.38,
        85.94,
        90.25,
        93.44,
        82.85,
        85.85,
        38.71,
        59.01,
        58.86,
        48.76,
    ],
]
print(f"linear_mnist: {sum(linear_mnist[0])/len(linear_mnist[0])}")
print(f"accuracy: {sum(linear_mnist[1])/len(linear_mnist[1])}")

linear_fashionmnist = [
    [0.2, 0.5, 0.5, 0.8, 0.2, 0.8, 0.8, 0.8, 0.2, 0.8, 0.5, 0.8, 0.2, 0.2, 0.2, 0.5],
    [
        79.42,
        88.95,
        77.32,
        84.01,
        80.88,
        88.93,
        77.16,
        84.00,
        75.47,
        84.48,
        65.85,
        80.94,
        54.79,
        66.28,
        58.48,
        78.15,
    ],
]
print(f"linear_fashionmnist: {sum(linear_fashionmnist[0])/len(linear_fashionmnist[0])}")
print(f"accuracy: {sum(linear_fashionmnist[1])/len(linear_fashionmnist[1])}")

linear_emnist = [
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.2],
    [
        79.20,
        81.27,
        79.44,
        84.71,
        60.84,
        57.38,
        17.32,
        17.15,
    ],
]
print(f"linear_emnist: {sum(linear_emnist[0])/len(linear_emnist[0])}")
print(f"accuracy: {sum(linear_emnist[1])/len(linear_emnist[1])}")

cnn_mnist = [
    [0.5, 0.3, 0.3, 0.5, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.3, 0.5, 0.8, 0.8, 0.8, 0.8],
    [
        98.93,
        99.36,
        89.12,
        99.43,
        98.72,
        99.21,
        88.97,
        99.37,
        99.09,
        99.48,
        89.15,
        99.56,
        98.78,
        99.26,
        89.25,
        99.31,
    ],
]
print(f"cnn_mnist: {sum(cnn_mnist[0])/len(cnn_mnist[0])}")
print(f"accuracy: {sum(cnn_mnist[1])/len(cnn_mnist[1])}")

cnn_fashionmnist = [
    [0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.2, 0.5, 0.2, 0.8, 0.8, 0.2, 0.8],
    [
        88.35,
        91.82,
        80.17,
        92.85,
        86.34,
        91.93,
        79.77,
        92.52,
        87.75,
        92.90,
        78.67,
        93.56,
        87.26,
        91.38,
        78.41,
        91.92,
    ],
]
print(f"cnn_fashionmnist: {sum(cnn_fashionmnist[0])/len(cnn_fashionmnist[0])}")
print(f"accuracy: {sum(cnn_fashionmnist[1])/len(cnn_fashionmnist[1])}")

cnn_emnist = [
    [0.5, 0.8, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8],
    [88.69, 90.29, 88.26, 90.15, 89.78, 90.99, 87.79, 90.22],
]
print(f"cnn_emnist: {sum(cnn_emnist[0])/len(cnn_emnist[0])}")
print(f"accuracy: {sum(cnn_emnist[1])/len(cnn_emnist[1])}")

cnn_cifar10 = [
    [0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.1, 0.3, 0.3, 0.1, 0.8, 0.8, 0.8, 0.8],
    [
        61.00,
        71.71,
        58.64,
        74.65,
        56.40,
        70.57,
        49.35,
        71.26,
        70.88,
        81.15,
        68.21,
        82.43,
        68.56,
        75.56,
        63.96,
        77.20,
    ],
]
print(f"cnn_cifar10: {sum(cnn_cifar10[0])/len(cnn_cifar10[0])}")
print(f"accuracy: {sum(cnn_cifar10[1])/len(cnn_cifar10[1])}")

cnn_cifar100 = [
    [0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5],
    [39.77, 42.07, 39.23, 38.93, 52.79, 56.12, 42.30, 44.35],
]
print(f"cnn_cifar100: {sum(cnn_cifar100[0])/len(cnn_cifar100[0])}")
print(f"accuracy: {sum(cnn_cifar100[1])/len(cnn_cifar100[1])}")

resnet_cifar10 = [
    [0.5, 0.7, 0.3, 0.5, 0.3, 0.7, 0.3, 0.3, 0.5, 0.3, 0.7, 0.3, 0.8, 0.5, 0.8, 0.5],
    [
        49.03,
        70.54,
        56.42,
        73.68,
        48.91,
        71.20,
        56.48,
        73.23,
        46.34,
        57.85,
        54.93,
        69.65,
        54.38,
        70.43,
        53.04,
        73.53,
    ],
]
print(f"resnet_cifar10: {sum(resnet_cifar10[0])/len(resnet_cifar10[0])}")
print(f"accuracy: {sum(resnet_cifar10[1])/len(resnet_cifar10[1])}")

resnet_cifar100 = [
    [0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.5, 0.2],
    [36.86, 40.93, 38.60, 43.17, 39.16, 42.92, 45.67, 50.37],
]
print(f"resnet_cifar100: {sum(resnet_cifar100[0])/len(resnet_cifar100[0])}")
print(f"accuracy: {sum(resnet_cifar100[1])/len(resnet_cifar100[1])}")
