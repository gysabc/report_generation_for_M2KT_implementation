# # import pandas as pd
# #
# # # 创建一个没有标题行的DataFrame对象
# # data = [[1, 'Alice', 25],
# #         [2, 'Bob', 30],
# #         [3, 'Charlie', 35]]
# # df = pd.DataFrame(data, columns=None)
# #
# # # 遍历DataFrame中的每一行，并打印出每一行的数据
# # for index, row in df.iterrows():
# #     print(f"Index: {index}, ID: {row[0]}, Name: {row[1]}, Age: {row[2]}")
# #     print(row[1:].tolist())
# #     print(type(row))
# #     print("========")
# print((1,)+(2,3)+(4,))
# # print((1)+(2,3)+(4))
# print((1)+(2)+(4))
import torch
# import torchvision
# from tensorboardX import SummaryWriter
# from torchvision import datasets, transforms
#
# # Writer will output to ./runs/ directory by default
# writer = SummaryWriter("temp")
#
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# model = torchvision.models.resnet50(False)
# # Have ResNet model take in grayscale rather than RGB
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# images, labels = next(iter(trainloader))
#
# grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
# writer.close()
import torch

# 假设 image_1 和 image_2 是两个形状相同的图像张量
image_1 = torch.randn(3, 32, 32)
image_2 = torch.randn(3, 32, 32)

# 使用 torch.stack() 将两个图像沿着新的维度堆叠在一起
image = torch.stack((image_1, image_2), 0)

# 现在，image 是一个形状为 (6, 32, 32) 的张量，其中前三个元素是 image_1,后三个元素是 image_2
print(image.shape)
