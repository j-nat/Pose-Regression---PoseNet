import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import netron

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock
        self.weights = weights
        self.key = key

        # 1x1 conv branch
        self.b1 = nn.Sequential(
                init(f'inception_{self.key}/1x1',nn.Conv2d(in_channels, n1x1, kernel_size=1, stride=1, padding=0), self.weights),
                nn.ReLU(True)
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
                init(f'inception_{self.key}/3x3_reduce',nn.Conv2d(in_channels,n3x3red, kernel_size=1,stride=1,padding=0), self.weights),
                nn.ReLU(True),
                init(f'inception_{self.key}/3x3',nn.Conv2d(n3x3red,n3x3,  kernel_size=3, stride=1, padding=1), self.weights),
                nn.ReLU(True)

        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
                init(f'inception_{self.key}/5x5_reduce',nn.Conv2d(in_channels,n5x5red, kernel_size=1,stride=1,padding=0), self.weights),
                nn.ReLU(True),
                init(f'inception_{self.key}/5x5',nn.Conv2d(n5x5red,n5x5, kernel_size=5, stride=1, padding=2), self.weights),
                nn.ReLU(True)

        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                nn.ReLU(True),
                init(f'inception_{self.key}/pool_proj',nn.Conv2d(in_channels,pool_planes,kernel_size=1, stride=1,padding=0), self.weights),
                nn.ReLU(True)
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        # print(b1.size(), b2.size(), b3.size(), b4.size())
        return torch.cat([b1, b2, b3, b4], 1)


class LossHeader(nn.Module):
    def __init__(self, in_channels, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers
        self.layer1 = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3, padding = 0),
            nn.ReLU(),
            init(f"{key}/conv", nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            init(f"{key}/fc", nn.Linear(2048, 1024), weights)
        )
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(1024,3)
        self.fc2 = nn.Linear(1024,4)


    def forward(self, x):
        # TODO: Feed data through loss headers
#
        out1 = self.layer1(x)
        out2 = torch.flatten(out1, 1)
        out3 = self.layer2(out2)

        out = self.dropout(out3)

        loss_xyz = self.fc1(out)
        loss_wpqr = self.fc2(out)

        return loss_xyz, loss_wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.LocalResponseNorm(size=5),

            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),

            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
         )


        # Example for InceptionBlock initialization
        self._3a = InceptionBlock( 192 , 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock( 256 , 128, 128, 192, 32, 96, 64, "3b", weights)

        self.mp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            )

        self._4a = InceptionBlock( 480 , 192, 96, 208, 16, 48, 64, "4a", weights)

        self._aux1 = LossHeader(512, "loss1", weights)

        self._4b = InceptionBlock( 512 , 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock( 512 , 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock( 512 , 112, 144, 288, 32, 64, 64, "4d", weights)

        self._aux2= LossHeader(528,"loss2", weights)

        self._4e = InceptionBlock( 528 , 256, 160, 320, 32, 128, 128, "4e", weights)

        self.mp4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self._5a = InceptionBlock( 832 , 256, 160, 320, 32, 128, 128,"5a", weights)
        self._5b = InceptionBlock( 832 , 384, 192, 384, 48, 128, 128, "5b", weights)

        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            #init("loss3/classifier", nn.Linear( 2048, 1024), weights)
            nn.Linear(1024, 2048)
        )

        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 3)
        self.fc3 = nn.Linear(2048, 4)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward

        out1 = self.pre_layers(x)
        out2 = self._3a(out1)
        out3 = self._3b(out2)

        out4 = self.mp3(out3)
        out5 = self._4a(out4)

        loss1_xyz, loss1_wpqr = self._aux1(out5)

        out6 = self._4b(out5)
        out7 = self._4c(out6)
        out8 = self._4d(out7)

        loss2_xyz, loss2_wpqr = self._aux2(out8)

        out9 = self._4e(out8)

        out10 = self.mp4(out9)
        out11 = self._5a(out10)
        out12 = self._5b(out11)

        out13 = self.avg_pool(out12)
        out14 = self.fc1(out13)
        out = self.dropout(out14)

        loss3_xyz = self.fc2(out)
        loss3_wpqr = self.fc3(out)



        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr

        t3 = torch.divide(poseGT[:,3:],torch.linalg.vector_norm(poseGT[:,3:],dim=1,ord=2).unsqueeze(-1))

        loss1 = torch.linalg.vector_norm((p1_xyz - poseGT[:,0:3]), dim=1, ord=2) + self.w1_wpqr * torch.linalg.vector_norm((p1_wpqr - t3), dim=1, ord=2)
        loss2 = torch.linalg.vector_norm((p2_xyz - poseGT[:,0:3]), dim=1, ord=2) + self.w2_wpqr * torch.linalg.vector_norm((p2_wpqr - t3), dim=1, ord=2)
        loss3 = torch.linalg.vector_norm((p3_xyz - poseGT[:,0:3]), dim=1, ord=2) + self.w3_wpqr * torch.linalg.vector_norm((p3_wpqr - t3), dim=1, ord=2)
        loss = loss1 * self.w1_xyz + loss2 * self.w2_xyz + loss3 * self.w3_xyz

        return loss.mean()