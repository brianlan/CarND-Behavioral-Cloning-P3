import time
from math import ceil
from os.path import join as opj, dirname

import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torchvision.transforms as T

from dataset import ImageFolder
from network import Net
from settings import logger
from tensorboard_logger import Logger
from utils import mkdir_r


BASE_DIR = '/home/rlan/projects/self-driving-car/CarND-Behavioral-Cloning-P3'
INDICES_PATHS = [opj(BASE_DIR, 'training', 'driving_log.csv'),
                 opj(BASE_DIR, 'training2', 'driving_log.csv'),
                 opj(BASE_DIR, 'training3', 'driving_log.csv'),
                 opj(BASE_DIR, 'training4', 'driving_log.csv'),
                 opj(BASE_DIR, 'training5', 'driving_log.csv')]
CHECKPOINTS_PATH = '/home/rlan/projects/self-driving-car/CarND-Behavioral-Cloning-P3/checkpoints'
MAX_EPOCH = 30
BATCH_SIZE = 128


if __name__ == '__main__':
    cur_time = str(int(time.time()))
    tb_logger = Logger(opj('log', cur_time))
    transform = T.Compose([T.Resize(size=(160, 160)),
                           T.ToTensor(),
                           T.Lambda(lambda x: x - 0.5)])
    dataset = ImageFolder(INDICES_PATHS, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
    net = Net().cuda() if torch.cuda.is_available() else Net()
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss_fn = torch.nn.MSELoss()
    niter_per_epoch = ceil(len(dataset) / BATCH_SIZE)

    for epoch in range(MAX_EPOCH):
        lr_scheduler.step()
        for i_batch, sampled_batch in enumerate(loader):
            data, target = sampled_batch

            if torch.cuda.is_available():
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            pred = net(data)
            loss = loss_fn(pred, target.float())
            loss.backward()
            optimizer.step()
            logger.info('[epoch: {}, batch: {}] Training loss: {}'.format(epoch, i_batch, loss.data[0]))
            tb_logger.scalar_summary('loss', loss.data[0], epoch * niter_per_epoch + i_batch + 1)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            tb_logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
            tb_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch + 1)

        if (epoch + 1) % 10 == 0:
            cp_path = opj(CHECKPOINTS_PATH, cur_time, 'model_%s' % epoch)
            mkdir_r(dirname(cp_path))
            torch.save(net.state_dict(), cp_path)

