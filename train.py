import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torchvision.transforms as T

from dataset import ImageFolder
from network import Net
from settings import logger


DATA_DIR = '/home/rlan/projects/self-driving-car-engineer/CarND-Behavioral-Cloning-P3/training_1/IMG'
INDICES_PATH = '/home/rlan/projects/self-driving-car-engineer/CarND-Behavioral-Cloning-P3/training_1/driving_log.csv'
MODEL_SAVE_PATH = '/home/rlan/projects/self-driving-car-engineer/CarND-Behavioral-Cloning-P3/training_1/model_result'
MAX_EPOCH = 10


if __name__ == '__main__':
    transform = T.Compose([T.ToTensor()])
    dataset = ImageFolder(DATA_DIR, INDICES_PATH, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=4)
    net = Net().cuda() if torch.cuda.is_available() else Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(MAX_EPOCH):
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

    torch.save(net.state_dict(), MODEL_SAVE_PATH)
