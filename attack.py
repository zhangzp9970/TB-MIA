import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydl import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import *
from tqdm import tqdm

data_workers = 4
batch_size = 64
min_step = 5*30596
log_interval = 100
test_interval = 100
classes = 10
root_dir = "./log"
train_file = "/seu_share/home/huangjie/hj_zhangzp/source/MNISTItrain.txt"
test_file = "/seu_share/home/huangjie/hj_zhangzp/source/MNISTCtrain.txt"
test_pkl = "/seu_share/home/huangjie/hj_zhangzp/source/log/Jul28_21-15-01/best.pkl"

cudnn.benchmark = True
cudnn.deterministic = True
seed = 9970
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f'{root_dir}/{now}'
logger = SummaryWriter(log_dir)

with open(os.path.join(log_dir, os.path.basename(__file__)), 'w', encoding='utf-8') as fout:
    with open(os.path.abspath(__file__), 'r', encoding='utf-8') as fin:
        file = fin.read()
        fout.write(file)

transform = Compose([
    Grayscale(num_output_channels=1),
    Resize((32, 32)),
    ToTensor()
])

train_ds = FileListDataset(list_path=train_file, transform=transform)
test_ds = FileListDataset(list_path=test_file, transform=transform)

train_dl = DataLoader(dataset=train_ds, batch_size=batch_size,
                      shuffle=True, num_workers=data_workers, drop_last=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size,
                     shuffle=False, num_workers=data_workers, drop_last=False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(
            num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2,
                               padding=0, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(
            num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(
            num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(in_features=8192, out_features=50, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.mp(x)
        x = self.relu(x)
        x = x.view(-1, 8192)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Inversion(nn.Module):
    def __init__(self):
        super(Inversion, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=10, out_channels=512, kernel_size=(4, 4), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(
            num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(
            num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(
            num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 10, 1, 1)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.tanh(x)
        x = self.deconv4(x)
        x = self.sigmod(x)
        return x


myclassifier = nn.DataParallel(Classifier()).train(False).to(output_device)
myinversion = nn.DataParallel(Inversion()).train(True).to(output_device)
optimizer = optim.Adam(myinversion.parameters(), lr=0.0002,
                       betas=(0.5, 0.999), amsgrad=True)

assert os.path.exists(test_pkl)
data = torch.load(open(test_pkl, 'rb'))
myclassifier.load_state_dict(data['myclassifier'])

global_step = 0
best_loss = 997
total_steps = tqdm(range(min_step), desc='global step')
epoch_id = 0

while global_step < min_step:
    iters = tqdm(train_dl, desc=f'epoch {epoch_id} ', total=len(train_dl))
    epoch_id += 1
    for i, (im, label) in enumerate(iters):
        im = im.to(output_device)
        label = label.to(output_device)
        with torch.no_grad():
            out = myclassifier.forward(im)
            after_softmax = F.softmax(out, dim=-1)
            after_softmax = torch.log(after_softmax)+50
        rim = myinversion.forward(after_softmax)
        mse = F.mse_loss(rim, im)
        with OptimizerManager([optimizer]):
            loss = mse
            loss.backward()
        global_step += 1
        total_steps.update()
        if global_step % log_interval == 0:
            logger.add_scalar('loss', loss, global_step)
            logger.add_image(f'train inverted image {label[0]}',
                             rim[0], label[0])
        if global_step % test_interval == 0:
            mseloss = 0
            with TrainingModeManager([myinversion], train=False) as mgr, \
                    torch.no_grad():
                for i, (im, label) in enumerate(tqdm(test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    out = myclassifier.forward(im)
                    after_softmax = F.softmax(out, dim=-1)
                    rim = myinversion.forward(after_softmax)
                    mse = F.mse_loss(rim, im, reduction='sum')
                    mseloss += mse.item()
                    logger.add_image(
                        f'test inverted image {label[0]}', rim[0], i)
            mseloss /= len(test_dl.dataset)*32*32
            logger.add_scalar('mseloss', mseloss, global_step)
            clear_output()
            data = {
                "myinversion": myinversion.state_dict(),
            }
            if mseloss < best_loss:
                best_loss = mseloss
                with open(os.path.join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

                with open(os.path.join(log_dir, 'current.pkl'), 'wb') as f:
                    torch.save(data, f)
logger.close()
