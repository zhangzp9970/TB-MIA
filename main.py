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
train_file = "/seu_share/home/huangjie/hj_zhangzp/source/MNISTCtrain.txt"
test_file = "/seu_share/home/huangjie/hj_zhangzp/source/MNISTCtest.txt"

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


myclassifier = nn.DataParallel(Classifier()).to(output_device)
optimizer = optim.Adam(myclassifier.parameters(), lr=0.0002,
                       betas=(0.5, 0.999), amsgrad=True)

global_step = 0
best_acc = 0
total_steps = tqdm(range(min_step), desc='global step')
epoch_id = 0

while global_step < min_step:
    iters = tqdm(train_dl, desc=f'epoch {epoch_id} ', total=len(train_dl))
    epoch_id += 1
    for i, (im, label) in enumerate(iters):
        im = im.to(output_device)
        label = label.to(output_device)
        out = myclassifier.forward(im)
        ce = nn.CrossEntropyLoss()(out, label)
        with OptimizerManager([optimizer]):
            loss = ce
            loss.backward()
        global_step += 1
        total_steps.update()
        if global_step % log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(
                one_hot(label, classes)), variable_to_numpy(F.softmax(out, dim=-1)))
            acc_train = torch.tensor(
                [counter.reportAccuracy()]).to(output_device)
            logger.add_scalar('loss', loss, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)
        if global_step % test_interval == 0:
            counters = [AccuracyCounter() for x in range(classes)]
            with TrainingModeManager([myclassifier], train=False) as mgr, \
                    Accumulator(['after_softmax', 'label']) as target_accumulator, \
                    torch.no_grad():
                for i, (im, label) in enumerate(tqdm(test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    out = myclassifier.forward(im)
                    after_softmax = F.softmax(out, dim=-1)
                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])
                    target_accumulator.updateData(globals())
            for x in target_accumulator:
                globals()[x] = target_accumulator[x]
            counters = [AccuracyCounter() for x in range(classes)]
            for (each_predict_prob, each_label) in zip(after_softmax, label):
                counters[each_label].Ntotal += 1.0
                each_pred_id = np.argmax(each_predict_prob)
                if each_pred_id == each_label:
                    counters[each_label].Ncorrect += 1.0
            acc_tests = [x.reportAccuracy()
                         for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)
            logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()
            data = {
                "myclassifier": myclassifier.state_dict(),
            }
            if acc_test > best_acc:
                best_acc = acc_test
                with open(os.path.join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

                with open(os.path.join(log_dir, 'current.pkl'), 'wb') as f:
                    torch.save(data, f)
logger.close()
