import os
import numpy as np

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Unet
from dataset import *
from util import *

from torchvision import transforms, datasets
import matplotlib.pyplot as plt

## Parser 생성
parser = argparse.ArgumentParser(description="Train the Unet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest='lr')
parser.add_argument("--batch_size", default=4, type=int, dest='batch_size')
parser.add_argument("--num_epoch", default=100, type=int, dest='num_epoch')

parser.add_argument("--data_dir", default='./datasets/BSR/BSDS500/data/images', type=str, dest='data_dir')
parser.add_argument("--log_dir", default='./log', type=str, dest='log_dir')
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument("--result_dir", default='./results', type=str, dest='result_dir')

parser.add_argument("--mode", default='train', type=str, dest='mode')
parser.add_argument("--train_continue", default= 'off', type=str, dest='train_continue')

parser.add_argument("--task", default="denoising", choices=["denoising", "inpainting", "super_resolution"], type=str,
                    dest="task")
parser.add_argument("--opts", nargs="+", default=["random", 30.0], dest="opts")

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")
parser.add_argument("--learning_type", default="plain", choices=['plain', 'residual'],type=str, dest="learning_type")

parser.add_argument("--network", default="unet", choices=["unet", "resnet", "autoencoder"],type=str,dest="network")


args = parser.parse_args()

##parameter
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker
learning_type = args.learning_type

network = args.network

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("mode: %s" % mode)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("task: %s" % task)
print("opts: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

##디렉토리 생성
result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')


if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))


#네트워크 학습
if mode == 'train':
    transform_train = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir,'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=False, num_workers=8)

    #부수적인 variable
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train/ batch_size)
    num_batch_val = np.ceil(num_data_val/batch_size)

else:
    # 네트워크 학습
    transform_test = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    # 부수적인 variable
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

#네트워크, 파라미터 생성
if network == "unet":
    net = Unet(nch=nch, nker=nker, norm="bnorm", learning_type=learning_type).to(device)
# elif network == "resnet":
#     net = Resnet().to(device)

# fn_loss = nn.BCEWithLogitsLoss().to(device) #for segmentation
fn_loss = nn.MSELoss().to(device) #for regression

optim = torch.optim.Adam(net.parameters(), lr = lr)

#부수적인 function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x*std) +mean
# fn_class = lambda x: 1.0 * (x > 0.5) #not used in regression

# Tensorboard를 위한 SummaryWriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

#네트워크 저장, 불러오기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim, epoch

#네트워크 학습
st_epoch = 0

if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch+1, num_epoch+1):
        net.train()
        loss_arr=[]

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            #Tensorboard 저장
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train*(epoch - 1) +batch

            plt.imsave(os.path.join(result_dir_train, 'png', "%04d_label.png" % id), label[0])
            plt.imsave(os.path.join(result_dir_train, 'png', "%04d_input.png" % id), input[0])
            plt.imsave(os.path.join(result_dir_train, 'png', "%04d_output.png" % id), output[0])


            # writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr=[]

            for batch, data in enumerate(loader_val,1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_val * (epoch - 1) + batch

                plt.imsave(os.path.join(result_dir_val, 'png', "%04d_label.png" % id), label[0])
                plt.imsave(os.path.join(result_dir_val, 'png', "%04d_input.png" % id), input[0])
                plt.imsave(os.path.join(result_dir_val, 'png', "%04d_output.png" % id), output[0])

                # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

            if epoch % 50 ==0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
    writer_train.close()
    writer_val.close()

else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("Test: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                np.save(os.path.join(result_dir_test, 'numpy', 'label_%04d.npy' % id), label_)
                np.save(os.path.join(result_dir_test, 'numpy', 'input_%04d.npy' % id), input_)
                np.save(os.path.join(result_dir_test, 'numpy', 'output_%04d.npy' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', 'label_%04d.png' % id), label_)
                plt.imsave(os.path.join(result_dir_test, 'png', 'input_%04d.png' % id), input_)
                plt.imsave(os.path.join(result_dir_test, 'png', 'output_%04d.png' % id), output_)

    print("Average Test: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))

