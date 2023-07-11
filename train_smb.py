# train an SDB network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import pandas as pd

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader, generate_noise

# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='experiments/CIFAR10/adversarial_teacher/resnet18_self', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])

import os
print('GPU id {}'.format(device_ids))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])



# ************************** training function **************************
def train_epoch_kd_adv(model, model_ad, model_rad, optim, data_loader, epoch, params, noise):
    model.train()
    model_ad.eval()
    model_rad.eval()
    tch_loss_avg = RunningAverage()
    ad_loss1_avg = RunningAverage()
    ad_loss2_avg = RunningAverage()
    rad_loss_avg = RunningAverage()
    loss_avg = RunningAverage()


    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if params.cuda:
                train_batch = train_batch.cuda()  # (B,3,32,32)
                labels_batch = labels_batch.cuda()  # (B,)

            noisy_train_batch = params.lamb * train_batch + \
                                (1-params.lamb) * noise.unsqueeze(0).repeat(train_batch.size()[0],1,1,1)

            # compute (teacher) model output and loss
            output_tch = model(train_batch)  # logit without SoftMax
            output_tch_noisy = model(noisy_train_batch)

            # teacher loss: CE(output_tch, label)
            tch_loss = 2*nn.CrossEntropyLoss()(output_tch, labels_batch) + nn.CrossEntropyLoss()(output_tch_noisy, labels_batch)

            # ############ adversarial loss ####################################
            # computer pre-trained model output
            with torch.no_grad():
                output_stu = model_ad(train_batch)  # logit without SoftMax
            output_stu = output_stu.detach()

            # knowledge disturbance loss
            T = params.temperature
            adv_loss1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_stu/ T, dim=1),
                                      F.softmax(output_tch/ T, dim=1)) * (T * T)-nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_tch_noisy.detach(), dim=1),
                                      F.softmax(output_tch, dim=1))    # wish to max this item
            # Maintain loss      
            #adv_loss2 = nn.MSELoss(reduction='batchmean')(F.relu(output_stu), F.relu(output_tch_noisy))   # wish to max this item
            adv_loss2 =  nn.MSELoss(reduction='batchmean')(F.softmax(output_stu/ T, dim=1),
                                     F.softmax(output_tch_noisy / T, dim=1))
            #adv_loss2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_tch_noisy/ T, dim=1),
            #                         F.softmax(output_stu / T, dim=1)) * (T * T) + nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_stu/ T, dim=1),
            #                         F.softmax(output_tch_noisy / T, dim=1)) * (T * T)
            # ############ random loss ####################################
            # computer random model output
            with torch.no_grad():
                output_rad = model_rad(train_batch)  # logit without SoftMax
            output_rad = output_rad.detach()

            random_loss = 0.01 * nn.MSELoss(reduction='batchmean')(F.relu(output_stu), F.relu(output_tch_noisy))-nn.MSELoss(reduction='batchmean')(F.relu(output_rad), F.relu(output_tch_noisy))   # wish to max this item
            # total loss
            loss = tch_loss - params.weight * (adv_loss1) + adv_loss2 - params.eta * random_loss

            # ############################################################

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())
            tch_loss_avg.update(tch_loss.item())
            ad_loss1_avg.update(adv_loss1.item())
            ad_loss2_avg.update(adv_loss2.item())
            rad_loss_avg.update(random_loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            
 
          
    return loss_avg(), tch_loss_avg(), ad_loss1_avg(), ad_loss2_avg(), rad_loss_avg()


def evaluate(model, loss_fn, data_loader, params, noise):
    model.eval()
    # summary for current eval loop
    summ = []
    
    class_batch_num = torch.zeros([class_num]).long().cuda()
    class_correct = torch.zeros([class_num]).long().cuda()

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)
            noisy_data_batch = params.lamb * data_batch + \
                                (1-params.lamb) * noise.unsqueeze(0).repeat(data_batch.size()[0],1,1,1)

                

            # compute model output
            output_batch = model(data_batch)
            noise_output_batch = model(noisy_data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            noise_output_batch = noise_output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            noise_output_batch = np.argmax(noise_output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])
            noise_acc = 100.0 * np.sum(noise_output_batch == labels_batch) / float(labels_batch.shape[0])
            summary_batch = {'acc': acc, 'noise_acc': noise_acc, 'loss': loss.item()}
            summ.append(summary_batch)


    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean


def train_and_eval_kd_adv(model, model_ad, model_rad, optim, train_loader, dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate

    noise_data = generate_noise(params)
    torch.save(noise_data, os.path.join(args.save_path, 'noise_data.pth'))
    if params.cuda:
        noise_data = noise_data.cuda()

    for epoch in range(params.num_epochs):
        lr = adjust_learning_rate(optim, epoch, lr, params)
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss, train_tloss, train_aloss1, train_aloss2, train_rloss = train_epoch_kd_adv(model, model_ad, model_rad, optim,
                                                                  train_loader, epoch, params, noise_data)
        logging.info("- Train loss : {:05.3f}".format(train_loss))
        logging.info("- Train teacher loss : {:05.3f}".format(train_tloss))
        logging.info("- Train adversarial loss 1: {:05.3f}".format(train_aloss1))
        logging.info("- Train adversarial loss 2: {:05.3f}".format(train_aloss2))
        logging.info("- Train random loss : {:05.3f}".format(train_rloss))

        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, nn.CrossEntropyLoss(), dev_loader, params, noise_data)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)

        # save model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))
        
        if (epoch+1)%10==0 or epoch==0:
            pd_data = pd.DataFrame(columns=class_name, data=s_train_acc)
            pd_data.to_csv(os.path.join(args.save_path, 'train_acc.csv'), encoding='gbk')
            pd_data = pd.DataFrame(columns=class_name, data=s_test_acc)
            pd_data.to_csv(os.path.join(args.save_path, 'test_acc.csv'), encoding='gbk')


def adjust_learning_rate(opt, epoch, lr, params):
    if epoch in params.schedule:
        lr = lr * params.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)

    # ############################################ Model ############################################
    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    elif params.dataset == 'tiny_imagenet':
        num_class = 200
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))

    logging.info('Create Model --- ' + params.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if params.model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif params.model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif params.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif params.model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif params.model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif params.model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)


    # DenseNet *********************************************
    elif params.model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif params.model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif params.model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif params.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.model_name == 'net':
        model = Net(num_class, params)

    elif params.model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()

    # Adversarial model *************************************************************
    logging.info('Create Adversarial Model --- ' + params.adversarial_model)

    # ResNet 18 / 34 / 50 ****************************************
    if params.adversarial_model == 'resnet18':
        adversarial_model = ResNet18(num_class=num_class)
        random_model = ResNet18(num_class=num_class)
    elif params.adversarial_model == 'resnet34':
        adversarial_model = ResNet34(num_class=num_class)
    elif params.adversarial_model == 'resnet50':
        adversarial_model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.adversarial_model.startswith('preresnet20'):
        adversarial_model = PreResNet(depth=20)
    elif params.adversarial_model.startswith('preresnet32'):
        adversarial_model = PreResNet(depth=32)
    elif params.adversarial_model.startswith('preresnet56'):
        adversarial_model = PreResNet(depth=56)
    elif params.adversarial_model.startswith('preresnet110'):
        adversarial_model = PreResNet(depth=110)

    # DenseNet *********************************************
    elif params.adversarial_model == 'densenet121':
        adversarial_model = densenet121(num_class=num_class)
    elif params.adversarial_model == 'densenet161':
        adversarial_model = densenet161(num_class=num_class)
    elif params.adversarial_model == 'densenet169':
        adversarial_model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.adversarial_model == 'resnext29':
        adversarial_model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.adversarial_model == 'mobilenetv2':
        adversarial_model = MobileNetV2(class_num=num_class)

    elif params.adversarial_model == 'shufflenetv2':
        adversarial_model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.adversarial_model == 'net':
        adversarial_model = Net(num_class, params)
        random_model = Net(num_class, params)

    elif params.adversarial_model == 'mlp':
        adversarial_model = MLP(num_class=num_class)

    else:
        adversarial_model = None
        print('Not support for model ' + str(params.adversarial_model))
        exit()

    if params.cuda:
        model = model.cuda()
        adversarial_model = adversarial_model.cuda()
        random_model = random_model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        adversarial_model = nn.DataParallel(adversarial_model, device_ids=device_ids)
        random_model = nn.DataParallel(random_model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Train from scratch ')

    # load trained Adversarial model ****************************
    logging.info('- Load Trained adversarial model from {}'.format(params.adversarial_resume))
    checkpoint = torch.load(params.adversarial_resume)
    adversarial_model.load_state_dict(checkpoint['state_dict'])

    # ############################### Optimizer ###############################
    if params.model_name == 'net' or params.model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=params.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        logging.info('Optimizer: SGD')

    # ************************** train and evaluate **************************
    train_and_eval_kd_adv(model, adversarial_model, random_model, optimizer, trainloader, devloader, params)

