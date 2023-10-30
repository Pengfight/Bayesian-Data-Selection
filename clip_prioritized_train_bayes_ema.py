'''
CUDA_VISIBLE_DEVICES=5 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset cifar100_clip --save_name bayes_e200_ema --alpha .3 --tau 12 --num_effective_data 1000 --prior_precision 1

CUDA_VISIBLE_DEVICES=4 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset cifar10_clip --save_name bayes_e200_tau4_alpha.2_2e2d_ema --alpha .2 --num_effective_data 200 --prior_precision 10 --tau 4

CUDA_VISIBLE_DEVICES=2 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset cifar100_lt --save_name bayes_e200_ema_alpha02_tau6_sel01_n100p10 --alpha .2 --tau 6 --num_effective_data 100 --prior_precision 10 --lt

CUDA_VISIBLE_DEVICES=4 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset cifar10_lt --save_name bayes_e200_tau4_alpha.2_2e2d_ema --alpha .2 --num_effective_data 200 --prior_precision 10 --tau 4 --lt

'''
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cf

import os
import time
import argparse
import numpy as np
import torchvision
import warnings
from datasets.clothing import clothing_dataset
import torchvision.transforms as transforms
from datasets.webvision import webvision_dataset



from utils import check_dir, prepare_dset, prepare_dset_lt, update_print, clip_classifier, ECELoss

from torch.utils.tensorboard import SummaryWriter

import clip

# pip install ema-pytorch
from ema_pytorch import EMA

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--warmup', default=-1, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--decay', default=0.01, type=float)
parser.add_argument('--gamma', default=0.2, type=float, help='gamma')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimize method')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=320, type=int)
parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10/cifar100')
parser.add_argument('--milestones', nargs='+', default=[60,120,160], type=int)

parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')

parser.add_argument('--ynoise_type', default='symmetric', type=str, help='symmetric/pairflip')
parser.add_argument('--ynoise_rate', default=0.0, type=float, help='label noise rate')
parser.add_argument('--xnoise_type', default='blur', type=str, help='gaussian/blur')
parser.add_argument('--xnoise_arg', default=1, type=float)
parser.add_argument('--xnoise_rate', default=0.0, type=float)
parser.add_argument('--trigger_size', type=int, default=3)
parser.add_argument('--trigger_ratio', type=float, default=0.)

parser.add_argument('--clip_arch', default='ViT-B/16', type=str)
parser.add_argument('--random_state', type=int, default=20)
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--save_name', default='default', type=str)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--tau', default=12, type=float)
parser.add_argument('--select_rate', default=0.1, type=float)
parser.add_argument('--adaptive_alpha', action='store_true', default=False)
parser.add_argument('--lt', action='store_true', default=False)


parser.add_argument('--split', default=0.5, type=float)
parser.add_argument('--num_effective_data', type=int, default=10000)
parser.add_argument('--prior_precision', default=1, type=float)
parser.add_argument('--laplace_momentum', default=0.99, type=float)
parser.add_argument('--n_f_samples', type=int, default=256)

parser.add_argument('--ema_momentum', default=0.99, type=float)
parser.add_argument('--num_classes', type=int, default=1000)


def main():
    args = parser.parse_args()
    setup_seed(args.random_state)
    print(list(args.milestones), args)    

    # Hyper Parameter settings
    best_acc = 0
    num_epochs, batch_size, optim_type = args.num_epochs, args.batch_size, args.optim_type

    # Dataloader
    print('\n[Phase 1] : Data Preparation')
    if args.lt:
        trainset, testset = prepare_dset_lt(args)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]) 
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        if args.dataset == 'webvision':
            trainset = webvision_dataset(transform=transform_train, mode="all", num_class=args.num_classes)
            testset =  webvision_dataset(transform=transform_test, mode="test", num_class=args.num_classes)
        else:
            trainset, testset, trainvalset = prepare_dset(args)
    num_classes = args.num_classes
    classnames = trainset.classnames
    template = trainset.template
    
    rng = np.random.RandomState(args.random_state)
    random_permutation = rng.permutation(len(trainset))
    print(random_permutation[:10])

    valset = torch.utils.data.Subset(trainset, random_permutation[int(len(trainset) * args.split):])
    trainset = torch.utils.data.Subset(trainset, random_permutation[:int(len(trainset) * args.split)])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)

    print('\n[Phase 2] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    # the model 
    net, file_name = getNetwork(args, num_classes)
    net.cuda()

    # the ema model 
    ema_net = EMA(
        net,
        beta = args.ema_momentum,              # exponential moving average factor
        update_after_step = 0,    # only after this number of .update() calls will it start updating
        update_every = 5,          # how often to actually update, to save on compute (updates every 10th .update() call)
    )
    ema_net.eval()

    # the ema model
    net = KFCALLAWrapper(net, args.num_effective_data, args.prior_precision, args.n_f_samples, momentum=args.laplace_momentum)
    net = net.cuda()
    # net = torch.nn.DataParallel(net)
    print(net)

    # the clip model
    clip_clf = CLIPZeroShotClassifier(classnames, template, args.dataset, args.clip_arch).cuda()
    clip_clf.eval()

    if optim_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,  nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(args.milestones), gamma=args.gamma)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(net.parameters()) #, lr=args.lr, weight_decay=args.decay)
        scheduler = None #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = FocalLoss().cuda()

    writer = SummaryWriter(log_dir='logs_abla/' + file_name)

    elapsed_time, best_acc = 0, 0
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # model training
        net.train()
        correct, total = 0, 0
        alpha =args.alpha
        for batch_idx, ((inputs, xnoisy), (targets, true_tar)) in enumerate(trainloader):
            iter = epoch * len(trainloader) + batch_idx
            inputs, targets = inputs.cuda(), targets.cuda()

            if epoch > args.warmup:
                with torch.no_grad():
                    f_samples, outputs, stds, L_U_T_inverse = net(inputs, selection_pass=True)
                    clip_outputs = clip_clf(inputs, tau=args.tau)
                    if args.adaptive_alpha:
                        # bnn_acc = outputs.argmax(-1).eq(targets).float().mean()
                        # clip_acc = clip_outputs.argmax(-1).eq(targets).float().mean()
                        # alpha = bnn_acc/(bnn_acc + clip_acc)
                        bayes_loss = criterion(outputs, targets).item()
                        clip_loss = criterion(clip_outputs, targets).item()
                        alpha = clip_loss/(bayes_loss + clip_loss)

                    first_term = - F.cross_entropy(f_samples.flatten(0, 1), targets.repeat_interleave(f_samples.shape[1]), reduction='none').view(f_samples.shape[0], f_samples.shape[1]).mean(1)
                    second_term = - F.cross_entropy(clip_outputs, targets, reduction='none')
                    third_term = - F.cross_entropy(f_samples.softmax(-1).mean(1).log(), targets, reduction='none')
                    select_obj = alpha * first_term + (1 - alpha) * second_term - third_term
                    _, indices = torch.topk(select_obj, int(targets.shape[0] * args.select_rate))

                    writer.add_scalar("Select/clip_acc", clip_outputs.argmax(-1).eq(targets).float().mean(), iter)
                    writer.add_scalar("Select/first_minus_third_term", first_term[indices].mean() - third_term[indices].mean(), iter)
                    writer.add_scalar("Select/second_term", second_term[indices].mean(), iter)
                    writer.add_scalar("Select/third_term", third_term[indices].mean(), iter)
                    writer.add_scalar("Select/select_obj", select_obj[indices].mean(), iter)
                    writer.add_scalar("Select/alpha", alpha, iter)
                    if stds is not None:
                        writer.add_scalar("Select/stds", stds[indices].mean(), iter)
                    if L_U_T_inverse is not None:
                        writer.add_scalar("Select/L_U_T_inverse", L_U_T_inverse.item(), iter)
                    writer.add_scalar("Select/f_mean", outputs[indices].mean(), iter)
            else:
                _, indices = torch.topk(torch.randn(inputs.shape[0]).cuda(), int(targets.shape[0] * args.select_rate))

            inputs = inputs[indices]
            targets = targets[indices]

            outputs = net(inputs, y=targets)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                acc = predicted.eq(targets.data).float().cpu().mean()
                total += targets.size(0)
                correct += acc.item() * targets.size(0)
            
            update_print('| Epoch [%3d/%3d]  Iter[%3d/%3d]  Loss: %.4f  Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1, len(trainloader), loss.item(), 100.*correct/total))

            writer.add_scalar("Train/loss", loss, iter)
            writer.add_scalar("Train/acc", acc, iter)

            ema_net.update()


        writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], epoch)
        if scheduler is not None:
            scheduler.step()

        # model eval
        net.eval()
        with torch.no_grad():
            logits_list, ema_logits_list, labels_list = [], [], []
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                logits = net(inputs); ema_logits = ema_net(inputs)
                logits_list.append(logits); ema_logits_list.append(ema_logits); labels_list.append(targets)

            logits, ema_logits, labels = torch.cat(logits_list, 0), torch.cat(ema_logits_list, 0), torch.cat(labels_list,0)

            eval_loss = F.cross_entropy(logits, labels).item()
            ema_eval_loss = F.cross_entropy(ema_logits, labels).item()
            ece = ECELoss(logits, labels).item()
            ema_ece = ECELoss(ema_logits, labels).item()
            acc = logits.argmax(-1).eq(labels).float().mean().item()
            ema_acc = ema_logits.argmax(-1).eq(labels).float().mean().item()
            print(f'| Eval results : Acc1={round(acc, 4)}', f'ECE={round(ece, 4)}, Acc1 (ema)={round(ema_acc, 4)}', f'ECE (ema)={round(ema_ece, 4)}')

            writer.add_scalar("Eval/loss", eval_loss, epoch)
            writer.add_scalar("Eval/acc", acc, epoch)
            writer.add_scalar("Eval/ece", ece, epoch)
            writer.add_scalar("Eval/EMA/loss", ema_eval_loss, epoch)
            writer.add_scalar("Eval/EMA/acc", ema_acc, epoch)
            writer.add_scalar("Eval/EMA/ece", ema_ece, epoch)

            if acc > best_acc and epoch <=200:
                best_acc = acc 
                # print('New Best Model')
                if args.save_model:
                    state = {
                        'net': net.state_dict(), 
                        'opt': optimizer.state_dict(), 
                        'acc': acc,
                        'epoch': epoch
                    }
                    save_point = 'checkpoint'
                    check_dir(save_point)
                    base_dir = os.path.join(save_point, "default")
                    check_dir(base_dir)
                    save_path = os.path.join(base_dir, file_name + '.pkl')
                    print('Save Model to', save_path)
                    torch.save(state, save_path)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

    print('Best Acc:', best_acc)
    writer.flush()
    writer.close()

class KFCALLAWrapper(nn.Module):
    def __init__(self, net, num_effective_data, prior_precision, n_f_samples, last_layer_name="fc", momentum=0.99) :
        super(KFCALLAWrapper, self).__init__()
        self.net = net
        self.num_effective_data = num_effective_data
        self.prior_precision = prior_precision
        self.n_f_samples = n_f_samples
        self.momentum = momentum

        self.input_features_of_last_layer = None
        self.fhook = getattr(self.net, last_layer_name).register_forward_hook(self.forward_hook())

        with torch.no_grad():
            self.net.training = False
            out = self.net(torch.zeros(1, 3, 32, 32).cuda())
            self.net.training = True

            feature_dim = self.input_features_of_last_layer.shape[-1]
            out_dim = out.shape[-1]

        self.register_buffer("num_data", torch.Tensor([0]))
        self.register_buffer("A", torch.zeros(feature_dim, feature_dim))
        self.register_buffer("G", torch.zeros(out_dim, out_dim))
        self.register_buffer("G2", torch.zeros(out_dim, out_dim))
    
    def forward_hook(self):
        def hook(module, input, output):
            self.input_features_of_last_layer = input[0]
        return hook

    def forward(self, x, selection_pass=False, y=None):
        bs = x.shape[0]
        if selection_pass:
            self.net.apply(_freeze)
        out = self.net(x)

        if selection_pass:
            self.net.apply(_unfreeze)

            if self.num_data.item() == 0:
                return out[:, None, :], out, None, None
            
            with torch.no_grad():
                V = np.sqrt(self.num_effective_data) * self.A
                V.diagonal().add_(np.sqrt(self.prior_precision))
                L_V = psd_safe_cholesky(V)
                U = np.sqrt(self.num_effective_data) * self.G
                U.diagonal().add_(np.sqrt(self.prior_precision))
                L_U = psd_safe_cholesky(U)
                
                V_inv = torch.cholesky_inverse(L_V)
                stds = (self.input_features_of_last_layer @ V_inv * self.input_features_of_last_layer).sum(-1).clamp(min=1e-6).sqrt()
                L_f = stds.view(-1, 1, 1) * L_U.T.inverse()
                f_samples = out[:, None, :] + torch.randn((bs, self.n_f_samples, out.shape[-1])).to(x.device) @ L_f
                return f_samples, out, stds, torch.linalg.matrix_norm(L_U.T.inverse(), ord=2)
        elif self.training:
            assert y is not None
            with torch.no_grad():
                
                feature_cov = self.input_features_of_last_layer.T @ self.input_features_of_last_layer / bs
                if self.num_data.item() == 0:
                    self.A.data.copy_(feature_cov)
                else:
                    self.A.mul_(self.momentum).add_(feature_cov, alpha = 1-self.momentum)

                prob = out.softmax(-1)
                grad = prob - F.one_hot(y, out.shape[-1])
                grad_cov = grad.T @ grad / bs
                if self.num_data.item() == 0:
                    self.G.data.copy_(grad_cov)
                else:
                    self.G.mul_(self.momentum).add_(grad_cov, alpha = 1-self.momentum)
                
                # grad_cov2 = (prob.diag_embed() - prob[:, :, None] * prob[:, None, :]).mean(0)
                # if self.num_data.item() == 0:
                #     self.G2.data.copy_(grad_cov2)
                # else:
                #     self.G2.mul_(self.momentum).add_(grad_cov2, alpha = 1-self.momentum)
                self.num_data.add_(bs)
                # print(self.A[:10,:10], self.G[:10,:10], self.G2[:10,:10])

        return out

class CLIPZeroShotClassifier(nn.Module):
    def __init__(self, classnames, template, dataset, arch="RN50") :
        super(CLIPZeroShotClassifier, self).__init__()
        clip_model, preprocess = clip.load(arch, jit=False)
        clip_model.eval()
        self.clip_model = clip_model
        clip_weights = clip_classifier(classnames, template, clip_model)
        self.register_buffer('clip_weights', clip_weights)

        self.register_buffer('old_mean', torch.Tensor(cf.mean[dataset]))
        self.register_buffer('old_std', torch.Tensor(cf.std[dataset]))
        # self.register_buffer('old_mean', torch.Tensor([0.485, 0.456, 0.406]))
        # self.register_buffer('old_std', torch.Tensor([0.229, 0.224, 0.225]))
        
        self.register_buffer('new_mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]))
        self.register_buffer('new_std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        self.input_size = preprocess.transforms[0].size
    
    @torch.no_grad()
    def forward(self, inputs, tau=12.):
        inputs = inputs.mul(self.old_std.view(-1, 1, 1)).add(self.old_mean.view(-1, 1, 1))
        if inputs.shape[2] == 32:
            inputs = F.interpolate(inputs, self.input_size, mode='bicubic')
        inputs = inputs.sub(self.new_mean.view(-1, 1, 1)).div(self.new_std.view(-1, 1, 1))

        input_features = self.clip_model.encode_image(inputs)
        clip_logits = tau * input_features @ self.clip_weights
        return clip_logits

def getNetwork(args, num_classes, pretrained=False):
    
    if args.dataset == 'webvision':
        model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
        file_name = 'resnet18_' + args.dataset + '_' + args.save_name
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes, bias=True)

    else:
        model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
        file_name = 'resnet18_' + args.dataset + '_' + args.save_name
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes, bias=True)

    return model, file_name

def _freeze(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.track_running_stats = False

def _unfreeze(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.track_running_stats = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        # return torch.randn_like(A).tril()
        raise e

if __name__ == '__main__':
    main()