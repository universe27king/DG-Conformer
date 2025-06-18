from . import loss_reweighting
import torch
import torch.nn as nn
from torch.autograd import Variable
from .schedule import lr_setter


def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)        # 特征图叠到一起：（128+batchsize，512）128是pre_features定义的参数args.n_feature；cfeaturec是batchsize
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)
    # optimizerbl = torch.optim.SGD([weight], lr=0.1, momentum=0.9)     # 上面一行是原文用法但是原文主网络训练lr0.01所以尝试等比例缩小
    # optimizerbl = torch.optim.Adam([weight], lr=0.1, betas=(0.5, 0.999), weight_decay=1e-4)
    
    
    for epoch in range(args.epochb):   # number of epochs to balance
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)         # 权重叠到一起：（128+batchsize，1）
        optimizerbl.zero_grad()

        lossb = loss_reweighting.lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        #print('lossb', lossb)
        #print('lossp', lossp)
        
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        
        #print('lambdap', lambdap)
        #print('lossg', lossg)
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    softmax_weight = softmax(weight)
    #softmax_weight = weight

    return softmax_weight, pre_features, pre_weight1
