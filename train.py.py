import os
import numpy as np
import random
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
import torchvision.transforms.functional as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm
import argparse
from torch.backends import cudnn
from model import SwinTransformerSys
from tqdm import tqdm
#from utils import parse_args,rgb_to_y,psnr,ssim


class Config(object):
    def __init__(self,args):
        self.data_path=args.data_path
        self.data_name=args.data_name
        self.save_path=args.save_path
        self.num_blocks=args.num_blocks
        self.num_heads=args.num_heads
        self.channels=args.channels
        self.expansion_factor=args.expansion_factor
        self.num_refinement=args.num_refinement
        self.num_iter=args.num_iter
        self.batch_size=args.batch_size
        self.patch_size=args.patch_size
        self.learning_rate=args.lr
        self.milestones=args.milestones
        self.num_workers=args.num_workers
        self.seed=args.seed
        self.model_file=args.model_file
        self.test_path=args.test_path
def init_args(self,args):
    if not os.path.exists(args.save_path):
       os.makedirs(args.save_path)

    elif args.seed>=0:
       random.seed(args.seed)
       np.random.seed(args.seed)
       torch.manual_seed(args.seed)
       torch.cuda.manual_seed(args.seed)
       cudnn.deterministic=True
       cudnn.benchmark=False
    return Config(args)

def pad_image_needed(img,size):
    width,height=T.get_image_size(img)
    if width<size[1]:
        img=T.pad(img,[size[1]-width,0],padding_mode='reflect')

    elif height<size[0]:
        img=T.pad(img,[0,size[0]-height],padding_mode='reflect')


def rgb_to_y(x):
    rgb_to_grey=torch.tensor([0.256789,0.504129,0.097906],dtype=x.dtype,device=x.device).view(1,-1,1,1)
    return torch.sum(x*rgb_to_grey,dim=1,keepdim=True)
    
def psnr(x,y,data_range=255.0):
    x=x/data_range
    y=y/data_range
    mse=torch.mean((x-y)**2)
    score=-10*torch.log10(mse)
    return score

def ssim(x,y,kernel_size=11,kernel_sigma=0.5,data_range=255.0,k1=0.01,k2=0.03):
    x=x/data_range
    y=y/data_range
    
    #average pool image if the size is large enough
    f=max(1,round(min(x.size()[-2:]/256)))
    if f>1:
        x,y=F.avg_pool2d(x,kernel_size=f),F.avg_pool2d(y,kernel_size=f)
    #gausian filter
    coords=torch.arange(kernel_size,dtype=x.dtype,device=x.device)
    coords-=(kernel_size-1)/2.0
    g=coords**2
    g=(-(g.unsqueeze(0)+g.unsqueeze(1))/(2*kernel_sigma**2)).exp()
    g/=g.sum()
    kernel=g.unsqueeze(0).repeat(x.size(1),1,1,1)

    #compute
    c1,c2=k1**2,k2**2
    n_channels=x.size(1)
    mu_x=F.conv2d(x,weight=kernel,stride=1,padding=0,groups=n_channels)
    mu_y=F.conv2d(y,weight=kernel,stride=1,padding=0,groups=n_channels)

    mu_xx,mu_yy,mu_xy=mu_x**2,mu_y**2,mu_x*mu_y
    sigma_xx=F.conv2d(x**2,weight=kernel,stride=1,padding=0,groups=n_channels)-mu_xx
    sigma_yy=F.conv2d(y**2,weight=kernel,stride=1,padding=0,groups=n_channels)-mu_yy
    sigma_xy=F.conv2d(x*y,weight=kernel,stride=1,padding=0,groups=n_channels)-mu_xy

    # contrast sensitivity (CS) with alpha=beta=gamma=1
    cs=(2.0*sigma_xy+c2)/(sigma_xx+sigma_yy+c2)

    #structural similarity
    ss=(2.0*mu_xy+c1)/(mu_xx+mu_yy+c1)*cs
    return ss.mean()




class RainDataset(Dataset):
    def __init__(self,data_path,data_name,data_type,patch_size=None,length=None):
        super().__init__()
        self.data_path,self.data_name,self.data_type,self.patch_size=data_path,data_name,data_type
          #pass
        self.rain_images=sorted(glob.glob('{}/{}/{}/rain/*.png'.format(data_path,data_name,data_type)))
        self.norain_images=sorted(glob.glob('{}/{}/{}/norain/*.png'.format(data_path,data_name,data_type)))
        #making sure the length of the training and test is different
        self.num=len(self.rain_images)
        self.sample_num=length if data_type=="train" else self.num
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self,idx):
        image_name=os.path.basename(self.rain_images[idx%self.num])
        rain=T.to_tensor(Image.open(self.rain_images[idx%self.num]))
        norain=T.to_tensor(Image.open(self.norain_images[idx%self.num]))
        h,w=rain.shape[1:]

        if self.data_type=="train":
           #making sure that image need to be cropped
           rain=pad_image_needed(rain,(self.patch_size,self.patch_size))
           norain=pad_image_needed(norain,(self.patch_size,self.patch_size))
           i,j,th,tw=T.RandomCrop.get_params(rain,(self.patch_size,self.patch_size))
           rain=T.crop(rain,i,j,th,tw)
           norain=T.crop(norain(i,j,th,tw))
           if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
           if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        else:
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = T.RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
        return rain,norain,h,w
    
def test_loop(net,data_loader,num_iter):
    #mention the images which you want to track during training
    intermediate_tracked_images=['norain-11.png','norain-32.png','norain-40.png','norain-48.png','norain-53.png','norain-83.png']
    #mention the iterations at which you want to track the images
    intermediate_tracked_iters = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 30000, 60000, 100000, 130000, 160000, 180000, 200000, 230000, 260000, 290000, 320000, 340000, 350000, 380000, 400000]
    net.eval()
    total_psnr,total_ssim,count=0.0,0.0,0
    with torch.no_grad:
        test_bar=tqdm(data_loader,intial=1,dynamic_ncols=True)
        for rain, norain, name, h,w in test_bar:
            rain,norain=rain.cuda(),norain.cuda()
            #out=torch.clamp(torch.clamp(model(rain)[:,;,h,w],0,1).mul(225)),0,255)
            out=torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()

            #compute the metrics with y channel and double the precision
            y,gt=rgb_to_y(out.double()),rgb_to_y(norain.double())
            current_psnr=psnr(y,gt)
            current_ssim=ssim(y,gt)
            total_psnr+=current_psnr.item()
            total_ssim+=current_ssim.item()
            count+=1
            save_path='{}/{}/{}'.format(args.data_name,args.data_name,name[0])

            #saving the intermediate images
            if num_iter in intermediate_tracked_iters:
                if name[0] in intermediate_tracked_images:
                    this_psnr=round(total_psnr/count,2)
                    image_file_name=f"{name[0].split('.')[0]}_psnr_{this_psnr}.{name[0].split('.')[1]}"
                    this_inter_save_path=f'this_intermediate_train_images/{image_file_name}'
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    Image.fromarray(out.squeeze(dim=0).permute(1,2,0).contiguous().cpu().numpy()).save(this_inter_save_path)
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1,2,0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 1 if args.model_file else args.num_iter,
                                             total_psnr / count, total_ssim / count))
    return total_psnr/count, total_ssim/count
   
def save_loop(net,data_loader,num_iter):
    global best_psnr,best_ssim
    val_psnr,val_ssim=test_loop(net,data_loader,num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    #save statistics
    data_frame=pd.DataFrame(data=results,index=range(1,(num_iter if args.model_file else num_iter // 1000)))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    if val_psnr>best_psnr and val_ssim>best_ssim:
        best_psnr,best_ssim=val_psnr,val_ssim
        with open('{}/{}.txt'.format(args.data_path,args.data_name),'w') as f:
             f.write('Iter: {} PSNR: {:.2f} SSIM: {:.3f}'.format(num_iter,best_psnr,best_ssim))
        #saving the model
        torch.save(model.state_dict(),'{}/{}.pth'.format(args.save_path,args.data_name))

def parse_args():
    desc="Pytorch Implementation of SwinStorm- High Resolution Image-DeRaining"
    parser=argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path',type=str,default='data')
    parser.add_argument('--data_name',type=str,default='rain100H',choices=['rain100L','rain100H'])
    parser.add_argument('--save_path',type=str,default='result')
    parser.add_argument('--num_blocks',type=int,default=[4,6,6,8],nargs='+',
                        help='number of transformer blocks at each level')
    parser.add_argument('--num_heads',type=int,default=[1,2,4,8],nargs='+',
                        help='number of attention heads at each level')
    parser.add_argument('--channels',type=int,nargs='+',default=[48,96,192,384],
                        help='number of channels for each level')
    parser.add_argument('--expansion_factor',type=float,default=2.66,help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement',type=int,default=4,help='number of channels for refinement')
    parser.add_argument('--num_iter',type=int,default=240000,help='number of iterations for training')
    parser.add_argument('--batch_size',type=int,default=[4,4,4,4,4,4],help='batch_size of loading images for progresiive learning')
    parser.add_argument("--patch_size",type=int,default=[224,224,224,224,224,224],help='patch_size of each image for progressive learning')
    parser.add_argument("--lr",type=float,default=0.0003,help='intial learning rate')
    parser.add_argument("--milestones",type=int,default=[92000, 156000, 204000, 240000],help='when to change patch size and batch size')
    parser.add_argument("--num_workers",type=int,default=8,help='number of data loading workers')
    parser.add_argument('--seed',type=int,default=-1,help='random seed -1 for no manual seed')
    
    #model file is None that means training stage else means testing stage
    parser.add_argument('--model_file',type=str,default=None,help='path for pretrained model file')
    parser.add_argument('--test_path',type=str,default='',help='path for validation/test data')

    return init_args(parser.parse_args())

if __name__=='__main__':
    args=parse_args()
    test_dataset=RainDataset(args.data_path,args.data_name,'test',args.patch_size[0])
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers)

    results={'PSNR': [],'SSIM':[]}
    best_psnr=0.0
    best_ssim=0.0

    #intializing the model
    model =SwinTransformerSys().cuda()
    #print(model)
    #exit()
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        save_loop(model,test_loader,1)
    else:
        optimizer=AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        lr_scheduler=CosineAnnealingLR(optimizer,T_max=args.num_iter,eta_min=1e-6)
        total_loss=0.0
        total_num=0
        results['loss']=list()
        i=0
        train_bar=tqdm(range(1,args.num_iter+1),initial=1,dynamic_ncols=True)
        for n_iter in train_bar:
            #progressive learning
            if n_iter==1 or n_iter-1 in args.milestones:
                if i<len(args.milestones):
                    end_iter=args.milestones[i]
                else:
                    end_iter=args.num_iter
                if i>0:
                    start_iter=args.milestones[i-1]
                else:
                    start_iter=0
                length=args.batch_size[i]*(end_iter-start_iter)
                train_dataset=RainDataset(args.data_path,args.data_name,'train',args.patch_size[1],length)
                num=15
                a_org, b_org, _, _, _ = test_dataset.__getitem__(num)
                print(a_org.shape)
                train_loader=iter(DataLoader(train_dataset,args.batch_size[i],shuffle=True,num_workers=args.num_workers))
                i+=1
            model.train()
            rain,norain,name,h,w=next(train_loader)
            rain=rain.cuda()
            norain=norain.cuda()
            out=model(rain)
            loss=F.l1_loss(out,norain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num+=rain.size(0)
            total_loss+=loss.item()*rain.size(0)
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'.format(n_iter,args.num_iter,total_loss/total_num))
            lr_scheduler.step()

            if n_iter%1000==0:
                results['loss'].append('{:.3f}'.format(total_loss/total_num))
                save_loop(model,test_loader,n_iter)


               




