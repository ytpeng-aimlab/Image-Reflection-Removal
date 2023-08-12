import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler
import Block_version_5
import matplotlib.pyplot as plt
from data.dataloader_v2 import *
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import L1Loss
from loss import ssim

L1_loss = torch.nn.L1Loss()
Reconstruct_loss = torch.nn.L1Loss()

def DCLoss(img):
    maxpool = nn.MaxPool3d((3, 15, 15), stride=1, padding=(0, 15//2, 15//2))
    dc = maxpool(1-img[:, None, :, :, :])
    target = torch.FloatTensor(dc.shape).zero_().cuda()
    loss =  L1Loss(reduction='mean')(dc, target)
    return -loss

def l1_loss(input,output):
    return torch.mean(torch.abs(input - output))

def feat_loss(vgg_feature,output_t,gt_t):      
    feat_loss = 0
    conv_list = [2, 7, 12, 21, 30]
    sigma_list = [1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5]
    i = 0
    for index in range(31):
        torch.cuda.empty_cache()
        output_t = vgg_feature.features[index](output_t)
        gt_t = vgg_feature.features[index](gt_t)
        if index in conv_list:
            feat_loss+= l1_loss(output_t,gt_t)*sigma_list[i]
            i += 1
    return feat_loss  

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def _train(teacher_T, teacher_R, student, args):
    writer = SummaryWriter()

    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    
    optimizer_teacher_T = torch.optim.Adam(teacher_T.parameters(), lr = args.learning_rate_TT, betas=(args.bata1, args.bata2))
    optimizer_teacher_R = torch.optim.Adam(teacher_R.parameters(), lr = args.learning_rate_TR, betas=(args.bata1, args.bata2))
    optimizer_student = torch.optim.Adam(student.parameters(), lr = args.learning_rate_S, betas=(args.bata1, args.bata2))
    
    # optimizer_teacher_T = torch.optim.SGD(teacher_T.parameters(), lr = args.learning_rate_TT, momentum=0.9)
    # optimizer_teacher_R = torch.optim.SGD(teacher_R.parameters(), lr = args.learning_rate_TR, momentum=0.9)
    # optimizer_student = torch.optim.SGD(student.parameters(), lr = args.learning_rate_S, momentum=0.9)

    teacher_T.train() ; teacher_T.to(device)
    teacher_R.train() ; teacher_R.to(device)
    student.train() ; student.to(device)
    
    if torch.cuda.device_count() > 1:
        # model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        teacher_T = torch.nn.parallel.DistributedDataParallel(teacher_T, device_ids=[args.local_rank], output_device=args.local_rank)
        teacher_R = torch.nn.parallel.DistributedDataParallel(teacher_R, device_ids=[args.local_rank], output_device=args.local_rank)
        student   = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # teacher_T = nn.DataParallel(teacher_T)
    # teacher_R = nn.DataParallel(teacher_R)
    # student = nn.DataParallel(student)

    vgg_feature = models.vgg19()
    vgg_feature.to(device)
    
    epoch = 1
    if args.resume_TT: 
        teacher_T.load_state_dict(torch.load(args.resume_TT)['model'])
    if args.resume_TR: 
        teacher_R.load_state_dict(torch.load(args.resume_TR)['model'])
    if args.resume_S: 
        student.load_state_dict(torch.load(args.resume_S)['model'])
        epoch =  torch.load(args.resume_S)['epoch']+1

    # train_dataset = datasets(os.path.join(args.data_dir,'train'), 'input' , 'gt', transform )
    # train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)   
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, True)

    max_iter = len(dataloader)
    Charbonnier_loss = L1_Charbonnier_loss()
    
    for epoch_idx in range(epoch, args.num_epoch):        
        train_loss = 0
        for batch_idx , (reflect_image , clean_image) in enumerate(dataloader):
            
            optimizer_teacher_T.zero_grad()
            optimizer_teacher_R.zero_grad()
            optimizer_student.zero_grad()

            reflect_image = reflect_image.to(device)
            clean_image = clean_image.to(device)
            
            compoment_image = reflect_image-clean_image  
            
            teacher_T_out1, teacher_T_out2, teacher_T_out3, teacher_T_out4, \
            teacher_T_out5, teacher_T_out6, teacher_T_out7, teacher_T_out8, \
            teacher_T_out = teacher_T(clean_image)
            teacher_T_loss = Charbonnier_loss(teacher_T_out, clean_image)

            teacher_R_out1, teacher_R_out2, teacher_R_out3, teacher_R_out4, \
            teacher_R_out5, teacher_R_out6, teacher_R_out7, teacher_R_out8, \
            teacher_R_out = teacher_R(compoment_image)
            teacher_R_loss = Charbonnier_loss(teacher_R_out, compoment_image)
            
            sout_share1, sout_share2, sout_share3 ,sout_share4, \
            sout_T1, sout_T2, sout_T3, sout_T4, \
            sout_R1, sout_R2, sout_R3, sout_R4, \
            student_T_out, student_R_out = student( reflect_image )
        
            dcl_loss = DCLoss(student_T_out)
            f_loss_T = feat_loss(vgg_feature, student_T_out , clean_image)
            f_loss_R = feat_loss(vgg_feature, student_R_out , compoment_image)            
            fea_loss = f_loss_R + f_loss_T
            
            RM_T_loss_1 = L1_loss(teacher_T_out1, sout_share1[:,64:,:,:])
            RM_T_loss_2 = L1_loss(teacher_T_out2, sout_share2[:,64:,:,:]) 
            RM_T_loss_3 = L1_loss(teacher_T_out3, sout_share3[:,64:,:,:])
            RM_T_loss_4 = L1_loss(teacher_T_out4, sout_share4[:,64:,:,:]) 
            RM_T_loss_5 = L1_loss(teacher_T_out5, sout_T1)
            RM_T_loss_6 = L1_loss(teacher_T_out6, sout_T2)
            RM_T_loss_7 = L1_loss(teacher_T_out7, sout_T3)
            RM_T_loss_8 = L1_loss(teacher_T_out8, sout_T4)
            
            RM_R_loss_1 = L1_loss(teacher_R_out1, sout_share1[:,:64,:,:])
            RM_R_loss_2 = L1_loss(teacher_R_out2, sout_share2[:,:64,:,:]) 
            RM_R_loss_3 = L1_loss(teacher_R_out3, sout_share3[:,:64,:,:])
            RM_R_loss_4 = L1_loss(teacher_R_out4, sout_share4[:,:64,:,:]) 
            RM_R_loss_5 = L1_loss(teacher_R_out5, sout_R1)
            RM_R_loss_6 = L1_loss(teacher_R_out6, sout_R2) 
            RM_R_loss_7 = L1_loss(teacher_R_out7, sout_R3)
            RM_R_loss_8 = L1_loss(teacher_R_out8, sout_R4) 
            
            RM_Top =  RM_T_loss_1 +  RM_T_loss_2 +  RM_T_loss_3 + RM_T_loss_4 + RM_T_loss_5 + RM_T_loss_6 + RM_T_loss_7 + RM_T_loss_8
            RM_Bot =  RM_R_loss_1 +  RM_R_loss_2 +  RM_R_loss_3 + RM_R_loss_4 + RM_R_loss_5 + RM_R_loss_6 + RM_R_loss_7 + RM_R_loss_8
            
            I_T_loss = L1_loss(reflect_image-teacher_T_out, teacher_R_out)       # + ssim(reflect_image-teacher_T_out, teacher_R_out)
            I_Tout_loss = L1_loss(reflect_image-teacher_T_out, student_R_out)    # + ssim(reflect_image-teacher_T_out, student_R_out)
            I_OUT_T_loss = L1_loss(reflect_image-student_T_out, teacher_R_out)   # + ssim(reflect_image-student_T_out, teacher_R_out)
            
            Res_loss = Reconstruct_loss(student_R_out, compoment_image) + Reconstruct_loss(student_T_out, clean_image) + 0.3 * (RM_Bot)  + 0.3 * (RM_Top)  + 0.3 *(I_OUT_T_loss + I_T_loss + I_Tout_loss) + 0.0001 * dcl_loss + 0.15 * fea_loss
            
            # ssim(student_R_out, compoment_image) + ssim(student_T_out, clean_image) +\
            
            train_loss += Res_loss.item()
            student_loss = Res_loss 

            teacher_T_loss.backward(retain_graph = True)
            teacher_R_loss.backward(retain_graph = True)
            student_loss.backward(retain_graph = True)
            
            optimizer_teacher_T.step()
            optimizer_teacher_R.step() 
            optimizer_student.step()
            
            if batch_idx % 20 == 0 :
                writer.add_image("reflection",      reflect_image[0],   global_step=batch_idx + (epoch_idx-1)* max_iter)
                writer.add_image("gt",              clean_image[0],     global_step=batch_idx + (epoch_idx-1)* max_iter)
                writer.add_image('Student_T_out',   student_T_out[0],   global_step=batch_idx + (epoch_idx-1)* max_iter)
                writer.add_image('Student_R_out',   student_R_out[0],   global_step=batch_idx + (epoch_idx-1)* max_iter)
            
            #     plt.subplot(2, 3, 1)
            #     plt.title('Student_T_out')
            #     plt.imshow(np.transpose( np.clip( student_T_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
            #     plt.subplot(2, 3, 2)
            #     plt.title('Student_R_out')
            #     plt.imshow(np.transpose( np.clip( student_R_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
            #     plt.subplot(2, 3, 3)  
            #     plt.title('Teacher_R_out')
            #     plt.imshow(np.transpose( np.clip( teacher_R_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
            #     plt.subplot(2, 3, 4)        
            #     plt.title('Teacher_T_out')
            #     plt.imshow(np.transpose( np.clip( teacher_T_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
            #     plt.subplot(2, 3, 5)        
            #     plt.title('reflect_image')
            #     plt.imshow(np.transpose( np.clip( reflect_image.cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
            #     plt.show()
            
            #print(f'epoch {epoch_idx} batch {batch_idx} , fid loss: {(I_OUT_T_loss + I_T_loss + I_Tout_loss)}, dcp loss: {dcl_loss}, fea loss: {fea_loss} \n')  
            print(f'epoch {epoch_idx} batch {batch_idx}, RM loss: {(RM_Bot+RM_Top)}, recon_t: {Reconstruct_loss(student_T_out, clean_image)}, recon_r: {Reconstruct_loss(student_R_out, compoment_image)}, fid loss: {(I_OUT_T_loss + I_T_loss + I_Tout_loss)} \n') 

        # save the model every epoch
        overwrite_name = os.path.join(args.model_save_dir, "teacher_R.pkl")
        torch.save({'model': teacher_R.state_dict(),
                    'optimizer': optimizer_teacher_R.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)
        
        overwrite_name = os.path.join(args.model_save_dir, "teacher_T.pkl")
        torch.save({'model': teacher_T.state_dict(),
                    'optimizer': optimizer_teacher_T.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)
        
        overwrite_name = os.path.join(args.model_save_dir, "student.pkl")
        torch.save({'model': student.state_dict(),
                    'optimizer': optimizer_student.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)
    
        if epoch_idx % args.save_freq == 0:
            # torch.save({'model':teacher_R.state_dict(),
            #             'optimizer': optimizer_teacher_R.state_dict(),'epoch':epoch_idx}
            #             ,args.model_save_dir+"teacher_R_"+str(epoch_idx)+".pkl")
            # torch.save({'model':teacher_T.state_dict(),
            #             'optimizer': optimizer_teacher_T.state_dict(),'epoch':epoch_idx}
            #             ,args.model_save_dir+"teacher_T_"+str(epoch_idx)+".pkl")
            # torch.save({'model':student.state_dict(),
            #             'optimizer': optimizer_student.state_dict(),'epoch':epoch_idx}
            #             ,args.model_save_dir+"student_"+str(epoch_idx)+".pkl")      
            torch.save({'model':student.state_dict()}
                        ,args.model_save_dir+"student_"+str(epoch_idx)+".pkl")         
            
        print(f'epcoh {epoch_idx} train loss {train_loss}\n')
        writer.add_scalar('Train Loss', train_loss, epoch_idx)
    
    save_name = os.path.join(args.model_save_dir, 'student_Final.pkl')
    torch.save({'model': student.state_dict()}, save_name)    