import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from data.dataloader_v2 import *
import lpips
from time import process_time
from torchvision.transforms import functional as F
def _test(student, args):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    batch_size = 1     
    student.eval();student.to(device)
    
    if torch.cuda.device_count() > 1:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank], output_device=args.local_rank)
    
    student.load_state_dict(torch.load(args.student_model)['model'], False)
    #test_dataset = datasets(os.path.join(args.data_dir,'test'),'input' , 'gt', transform )
    #test_dataloader = DataLoader(test_dataset, batch_size = batch_size , shuffle = False , num_workers=0 )   
    
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    total_sc = 0 
    for epoch in range(1):
        start = process_time()
        for batch_idx , (reflect_image , clean_image, name) in enumerate(dataloader):
            with torch.no_grad():
                print(name)  
                reflect_image = reflect_image.to(device)
                clean_image = clean_image.to(device)
                
                sout_share1,  sout_share2 , sout_share3 , sout_share4 , \
                sout_T1 , sout_T2 , sout_T3 , sout_T4 , \
                sout_R1 , sout_R2 , sout_R3 , sout_R4 , \
                student_T_out , student_R_out = student( reflect_image )  
                
                loss_fn = lpips.LPIPS(net='alex')
                d = loss_fn.forward(student_T_out.detach().cpu(),clean_image.cpu())            
                total_sc = total_sc + d
                
                pred_clip = torch.clamp(student_T_out, 0, 1)
                if args.save_image:
                    save_name = os.path.join(args.image_dir, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    pred.save(save_name)
                
                # plt.imsave( rf'./result/result_img/{batch_idx}.png' , np.transpose(
                #             np.clip(student_T_out.detach()[0,:,:,:].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
    
    #print( f'Average score of Lpips is {total_sc / len(test_dataloader)}' ) 