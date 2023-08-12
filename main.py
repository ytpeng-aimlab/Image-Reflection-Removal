import os
import torch
import argparse
from train import _train
from test import _test
import Block_version_5

def main(args):
    
    torch.distributed.init_process_group(backend='nccl')
    if not os.path.exists('results/'):
        os.makedirs('results/')
    if not os.path.exists(args.model_save_dir): #save weight
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.image_dir): # save image
        os.makedirs(args.image_dir)
    
    teacher_T = Block_version_5.T_teacher_net()
    teacher_R = Block_version_5.R_teacher_net()
    student = Block_version_5.student_net()

    if torch.cuda.is_available():
        teacher_T.cuda()
        teacher_R.cuda()
        student.cuda()

    if args.mode == 'train':
        _train(teacher_T, teacher_R, student, args)
    elif args.mode == 'test':
        _test(student, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='train_0706_2', type=str)
    parser.add_argument('--mode', default='train', choices=['train','test'], type=str)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument("--local_rank", type=int)
    # Training Detail
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epoch', type=int, default=120)
    parser.add_argument('--bata1', type=int, default=0.9)
    parser.add_argument('--bata2', type=int, default=0.999)

    parser.add_argument('--learning_rate_TT', type=float, default=0.0001)
    parser.add_argument('--learning_rate_TR', type=float, default=0.0001)
    parser.add_argument('--learning_rate_S', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str, default='./dataset/Reflection')
    
    # Resume
    parser.add_argument('--resume_TT', type=str, default='')
    parser.add_argument('--resume_TR', type=str, default='')
    parser.add_argument('--resume_S', type=str, default='')
    
    # Frequent
    parser.add_argument('--showimg_freq',type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=1) 
    parser.add_argument('--valid_freq', type=int, default=100)
    
    # Test
    parser.add_argument('--student_model', type=str, default='./results/train_0706_2/weights/student_Final.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.name, 'weights/')
    args.image_dir = os.path.join('results/', args.name , 'nature/')

    print(args)
    main(args)