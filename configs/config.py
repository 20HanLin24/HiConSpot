from yacs.config import CfgNode as CN
import os
import torch

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))   
GPU_NUM = torch.cuda.device_count()   


_C = CN()

_C.fp16 = True                  
_C.epochs = 100                  
_C.step_epoch = 15               
_C.save_freq = 50                
_C.pretrain = ''                 
_C.work_dir = ''                 
_C.step_dist = 0.014             

 
_C.data = CN()
_C.data.train = CN()
_C.data.train.data_root = "{}/datasets/Floor_Plan_CAD/cadt".format(ROOT)   
_C.data.train.repeat = 2                         
_C.data.train.split = "train"
_C.data.train.data_norm = "min"
 
_C.data.train.aug = CN()
_C.data.train.aug.aug_prob = 0.5                 
_C.data.train.aug.hflip = True                   
_C.data.train.aug.vflip = True                   
_C.data.train.aug.rotate = CN()
_C.data.train.aug.rotate.enable = False          
_C.data.train.aug.rotate.angle = [-180, 180]     
_C.data.train.aug.rotate2 = True                 
_C.data.train.aug.scale = CN()
_C.data.train.aug.scale.enable = True            
_C.data.train.aug.scale.ratio = [0.5, 1.5]       
_C.data.train.aug.shift = CN()
_C.data.train.aug.shift.enable = True            
_C.data.train.aug.shift.scale = [-0.5, 0.5]      
_C.data.train.aug.cutmix = CN()
_C.data.train.aug.cutmix.enable = True                  
_C.data.train.aug.cutmix.queueK = 32                    
_C.data.train.aug.cutmix.relative_shift = [-0.5, 0.5]   
 
_C.data.test = CN()
_C.data.test.data_root = "{}/datasets/Floor_Plan_CAD/cadt".format(ROOT)   
_C.data.test.repeat = 1
_C.data.test.split = "test"
_C.data.test.data_norm = "mean"
_C.data.test.aug = False

 
_C.dataloader = CN()
_C.dataloader.train = CN()
_C.dataloader.test = CN()
_C.dataloader.train.batch_size = 4
_C.dataloader.train.num_workers = 3
_C.dataloader.test.batch_size = 1
_C.dataloader.test.num_workers = 2

 
_C.optimizer = CN()              
_C.optimizer.type = "AdamW"      
_C.optimizer.lr = 0.0001         
_C.optimizer.weight_decay = 0.0001           
_C.optimizer.weight_decay_embed = 0.0        
_C.optimizer.decoder_multiplier = 1.0        
_C.optimizer.clip_gradients_enabled = True        
_C.optimizer.clip_gradients_type = "full_model"   
_C.optimizer.clip_gradients_norm_type = 2.0       
_C.optimizer.clip_gradients_value = 0.01          


 
_C.matcher = CN()
_C.matcher.num_classes = 35
_C.matcher.cost_class = 2.                   
_C.matcher.cost_mask = 5.                    
_C.matcher.cost_dice = 5.                    
_C.matcher.num_points = -1                   
_C.matcher.Disentangled_Matching = True      
_C.matcher.DM_weight = None                  
_C.matcher.alpha = 0.25                      
_C.matcher.gamma = 4.0                       

 
_C.criterion = CN()
_C.criterion.num_classes = 35               
_C.criterion.eos_coef = 0.1                 
_C.criterion.losses = ["labels", "masks"]   
_C.criterion.ignore_label = -1              
_C.criterion.class_weights = -1             
_C.criterion.num_points = -1                


 
_C.model = CN()
 
_C.model.in_channels = 12                                                                   
_C.model.semantic_classes = 35                                                              
_C.model.enc_depths = (2, 2, 2, 6, 2)
_C.model.enc_channels = (32, 64, 128, 256, 512)
_C.model.dec_channels = (64, 64, 128, 256)
_C.model.stride = (2, 2, 4, 4)
_C.model.patch_size = 256
_C.model.aggregation_mode = 'add'
_C.model.pool_mode = 'default'                                                              
_C.model.en_CFE = False                                                                     
_C.model.cfg_gnn_en = False                                                                 
_C.model.cfg_gnn_dim = 8                                                                    
 
_C.model.Embeding_backbone = "hrnet48"                                                       
_C.model.DOPRETRAINED = False                                                                
_C.model.PRETRAINED = f'{ROOT}/ModulePretrained/hrnetv2_w48_imagenet_pretrained.pth'         
_C.model.embed_interdim = 384
 
_C.model.num_decoders = 3                                                                   
_C.model.dropout = 0.0                                                                      
_C.model.pre_norm = False                                                                   
_C.model.num_heads = 8                                                                      
_C.model.shared_decoder = True                                                              
_C.model.dim_feedforward = 512                                                              
_C.model.hidden_dim = 256                                                                   
_C.model.gauss_scale = 1.0                                                                  
_C.model.normalize_pos_enc = False                                                          
_C.model.isolation_mode = "half"
_C.model.isolation_attn = True

 
_C.model.scalar = 1
_C.model.dn_mask_noise_scale = 0.0
_C.model.dn_label_noise_ratio = 0.2

_C.model.EXTRA = CN(new_allowed=True)                                                       
 
 

 
 
_C.model.NAME = 'Pointtransformer'                                                          
_C.model.enc_nsample = [16, 16, 16, 16, 16]                                                 


 
_C.root = "{}/datasets/Floor_Plan_CAD/cadt".format(ROOT)   

 
_C.WORKERS = 0           
_C.filter_num = 16       
_C.max_prim = 20000      
_C.seed = 2000           
_C.do_aug = True        


def update_config(cfg, args):
    cfg.defrost()   

    if hasattr(args, "opts") and args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "backbone_cfg") and args.backbone_cfg is not None:
        print(f"reading {cfg.model.Embeding_backbone} backbone cfg parameters")
        cfg.merge_from_file(args.backbone_cfg)  
 
    cfg.freeze()   

    if args.data_root is not None:
        cfg.defrost()
        cfg.data.train.data_root = args.data_root
        cfg.data.test.data_root = args.data_root
        cfg.root = args.data_root
    if hasattr(args, "cls_num") and args.cls_num is not None:
        cfg.defrost()
        cfg.model.semantic_classes = args.cls_num
        cfg.criterion.num_classes = args.cls_num
        cfg.criterion.contrast.num_classes = args.cls_num + 1

    if hasattr(args, "filter_num") and args.filter_num is not None:
        cfg.defrost()
        cfg.filter_num = args.filter_num

    if hasattr(args, "eos_coef") and args.eos_coef is not None:
        cfg.defrost()
        cfg.criterion.eos_coef = args.eos_coef

    if hasattr(args, "enc_nsample") and args.enc_nsample is not None:
        cfg.defrost()
        cfg.model.enc_nsample = args.enc_nsample

    if hasattr(args, "enc_stride") and args.enc_stride is not None:
        cfg.defrost()
        cfg.model.enc_stride = args.enc_stride

    if hasattr(args, "epochs") and args.epochs is not None:
        cfg.defrost()
        cfg.epochs = args.epochs

    if hasattr(args, "max_prim") and args.max_prim is not None:
        cfg.defrost()
        cfg.max_prim = args.max_prim

    if hasattr(args, "work_dir") and args.work_dir is not None:
        cfg.defrost()
        cfg.work_dir = args.work_dir
         

    if hasattr(args, "load_ckpt") and args.load_ckpt is not None:
        cfg.defrost()
        cfg.load_ckpt = args.load_ckpt
    if hasattr(args, "resume_ckpt") and args.resume_ckpt is not None:
        cfg.defrost()
        cfg.resume_ckpt = args.resume_ckpt

    if hasattr(args, "save_step") and args.save_step is not None:
        cfg.defrost()
        cfg.save_freq = args.save_step

    if hasattr(args, "lr_rate") and args.lr_rate is not None:
        cfg.defrost()
        cfg.optimizer.lr = args.lr_rate
    if hasattr(args, "input_dim") and args.input_dim is not None:
        cfg.defrost()
        cfg.model.in_channels = args.input_dim
    if hasattr(args, "enc_stride") and args.fp16 is not None:
        cfg.defrost()
        cfg.fp16 = args.fp16

    if hasattr(args, "matcher_cost_class") and args.matcher_cost_class is not None:
        cfg.defrost()
        cfg.matcher.cost_class = args.matcher_cost_class   
    if hasattr(args, "matcher_cost_mask") and args.matcher_cost_mask is not None:
        cfg.defrost()
        cfg.matcher.cost_mask = args.matcher_cost_mask   
    if hasattr(args, "matcher_cost_dice") and args.matcher_cost_dice is not None:
        cfg.defrost()
        cfg.matcher.cost_dice = args.matcher_cost_dice   

    if hasattr(args, "train_batch_size") and args.train_batch_size is not None:
        cfg.defrost()
        cfg.dataloader.train.batch_size = args.train_batch_size


    cfg.freeze()   
    return cfg


if __name__ == "__main__":
    print("start:")
    print(GPU_NUM)








