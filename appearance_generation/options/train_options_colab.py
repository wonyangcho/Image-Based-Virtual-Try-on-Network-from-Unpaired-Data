import easydict

class TrainOptionsColab():

    def __init__(self):
        self.parser = None
        self.initialized = False

    def initialize(self):
        
        self.parser = easydict.EasyDict({
                "name": 'vfr',                              #name of the experiment. It decides where to store samples and models'
                "gpu_ids": '0',                             #'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
                "checkpoints_dir": './checkpoints',         #'models are saved here'
                "model": "ov_pix2pixHD",                    #'which model to use'
                "norm": "instance",                         #'instance normalization or batch normalization'
                "use_dropout": False,                        #'use dropout for the generator'
                "data_type": 32,                            #"Supported data type i.e. 8, 16, 32 bit"
                "verbose": False,                            #'toggles verbose'
                "fp16": False,                              #'train with AMP'
                "local_rank": 0,                            #'local rank for distributed training
                
                # input/output sizes       
                
                "batchSize": 1,                             #'input batch size'
                "loadSize": 1024,                           #'scale images to this size'
                "fineSize": 512,                            #'then crop to this size'
                "label_nc": 20,                             #'# of input label channels'
                "input_nc": 20,                             #'# of input image channels'
                "densepose_nc": 3,                         #'# of denspose channels'
                "output_nc": 3,                            #'# of output image channels'    
                
                # for setting inputs

                "dataroot": "/home/dataset/",
                "resize_or_crop": "scale_width",            #'scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]'
                "serial_batches": False,                     #'if true, takes images in order to make batches, otherwise takes them randomly'
                "no_flip":False,                             #'if specified, do not flip the images for data argumentation'
                "nThreads": 2,                              #'# threads for loading data'
                "max_dataset_size": float("inf"),           #'Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.'

                # # for displays

                "display_winsize": 512,                     #'display window size'
                "tf_log": False,                            #'if specified, use tensorboard logging. Requires tensorflow installed'

                # for generator

                "netG": "global",                           #'selects model to use for netG'
                "ngf": 64,                                  #'# of gen filters in first conv layer'
                "n_downsample_global": 4,                   #'number of downsampling layers in netG'
                "n_blocks_global": 9,                       #'number of residual blocks in the global generator network'
                "n_blocks_local": 3,                        #'number of residual blocks in the local enhancer network
                "n_local_enhancers": 1,                     #'number of local enhancers to use'
                "niter_fix_global": 0,                      #'number of epochs that we only train the outmost local enhancer'

                # for instance-wise features

                "no_instance": False,                        #'if specified, do *not* add instance map as input'
                "instance_feat": False,                      #'if specified, add encoded instance features as input'
                "label_feat": False,                         #'if specified, add encoded label features as input'
                "feat_num": 30,                             #'vector length for encoded features    
                "load_features": False,                      #'if specified, load precomputed feature maps'
                "n_downsample_E": 4,                        #'# of downsampling layers in encoder'
                "nef": 16,                                  #'# of encoder filters in the first conv layer'
                "n_clusters": 10,                           #'number of clusters for features'

                # for displays

                "display_freq": 100,                        #'frequency of showing training results on screen'
                "print_freq": 100,                          #'frequency of showing training results on console'
                "save_latest_freq": 1000,                   #'frequency of saving the latest results'
                "save_epoch_freq": 5,                       #'frequency of saving checkpoints at the end of epochs'
                "no_html": False,                            #'do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/'
                "debug": False,                              #'only do one epoch and displays at each iteration'

                # for training

                "continue_train": False,                     #'continue training: load the latest model'
                "load_pretrain": "",                        #'load the pretrained model from the specified location'
                "which_epoch": "latest",                    #'which epoch to load? set to latest to use latest cached model'
                "phase": "train",                           #'train, val, test, etc'
                "niter": 100,                               #'# of iter at starting learning rate'
                "niter_decay": 100,                         #'# of iter to linearly decay learning rate to zero'
                "beta1": 0.5,                               #'momentum term of adam'
                "lr": 0.0002,                               #'initial learning rate for adam'


                # for discriminators 

                "num_D": 2,                                 #'number of discriminators to use'
                "n_layers_D": 3,                            #'only used if which_model_netD==n_layers'        
                "ndf": 64,                                  #'# of discrim filters in first conv layer'
                "lambda_feat": 10.0,                        #'weight for feature matching loss'
                "no_ganFeat_loss": False,                    #'if specified, do *not* use discriminator feature matching loss'
                "no_vgg_loss": False,                        #'if specified, do *not* use VGG feature matching loss'
                "no_lsgan": False,                           #'do *not* use least square GAN, if false, use vanilla GAN'
                "pool_size": 0,                             #'the size of image buffer that stores previously generated images'
                "no_ce_loss": False,                         #'if specified, do *not* use ce matching loss'

                "isTrain": False                         
                 
        
        })
        self.initialized = True
