class DefaultConfig(object):
    train_root = 'The path to trainout.h5'
    validation_root = 'The path to evalout.h5'
    lr = 0.0001
    batch_size = 16
    num_workers = 16
    epoch = 600
    
    outputs_dir = 'A folder for storing weights'
    
    cuda = True



opt = DefaultConfig()