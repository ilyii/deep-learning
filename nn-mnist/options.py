class Options:
    
    # General
    datapath = r"./data"
    savepath = r"./output"
    device = "cuda:0"
    seed = 0

    # Training
    n_epochs = 50
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    val_freq = 1
    ckpt_freq = 10

    # Model
    in_dim = 784 #28*28 
    hidden_dims = [100]
    out_dim = 10

    # Transform
    mean = 0.1307
    std = 0.3081