from diffusion_conditional_masked import *
from train_target import *
from train import test_acc


data_dict = {'mnist_m': 'MNIST_M', 'mnist': 'MNIST', 'svhn': 'SVHN','syn_digits':'SYN_DIGITS','usps':'USPS', 'sign':'SIGN','syn_sign':'SYN_SIGN', 'sign64':'SIGN64','syn_sign64':'SYN_SIGN64', 'mnist_m_synthetic':'MNIST_M_SYNTHETIC',
            'mnist_m_synthetic_epoch460':'MNIST_M_SYNTHETIC_EPOCH460', 'mnist_m_synthetic_pretrain':'MNIST_M_SYNTHETIC_PRETRAIN',\
             'mnist_m_synthetic_iterative':'MNIST_M_SYNTHETIC_ITERATIVE',}
import alternating_diffusion as alternating_diffusion


def discriminator_train(alternate_step, config):
    n_epoch = config['discriminator_epochs']
    save_dir = "./data/{}/alt_{}".format(config["model_dir"],  alternate_step)
    if(not(os.path.exists(save_dir))):
        os.mkdir(save_dir)

    if("augmentation" in config and config["augmentation"]):
        transforms_train = transforms.Compose(
                    [
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomInvert(p=0.5),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[train_mean], std=[train_std]),
                    transforms.Normalize([0.5], [0.5])
                    ])
    else:
        transforms_train = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    transforms_test = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    if('ignore_synthetic' in config and config['ignore_synthetic']):
        source_dataset = config['source_dataset']
    elif(alternate_step == 0):
        source_dataset = config['source_dataset']+","+ "mnist_m_synthetic_epoch460"
    else:
        source_dataset = config['source_dataset']+","+"mnist_m_{}_{}".format(config["model_dir"], alternate_step)


    data_set = alternating_diffusion.source_domain_numpy(root=config['base_root'], root_list=source_dataset, transform=transforms_train)
    loaders = torch.utils.data.DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True,  worker_init_fn=np.random.seed,drop_last=True)
    
    test_set = alternating_diffusion.domain_test_numpy(root= config['base_root'], root_t=config['target_dataset'], transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, worker_init_fn=np.random.seed, drop_last=True)

    print(len(data_set), len(test_set))

    run = wandb.init(
        project="pgm_project", 
        name=f"{config['model_dir']}_{alternate_step}",
        job_type="Train", 
        config=config,
    )

    model = Discriminator().to(config["device"])

    if('label_smoothing' in config and config["label_smoothing"]):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(n_epoch):
        print("Starting")
        loss, accuracy = train(model, criterion, optimizer, loaders)
        print("Train: {} - Loss: {}, Accuracy: {}".format(epoch, loss, accuracy))

        val_loss, val_accuracy, topk_val_accuracy = val(model, criterion, optimizer, test_loader)
        print("Test: {} - Loss: {}, Accuracy: {}, Topk: {}".format(epoch, val_loss, val_accuracy, topk_val_accuracy))

        wandb_metric = {'epoch':epoch, 'loss':loss, 'accuracy':accuracy,'val_accuracy':val_accuracy}
        wandb.log(wandb_metric)
        if((epoch+1) % 10 == 0 or epoch == int(n_epoch-1)):
            torch.save(model.state_dict(), os.path.join(save_dir, f"discriminator_{epoch}.pth"))
            #torch.save(model.state_dict(), os.path.join("models", run.name, f"epoch_{epoch}.pt"))
    wandb.finish()


def run():
    config = {'batch_size':256, 
              #"model_dir": 'baseline_mn,sv,syn', 
              #"model_dir": 'baseline_gold', 
              #"model_dir": 'baseline_mnistm_from_correctsyn_aug_ls', #'baseline_mnistm_from_diffusion', 
              "model_dir": 'baseline_mnistm_from_correctsyn_4_aug_ls',
              "diffusion_epochs": 60, 
              "discriminator_epochs": 150, 
              'device':"cuda:0",
              'base_root':'../data', 
              #'source_dataset': 'mnist_m_iteration_test_3,mnist_m_iteration_test_4',
              #'source_dataset': 'mnist_m_iteration_correctsyn_2,mnist_m_iteration_correctsyn_3,mnist_m_iteration_correctsyn_4',
              'source_dataset': 'mnist_m_iteration_correctsyn_4',
              #'source_dataset': 'mnist_m',
              'target_dataset':'mnist_m',
              'batch_size':256, 
              'resolution':28,'num_workers':4, 
              'lrate' : 1e-4, 'target_domain_index' : 1,
              #'synthetic_size': 21000,
             'synthetic_size': 22000,
             'n_classes': 10 + 1,
             'n_domains':4,
             'in_channels': 3,
             'ignore_synthetic': True,
             'augmentation':True,
             'label_smoothing': True}
    
    seed = 0
    set_seed(seed)

    for ds in config["source_dataset"].split(","):
        alternating_diffusion.data_dict[ds] = ds.upper()

    if(not(os.path.exists("./data/{}".format(config["model_dir"])))):
        os.mkdir("./data/{}".format(config["model_dir"]))
    discriminator_train(0, config)


if __name__ == "__main__":
    run()






    

