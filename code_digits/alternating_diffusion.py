from diffusion_conditional_masked import *
from train_target import *
from train import test_acc

data_dict = {'mnist_m': 'MNIST_M', 'mnist': 'MNIST', 'svhn': 'SVHN','syn_digits':'SYN_DIGITS','usps':'USPS', 'sign':'SIGN','syn_sign':'SYN_SIGN', 'sign64':'SIGN64','syn_sign64':'SYN_SIGN64', 'mnist_m_synthetic':'MNIST_M_SYNTHETIC',
            'mnist_m_synthetic_epoch460':'MNIST_M_SYNTHETIC_EPOCH460', 'mnist_m_synthetic_pretrain':'MNIST_M_SYNTHETIC_PRETRAIN',\
             'mnist_m_synthetic_iterative':'MNIST_M_SYNTHETIC_ITERATIVE',}


class source_domain_numpy(data.Dataset):
  def __init__(self,root, root_list, transform=None, target_transform=None):  # last four are dummies
    self.transform = transform
    self.data_list = []
    self.label_list = []
    self.len_s = 20000
    root_list = root_list.split(',')
    for root_s in root_list:
        source_root = root_s.upper() #data_dict[root_s]
        
        data_source, labels_source = torch.load(os.path.join(root, root_s, source_root + '_train.pt'))
        l = data_source.shape[0]
        if(l >= self.len_s):
            choosen_index = np.random.choice(l,self.len_s,replace=False)
        else:
            print(f"Oversampling : {root_s}")
            choosen_index = np.random.choice(l,self.len_s,replace=True)
        self.data_list.append(data_source[choosen_index])
        self.label_list.append(labels_source[choosen_index].squeeze())
    self.domain_num = len(root_list)
    

  def inverse_data(self, data, labels):
      data = np.concatenate([data, 255 - data], axis=0)
      labels = np.concatenate([labels] * 2, axis=0)
      return data, labels

  def pre_prcess(self,img):
    if len(img.shape) == 2:
      img = np.concatenate([np.expand_dims(img,axis=2)]*3,axis=2)
    img = Image.fromarray(img)
    return img

  def __getitem__(self, index):
    index_data = np.random.choice(20000,1).item()
    chosen_d = np.random.choice(self.domain_num, 1).item()
    data_s = self.data_list[chosen_d][index_data]
    label_s = self.label_list[chosen_d][index_data]

    img_s = self.pre_prcess(data_s)

    if self.transform is not None:
      img_s = self.transform(img_s)
    return img_s, label_s, chosen_d

  def __len__(self):
    return self.len_s*self.domain_num

class domain_test_numpy(data.Dataset):
  def __init__(self, root,  root_t, transform=None):  # last four are dummies
      self.transform = transform
      domain_root = root_t.upper() #data_dict[root_t]
      self.len_t = 9000
      self.data_domain, self.labels_domain = torch.load(os.path.join(root, root_t, domain_root + '_test.pt'))
      l = self.data_domain.shape[0]
      choosen_index = np.random.choice(l, self.len_t, replace=False)

      self.data_domain = self.data_domain[choosen_index]
      self.labels_domain = self.labels_domain[choosen_index]
      self.len_t = self.labels_domain.shape[0]

  def pre_prcess(self, img):
      if len(img.shape) == 2:
        img = np.concatenate([np.expand_dims(img, axis=2)] * 3, axis=2)
      img = Image.fromarray(img)
      return img

  def __getitem__(self, index):
    img_t = self.pre_prcess(self.data_domain[index])
    if self.transform is not None:
      img_t = self.transform(img_t)
    return img_t, self.labels_domain[index].squeeze()

  def __len__(self):
    return self.len_t


def diffusion_train(alternate_step, ddpm, config):
    # optionally load a model
    batch_size = config["batch_size"]
    save_dir = "./data/{}/alt_{}".format(config["model_dir"],  alternate_step)
    if(not(os.path.exists(save_dir))):
        os.mkdir(save_dir)
    
    transforms_train = transforms.Compose([transforms.Resize(28),transforms.ToTensor()]) #,transforms.Normalize([0.5], [0.5])])
    dataset = source_domain_numpy(root="../data", root_list='mnist,mnist_m,svhn,syn_digits', transform=transforms_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    n_epoch = config['diffusion_epochs']
    load_path = "./data/{}/alt_{}/diffusion_model_{}.pth".format(config["model_dir"], alternate_step-1, config["diffusion_epochs"]-1)
    if('start_diffusion_from_scratch' in config and config['start_diffusion_from_scratch']):
        pass
    elif(alternate_step == 0):
        n_epoch = 150
        pass
    elif(os.path.exists(load_path)):
        ddpm.load_state_dict(torch.load(load_path, map_location=config["device"]))
    else:
        ddpm.load_state_dict(torch.load("./data/diffusion_outputs_masked_sameC/model_490.pth", map_location=config["device"]))

    if(alternate_step == 0 and "start_with_discriminator" in config and config["start_with_discriminator"]):
        discriminator = Discriminator().to(config["device"])
        discriminator.load_state_dict(torch.load("./data/{}/alt_{}/discriminator_{}.pth".format(config["model_dir"],  alternate_step-1, config["discriminator_epochs"]-1), map_location=config["device"]))        
    elif(alternate_step == 0):
        discriminator = None
    else:
        discriminator = Discriminator().to(config["device"])
        discriminator.load_state_dict(torch.load("./data/{}/alt_{}/discriminator_{}.pth".format(config["model_dir"],  alternate_step-1, config["discriminator_epochs"]-1), map_location=config["device"]))

    optim = torch.optim.Adam(ddpm.parameters(), lr=config['lrate'])
    
    for ep in range(n_epoch):
        print(f'alternate step {alternate_step}, epoch {ep}')
        ddpm.train()
        optim.param_groups[0]['lr'] = config['lrate']*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        diffusion_loss_ema = None
        alpha = 0.01
        for x, c, d in pbar:
            optim.zero_grad()
            x = x.to(config['device'])
            c = c.to(config['device'])
            d = d.to(config['device'])

            if(discriminator is None):
                c[d == config['target_domain_index']] = config['n_classes']-1
            else:
                pseudo_outputs = discriminator((x[d == config['target_domain_index']]-0.5)/(0.5))[2]
                _, pseudo_labels = torch.max(pseudo_outputs.data, 1)
                c[d == config['target_domain_index']] = pseudo_labels
            
            loss_pos, ts, noise = ddpm(x, c, d)
            diffusion_loss = loss_pos.mean()
            loss = diffusion_loss
            loss.backward()
            if diffusion_loss_ema is None:
                diffusion_loss_ema = diffusion_loss.item()
            else:
                diffusion_loss_ema = 0.95 * diffusion_loss_ema + 0.05 * diffusion_loss.item()
            pbar.set_description(f"diffusion_loss: {diffusion_loss_ema:.4f}")
            optim.step()
        ddpm.eval()
        
        #if save_model and (ep % 10 == 0 or ep == int(n_epoch-1)):
        #torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
        #print('saved model at ' + save_dir + f"model_{ep}.pth")
        torch.save(ddpm.state_dict(), os.path.join(save_dir, "diffusion_model_{}.pth".format(config["diffusion_epochs"]-1)))
    return ddpm
        
def diffusion_inference(alternate_step, ddpm, config):
    batch_size = 128
    n_classes = config['n_classes']
    n_sample = config['n_classes'] * batch_size 
    req_size = config["synthetic_size"]
    ddpm.eval()
    x_gen_all = []
    with torch.no_grad():
        w = 2
        print("Inference Iterations: {}".format(int(req_size/(n_sample - batch_size))))
        for itr in tqdm(range(int(req_size/(n_sample - batch_size)))):
            x_gen, _ = ddpm.sample_from_one_domain(n_sample, (config['in_channels'], 28, 28), target_domain = 1, device=config['device'], guide_w=np.random.uniform(0.5,2.5))
            x_gen_all.append(x_gen.detach().cpu().numpy())

    synthetic_images = np.vstack(x_gen_all)
    labels = torch.tensor([i for j in range(int(len(synthetic_images)/n_classes)) for i in range(n_classes) ])
    synthetic_images_filtered, labels_filtered = synthetic_images[labels != n_classes-1], labels[labels != n_classes-1]
    
    synthetic_images_filtered = synthetic_images_filtered.transpose(0, 2, 3, 1)
    def convert_numpy_to_PIL(img):
        #img = (img*-1)+1
        img = (img - img.min())/(img.max() - img.min())
        return Image.fromarray((img * 255).astype(np.uint8))
    synthetic_images_filtered_32 = np.stack(np.array([convert_numpy_to_PIL(img) for img in synthetic_images_filtered]))

    #inf_dir = "../data/{}/".format(config["model_dir"])
    #if(not(os.path.exists(inf_dir))):
    #    os.mkdir(inf_dir)
    root_s = "mnist_m_{}_{}".format(config["model_dir"], alternate_step)
    save_dir = os.path.join("../data", root_s)
    print(save_dir)
    if(not(os.path.exists(save_dir))):
        print("Making", os.path.join(save_dir))
        os.mkdir(os.path.join(save_dir))   
    save_file = "MNIST_M_{}_{}_train.pt".format(config["model_dir"].upper(), alternate_step)
    torch.save([synthetic_images_filtered_32, labels_filtered.cpu().detach().numpy()], os.path.join(save_dir, save_file))
    data_dict[root_s] = "MNIST_M_{}_{}".format(config["model_dir"].upper(), alternate_step)

def discriminator_train(alternate_step, config):
    n_epoch = config['discriminator_epochs']
    save_dir = "./data/{}/alt_{}".format(config["model_dir"],  alternate_step)
    if(not(os.path.exists(save_dir))):
        os.mkdir(save_dir)
    if("augmentation" in config and config["augmentation"] and alternate_step >= 5):
        transforms_train = transforms.Compose(
                    [
                    transforms.Resize(28),
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

    #if(alternate_step == 0):
    #    source_dataset = config['source_dataset']+","+ "mnist_m_synthetic_epoch460"
    #else:
    if(alternate_step >=0):
        source_dataset = config['source_dataset']+","+"mnist_m_{}_{}".format(config["model_dir"], alternate_step)
    else:
        source_dataset = config['source_dataset']
        
    data_set = source_domain_numpy(root=config['base_root'], root_list=source_dataset, transform=transforms_train)
    loaders = torch.utils.data.DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True,  worker_init_fn=np.random.seed,drop_last=True)
    
    test_set = domain_test_numpy(root= config['base_root'], root_t=config['target_dataset'], transform=transforms_test)
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
              #"model_dir": 'iteration_correctsyn_aug_midnight', 
              "model_dir": 'start_with_discriminator_May29_diffusionfromscratch', 
              'start_with_discriminator':True,
              #'start_diffusion_from_scratch':True,
              "diffusion_epochs": 60, #100, 
              "discriminator_epochs": 150, 
              'device':"cuda:0",
              'base_root':'../data', 
              'source_dataset': 'mnist,svhn,syn_digits',
              'target_dataset':'mnist_m',
              'batch_size':256, 
              'resolution':28,'num_workers':4, 
              'lrate' : 1e-4, 
              'target_domain_index' : 1,
              'synthetic_size': 21000,
             #'synthetic_size': 1420,
             'n_classes': 10 + 1,
             'n_domains':4,
             'in_channels': 3,
             'augmentation':True,
             'label_smoothing':True}
    
    seed = 0
    set_seed(seed)
    #n_epoch = 1000
    batch_size = 256
    n_T = 400 
    n_feat = 128
    
    ddpm = DDPM(nn_model=ContextUnet(in_channels=config['in_channels'], n_feat=n_feat, n_classes=config['n_classes'], n_domains=config['n_domains']), betas=(1e-4, 0.02), n_T=n_T, device=config["device"], drop_prob=0.1)
    ddpm.to(config["device"])


    if(not(os.path.exists("./data/{}".format(config["model_dir"])))):
        os.mkdir("./data/{}".format(config["model_dir"]))

    #ddpm.load_state_dict(torch.load("./data/diffusion_outputs_masked_sameC/model_490.pth", map_location=config["device"]))
    
    mixed_epochs = 5
    pure_epochs = 5

    old_source = config['source_dataset']
    #diffusion_inference(0, ddpm, config)
    if(config["start_with_discriminator"]):
        discriminator_train(-1, config)
        
    for alternate_step in range(mixed_epochs+1):
        
        ddpm = diffusion_train(alternate_step, ddpm, config)
        diffusion_inference(alternate_step, ddpm, config)

        if(alternate_step == mixed_epochs):
            config['source_dataset'] = 'mnist_m_{}_{},mnist_m_{}_{}'.format(config["model_dir"], alternate_step -2,config["model_dir"], alternate_step -1)

        print(config['source_dataset'])
        discriminator_train(alternate_step, config)
        

    for alternate_step in range(mixed_epochs+1,mixed_epochs+1+pure_epochs):
        config['source_dataset'] = old_source
        ddpm = diffusion_train(alternate_step, ddpm, config)
        diffusion_inference(alternate_step, ddpm, config)
        config['source_dataset'] = 'mnist_m_{}_{},mnist_m_{}_{}'.format(config["model_dir"], alternate_step -2,config["model_dir"], alternate_step -1)
        discriminator_train(alternate_step, config)




if __name__ == "__main__":
    run()






    

