from train import *

def Discriminator():
    cmd="--loss_type Twin_AC --AC \
    --AC_weight 1.0 \
    --G_shared \
    --n_domain 4 \
    --shuffle --batch_size 200 \
    --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000 \
    --num_D_steps 4 --num_G_steps 1 --G_lr 2e-4 --D_lr 2e-4 \
    --source_dataset mnist,mnist_m,svhn,syn_digits --target_dataset mnist_m --num_workers 16 \
    --G_ortho 0.0 \
    --G_attn 0 --D_attn 0 --G_ch 64 --D_ch 64 \
    --G_init N02 --D_init N02 \
    --test_every 8000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2019 \
    --ema  --use_ema --ema_start 10000"
    parser = utils.prepare_parser()
    config2 = vars(parser.parse_args(cmd.split()))
    config2['resolution'] = 32#utils.imsize_dict[config['dataset']]
    config2['n_classes'] = 10#utils.nclass_dict[config['dataset']]
    config2['G_activation'] = utils.activation_dict[config2['G_nl']]
    config2['D_activation'] = utils.activation_dict[config2['D_nl']]
    
    config2['skip_init'] = True
    config2 = utils.update_config_roots(config2)
    
    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config2['model'])
    experiment_name = (config2['experiment_name'] if config2['experiment_name']
                       else utils.name_from_config(config2))
    # Next, build the model
    #G = model.Generator(**config2).to(device)
    D = model.Discriminator(**config2)
    return D