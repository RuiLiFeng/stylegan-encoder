import copy
import dnnlib
from dnnlib import EasyDict

import config
from metrics import metric_base


if 1:
    desc          = 'lia'                                                                  # Description string included in result subdir name.
    train         = EasyDict(run_func_name='encoder.training_loop.training_loop')         # Options for training loop.
    G             = EasyDict(func_name='training.networks_stylegan.G_style')               # Options for generator network.
    E             = EasyDict(func_name='encoder.encoder.fc_encoder')                       # Options for discriminator network.
    D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # Options for discriminator network.
    E_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for discriminator optimizer.
    E_loss        = EasyDict(func_name='encoder.encoder.encoder_loss', beta=10.0)      # Options for discriminator loss.
    dataset       = EasyDict()                                                             # Options for load_dataset().
    sched         = EasyDict()                                                             # Options for TrainingSchedule.
    grid          = EasyDict(size='4k', layout='random')                                   # Options for setup_snapshot_image_grid().
    metrics       = [metric_base.fid50k]                                                   # Options for MetricGroup.
    submit_config = dnnlib.SubmitConfig()                                                  # Options for dnnlib.submit_run().
    tf_config     = {'rnd.np_random_seed': 1000}                                           # Options for tflib.init_tf().

    # Dataset.
    desc += '-ffhq';     dataset = EasyDict(tfrecord_dir='ffhq');                 train.mirror_augment = True
    #desc += '-ffhq512';  dataset = EasyDict(tfrecord_dir='ffhq', resolution=512); train.mirror_augment = True
    #desc += '-ffhq256';  dataset = EasyDict(tfrecord_dir='ffhq', resolution=256); train.mirror_augment = True
    #desc += '-celebahq'; dataset = EasyDict(tfrecord_dir='celebahq');             train.mirror_augment = True
    #desc += '-bedroom';  dataset = EasyDict(tfrecord_dir='lsun-bedroom-full');    train.mirror_augment = False
    #desc += '-car';      dataset = EasyDict(tfrecord_dir='lsun-car-512x384');     train.mirror_augment = False
    #desc += '-cat';      dataset = EasyDict(tfrecord_dir='lsun-cat-full');        train.mirror_augment = False

    # Number of GPUs.
    #desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
    #desc += '-2gpu'; submit_config.num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
    #desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 8; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

    # Default options.
    train.total_kimg = 25000
    sched.lod_initial_resolution = 1024
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

#----------------------------------------------------------------------------

def main():
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, E_args=E, E_opt_args=E_opt, E_loss_args=E_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
