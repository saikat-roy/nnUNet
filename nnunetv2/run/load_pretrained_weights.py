import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import warnings


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
    else:
        saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


def load_pretrained_weights_upkern(network, fname):

    print("################### Resampled Loading pretrained weights from file ", fname, '###################')
    
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match. # Fabian wrote this.
    new_state_dict = {}
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict
    model_dict = network.state_dict()
    
    for k in model_dict.keys():
        # print(k, model_dict[k].shape, pretrained_dict[k].shape)

        if k in model_dict.keys() and k in pretrained_dict.keys():  # Common keys
            if 'bias' in k or 'norm' in k or 'dummy' in k:
                print(f"Key {k} loaded unchanged.")
                model_dict[k] = pretrained_dict[k]
            else:
                inc1, outc1, *spatial_dims1 = model_dict[k].shape
                inc2, outc2, *spatial_dims2 = pretrained_dict[k].shape
                print(inc1, outc1, spatial_dims1, inc2, outc2, spatial_dims2)

                assert inc1==inc2 # Please use equal in_channels in all layers for resizing pretrainer
                assert outc1 == outc2 # Please use equal out_channels in all layers for resizing pretrainer
                
                if spatial_dims1 == spatial_dims2:
                    model_dict[k] = pretrained_dict[k]
                    print(f"Key {k} loaded.")
                else:
                    if len(spatial_dims1)==3:
                        model_dict[k] = torch.nn.functional.interpolate(
                                                pretrained_dict[k], size=spatial_dims1,
                                                mode='trilinear'
                                                )
                        print(f"Key {k} interpolated trilinearly from {spatial_dims2}->{spatial_dims1} and loaded.")
                    elif len(spatial_dims1)==2:
                        model_dict[k] = torch.nn.functional.interpolate(
                                                pretrained_dict[k], size=spatial_dims1,
                                                mode='bilinear'
                                                )
                        print(f"Key {k} interpolated bilinearly from {spatial_dims2}->{spatial_dims1} and loaded.")
                    else:
                        raise TypeError('UpKern only supports 2D and 3D shapes.')
        else:   # Keys which are not shared
            warnings.warn(f"Key {k} in current_model:{k in model_dict.keys()} and pretrained_model:{k in pretrained_dict.keys()} and will not be loaded.")

    network.load_state_dict(model_dict)
    print("######## Weight Loading DONE ############")
