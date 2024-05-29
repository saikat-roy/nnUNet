from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.architectures.MedNeXt.MedNeXt import MedNeXt
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_MedNeXt(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_of_mednext_ds_outputs = 5
        self.num_epochs = 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                        self.network.parameters(), 
                        self.initial_lr, 
                        weight_decay=self.weight_decay,            
                        eps=1e-4        # default value 1e-8 might cause nans in mixed precision
                        )
        
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        # mod.decoder.deep_supervision = enabled
        mod.do_ds = enabled

    def _get_deep_supervision_scales(self):
        """
        MedNeXt is a static architecture and has 4 ds outputs + 1 standard output if DS is enabled.
        [[1,1,1]]+[[2,2,2] for i in range(self.num_of_mednext_ds_outputs)] guarantees default DS behaviour 
        for a 5 layer architecture.
        """
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                [[1,1,1]]+[[2,2,2] for i in range(self.num_of_mednext_ds_outputs)],), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
    

class nnUNetTrainer_MedNeXt_k5(nnUNetTrainer_MedNeXt):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4


# MedNeXt trainers for kernel size 3x3x3
class nnUNetTrainer_MedNeXt_S_kernel3(nnUNetTrainer_MedNeXt):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = 2,
            kernel_size = 3,
            deep_supervision = enable_deep_supervision,
            do_res=True,
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            checkpoint_style = None,
            grn = False
        )
        return network


class nnUNetTrainer_MedNeXt_B_kernel3(nnUNetTrainer_MedNeXt):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = [2,3,4,4,4,4,4,3,2],
            kernel_size = 3,
            deep_supervision = enable_deep_supervision,
            do_res = True,
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            checkpoint_style = None,
            grn = False
        )
        return network


class nnUNetTrainer_MedNeXt_M_kernel3(nnUNetTrainer_MedNeXt):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = [2,3,4,4,4,4,4,3,2],
            kernel_size = 3,
            deep_supervision = enable_deep_supervision,
            do_res = True,
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block',
            grn = False
        )
        return network


class nnUNetTrainer_MedNeXt_L_kernel3(nnUNetTrainer_MedNeXt):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = [3,4,8,8,8,8,8,4,3],
            kernel_size = 3,
            deep_supervision = enable_deep_supervision,
            do_res = True,
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block',
            grn = False
        )
        return network
    
    
# MedNeXt trainers for kernel size 5x5x5
class nnUNetTrainer_MedNeXt_S_kernel5(nnUNetTrainer_MedNeXt_k5):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = 2,
            kernel_size = 5,
            deep_supervision = enable_deep_supervision,
            do_res=True,
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            checkpoint_style = None,
            grn = False
        )
        return network


class nnUNetTrainer_MedNeXt_B_kernel5(nnUNetTrainer_MedNeXt_k5):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = [2,3,4,4,4,4,4,3,2],
            kernel_size = 5,
            deep_supervision = enable_deep_supervision,
            do_res = True,
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            checkpoint_style = None,
            grn = False
        )
        return network


class nnUNetTrainer_MedNeXt_M_kernel5(nnUNetTrainer_MedNeXt_k5):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = [2,3,4,4,4,4,4,3,2],
            kernel_size = 5,
            deep_supervision = enable_deep_supervision,
            do_res = True,
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block',
            grn = False
        )
        return network


class nnUNetTrainer_MedNeXt_L_kernel5(nnUNetTrainer_MedNeXt_k5):

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
        ):
        
        network = MedNeXt(
            in_channels = num_input_channels,
            n_channels = 32,
            n_classes = num_output_channels,
            exp_r = [3,4,8,8,8,8,8,4,3],
            kernel_size = 5,
            deep_supervision = enable_deep_supervision,
            do_res = True,
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block',
            grn = False
        )
        return network