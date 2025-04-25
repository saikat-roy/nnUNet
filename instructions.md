# Installation 

Download and install the abdomenatlas1.1 branch from the following repo

```
git clone https://github.com/saikat-roy/nnUNet.git nnUNet_mednext
cd nnUNet_mednext
git checkout abdomenatlas1.1
pip install .
```

__Note__: This is not yet an official release of nnUNet and is an _in-progress_ personal branch for a future integration of MedNeXt into the nnUNet main branch. The currently houses the code for 
AbdomenAtlas1.1 training and inference.

# Downloading the Model Files
After this custom branch of nnUNet has been installed and the `nnUNet_results` path has been set according to [installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) on the main nnUNet repo, please download the trained model checkpoint from -

```
https://zenodo.org/records/15282563
```
Unzip the contents into the `nnUNet_results` location that you should have already set while completing the nnUNet installation.

# Standard Inference
The inference script is very similar in principle to last time. However, the commands are slightly different but still simple. Just run the following for running inference on a folder with test data similar to last time:

```
cd <nnUNet_mednext folder>/nnunetv2/inference/

python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m $nnUNet_results/Dataset700_AbdomenAtlas1.1/nnUNetTrainer_MedNeXt_L_kernel3__nnUNetPlans__3d_fullres_mednext_8 --allowed_mirroring_axes 0 1 2
```
This will run with __full test-time augmentation__, and is typically the best performing but slowest.

## Reducing the degree of test-time augmentation for inference speed

As it was in for Touchstone, you can reduce the number of mirrored axis in test time augmentation for 
faster inference. The above uses 3 axes which is default. You can reduce it to 2 or 1 or none as follows:

### 2 axes
```
python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --allowed_mirroring_axes 0 1
```

### 1 axis
```
python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --allowed_mirroring_axes 0
```

### No Test Time augmentation (FASTEST)
```
python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --disable_tta
```

## Inference in multiple parts on Multiple GPUs
The above code assumes a single GPU for inference. However, if you have multiple gpus available, we 
can split the file list into parts and predict each on individual GPUs. For example, the following splits the entire file list into 4 parts and runs prediction for each part on a separate GPU.

```
CUDA_VISIBLE_DEVICES=0 python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --allowed_mirroring_axes 0 1 2 --num_parts 4 --part_id 0

CUDA_VISIBLE_DEVICES=1 python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --allowed_mirroring_axes 0 1 2 --num_parts 4 --part_id 1

CUDA_VISIBLE_DEVICES=2 python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --allowed_mirroring_axes 0 1 2 --num_parts 4 --part_id 2

CUDA_VISIBLE_DEVICES=3 python aa11_inference.py -i <input_path_to_test_data> -o <output_path_for_predictions> -m <model_folder_as_unzipped_earlier> --allowed_mirroring_axes 0 1 2 --num_parts 4 --part_id 3
```

__NOTE:__ This will make things faster in principle but might significantly increase RAM usage if all GPUs are on the same device, so please keep an eye on your system.


# Training details
Similar to Touchstone, the network was trained using the native Distributed Data Parallel (DDP) training supported by nnUNet on 8 A100 GPUs with a global batch size of $16$ (local batch size per GPU: $16/8 = 2$). The model was trained with a patch size of $128 \times 128 \times 128$ with an $1.0m \times 1.0mm \times 1.0mm$ isotropic spacing, with a learning rate of $1e-3$ using `AdamW` as the optimizer and nnUNet's standard data augmentation and learning rate scheduler.