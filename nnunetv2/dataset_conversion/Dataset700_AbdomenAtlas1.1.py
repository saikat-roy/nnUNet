from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


if __name__ == '__main__':
    """
    How to train our submission to the JHU benchmark
    
    1. Execute this script here to convert the dataset into nnU-Net format. Adapt the paths to your system!
    2. Run planning and preprocessing: `nnUNetv2_plan_and_preprocess -d 224 -npfp 64 -np 64 -c 3d_fullres -pl 
    nnUNetPlannerResEncL_torchres`. Adapt the number of processes to your System (-np; -npfp)! Note that each process 
    will again spawn 4 threads for resampling. This custom planner replaces the nnU-Net default resampling scheme with 
    a torch-based implementation which is faster but less accurate. This is needed to satisfy the inference speed 
    constraints.
    3. Run training with `nnUNetv2_train 224 3d_fullres all -p nnUNetResEncUNetLPlans_torchres`. 24GB VRAM required, 
    training will take ~28-30h.
    """


    base = '/mnt/cluster-data-all/roys/raw_data/nnUNet_raw_data_base/AbdomenAtlas'
    cases = subdirs(base, join=False, prefix='BDMAP')

    target_dataset_id = 700
    target_dataset_name = f'Dataset{target_dataset_id:3.0f}_AbdomenAtlas1.1'

    # raw_dir = '/home/s539y-remote/E132-Rohdaten/nnUNetv2/'
    raw_dir = '/mnt/cluster-data-all/roys/raw_data//nnUNet_raw_data_base/nnUNet_raw_data/'
    maybe_mkdir_p(join(raw_dir, target_dataset_name))
    imagesTr = join(raw_dir, target_dataset_name, 'imagesTr')
    labelsTr = join(raw_dir, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    for case in cases:
        shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
        shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTr, case + '.nii.gz'))

    labels = {
        "background": 0,
        "aorta": 1,
        "gall_bladder": 2,
        "kidney_left": 3,
        "kidney_right": 4,
        "liver": 5,
        "pancreas": 6,
        "postcava": 7,
        "spleen": 8,
        "stomach": 9,
        "adrenal_gland_left": 10,
        "adrenal_gland_right": 11,
        "bladder": 12,
        "celiac_trunk": 13,
        "colon": 14,
        "duodenum": 15,
        "esophagus": 16,
        "femur_left": 17,
        "femur_right": 18,
        "hepatic_vessel": 19,
        "intestine": 20,
        "lung_left": 21,
        "lung_right": 22,
        "portal_vein_and_splenic_vein": 23,
        "prostate": 24,
        "rectum": 25,
    }

    generate_dataset_json(
        join(raw_dir, target_dataset_name),
        {0: 'CT'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        labels,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        converted_by = 'Saikat_Roy',
        license='CC BY-NC-SA 4.0',
        reference='https://huggingface.co/datasets/BodyMaps/AbdomenAtlasDataMini',
    )