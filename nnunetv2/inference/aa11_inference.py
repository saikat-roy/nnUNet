from multiprocessing.pool import Pool
import nibabel as nib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from pathlib import Path


def convert_segmnetation_to_aa11_format(segmentation_file):
    mapping = {
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

    segmentation_file = Path(segmentation_file)
    dir_name = segmentation_file.parent/segmentation_file
    dir_name.mkdir(exist_ok=True, parents=True)
    seg_nifti = nib.load(segmentation_file.with_suffix(".nii.gz"))
    seg_data = seg_nifti.get_fdata()
    for k, v in mapping.items():
        class_seg = (seg_data == v).astype(np.uint8)
        class_nifti = nib.Nifti1Image(class_seg, seg_nifti.affine)
        nib.save(class_nifti, dir_name/f"{k}.nii.gz")
    (segmentation_file.with_suffix(".nii.gz")).unlink()


def aa11_inference_entry_point(allowed_mirroring_axes = (0, 1, 2), disable_tta=None):
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=("all",),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    if disable_tta is not None:
        args.disable_tta = disable_tta
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.allowed_mirroring_axes = allowed_mirroring_axes
    # why do some libraries not handle Path objects correctly?!
    input_files = [[str(p/"ct.nii.gz"),] for p in Path(args.i).iterdir() if (p/"ct.nii.gz").is_file()]
    output_files = [str(Path(args.o)/(p.name)) for p in Path(args.i).iterdir() if (p/"ct.nii.gz").is_file()]
    predictor.predict_from_files(input_files, output_files, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)
    with Pool(8) as p:
        p.map(convert_segmnetation_to_aa11_format, output_files)
    

def aa11_inference_entry_point_12():
    aa11_inference_entry_point(allowed_mirroring_axes=(0, 1))


def aa11_inference_entry_point_2():
    aa11_inference_entry_point(allowed_mirroring_axes=(1,))


def aa11_inference_entry_point_notta():
    aa11_inference_entry_point(disable_tta=True)


if __name__ == "__main__":
    aa11_inference_entry_point_notta()