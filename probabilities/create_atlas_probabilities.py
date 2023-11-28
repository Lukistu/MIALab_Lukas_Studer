"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str):
    """Brain tissue segmentation using decision forests.
    The main routine executes the medical image analysis pipeline.
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    for image in images:
        sitk.WriteImage(image.images[structure.BrainImageTypes.GroundTruth], os.path.join(result_dir, image.id_ + '_atlas_gt.nii.gz'), True)

    for label_index in range(1, 6):
        if len(images) > 0:
            size = images[0].images[structure.BrainImageTypes.GroundTruth].GetSize()
            origin = images[0].images[structure.BrainImageTypes.GroundTruth].GetOrigin()
            spacing = images[0].images[structure.BrainImageTypes.GroundTruth].GetSpacing()
            direction = images[0].images[structure.BrainImageTypes.GroundTruth].GetDirection()

            output_image = sitk.Image(size, sitk.sitkFloat32)
            output_image.SetOrigin(origin)
            output_image.SetSpacing(spacing)
            output_image.SetDirection(direction)

            for image in images:
                label_ground_truth = image.images[structure.BrainImageTypes.GroundTruth] == label_index
                output_image = output_image + sitk.Cast(label_ground_truth, sitk.sitkFloat32)

            output_image = output_image / float(len(images))
            sitk.WriteImage(output_image,
                            os.path.join(result_dir, 'average_atlas_gt_' + str(label_index) + '.nii.gz'), True)

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir)
