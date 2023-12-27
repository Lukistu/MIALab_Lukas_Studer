"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using registered atlas labels.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
import pymia.filtering.filter as fltr
import mialab.filtering.preprocessing as fltr_prep

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


# DO WE HAVE TO LOAD ATLAS LABELS HERE?---------------------------------------------------------------------------------


def main(result_dir: str, data_atlas_labels_dir: str, data_test_dir: str):
    """Brain tissue segmentation using registered atlas labels.
    """

    # --LOAD ATLAS LABELS?--
    atlas_labels = {}
    for file in os.listdir(data_atlas_labels_dir):
        if file.endswith(".nii.gz"):
            atlas_labels[file.split("_")[-1][0]] = sitk.ReadImage(os.path.join(data_atlas_labels_dir, file))

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator(result_dir)

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params = {'skullstrip_pre': True, 'normalization_pre': True, 'registration_pre': False,
                          'coordinates_feature': True, 'intensity_feature': False, 'gradient_intensity_feature': False,
                          'training': False}
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        # --TRANSFORM ATLAS LABELS TO NATIVE SPACE--
        # Learn about transformations here: https://simpleitk.org/SPIE2019_COURSE/02_images_and_resampling.html
        transformed_labels = {}
        transform_matrix = img.transformation.GetInverse()
        for key in atlas_labels.keys():
            transformed_labels[key] = sitk.Resample(atlas_labels[key], img.images[structure.BrainImageTypes.T1w],
                                                    transform_matrix, sitk.sitkNearestNeighbor)

        # -- COMBINE TRANSFORMED ATLAS LABELS --
        # Learn about handling sitk image data here: https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
        label_image = sitk.Image(img.images[structure.BrainImageTypes.GroundTruth].GetSize(), sitk.sitkInt8)
        label_array = sitk.GetArrayFromImage(label_image)
        for key in atlas_labels.keys():
            atlas_label_image = sitk.GetArrayFromImage(transformed_labels[key])
            label_array[atlas_label_image >= 0.35] = key # try changing 0.5 to something else to check how well this works!

        label_image = sitk.GetImageFromArray(label_array)

        # --EVALUATE TRANSFORMED ATLAS LABELS--
        evaluator.evaluate(label_image, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    # use two writers to report the results
    result_file = os.path.join(result_dir, 'atlas_results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'atlas_results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


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
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas_labels_from_training')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_test_dir)
