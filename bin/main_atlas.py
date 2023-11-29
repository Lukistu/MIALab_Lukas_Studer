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


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using registered atlas labels.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Evaluation of the segmentation
    """

    # LOAD ATLAS LABELS?------------------------------------------------------------------------------------------------
    putil.load_atlas_images(data_atlas_dir)

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)


    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        # ---------------------------------------------NEW-------------------------------------------------------------
        id_: str
        paths: dict

        # load atlas labels
        path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
        path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
        img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
        transform = sitk.ReadTransform(path_to_transform)
        atlas_labels = structure.BrainImage(id_, path, img, transform)

        # construct pipeline for atlas label pre-processing
        pipeline_atlas_label = fltr.FilterPipeline()
        atlas_registration_pre = True
        if atlas_registration_pre:
            pipeline_atlas_label.add_filter(fltr_prep.ImageRegistration())
            pipeline_atlas_label.set_param(fltr_prep.ImageRegistrationParameters(atlas_labels, img.transformation),
                                           len(pipeline_atlas_label.filters) - 1)

        # execute pipeline on the atlas label
        img.images[structure.BrainImageTypes.AtlasLabels] = pipeline_atlas_label.execute(
            img.images[structure.BrainImageTypes.AtlasLabels])

        transformed_atlas_labels = img

        # ---------------------------------------------NEW-------------------------------------------------------------

        # evaluate segmentation without post-processing
        evaluator.evaluate(transformed_atlas_labels, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
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
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
