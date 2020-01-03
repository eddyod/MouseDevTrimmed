import os
import argparse
from utilities2015 import execute_command, create_if_not_exists
from metadata import ROOT_DIR, MXNET_MODEL_ROOTDIR
#from data_manager import DataManager
from distributed_utilities import relative_to_local


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='This script downloads input data for demo.')

parser.add_argument("-d", "--demo_data_dir", type=str, help="Directory to store demo input data", default='demo_data')
args = parser.parse_args()

def download_to_demo(fp):
    create_if_not_exists(ROOT_DIR)
    #s3_http_prefix = 'https://s3-us-west-1.amazonaws.com/mousebrainatlas-data/'
    s3_http_prefix = 'https://s3-us-west-1.amazonaws.com/v0.2-required-data/'
    url = s3_http_prefix + fp
    demo_fp = os.path.join(ROOT_DIR, fp)
    execute_command('wget -N -P \"%s\" \"%s\"' % (os.path.dirname(demo_fp), url))
    return demo_fp


# Download raw JPEG2000 images
jpeg2000_files = ['MD662&661-F68-2017.06.06-07.39.27_MD661_2_0203',
    'MD662&661-F73-2017.06.06-09.53.20_MD661_1_0217',
    'MD662&661-F79-2017.06.06-11.52.28_MD661_1_0235',
    'MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250',
    'MD662&661-F89-2017.06.06-16.49.49_MD661_1_0265',
    'MD662&661-F94-2017.06.06-19.01.05_MD661_1_0280',
    'MD662&661-F99-2017.06.06-21.14.03_MD661_1_0295']
 
for img_name in jpeg2000_files:
    download_to_demo(os.path.join('jp2_files', 'DEMO998', img_name + '_lossless.jp2'))


# Download mxnet model

model_dir_name = 'inception-bn-blue'

fp = os.path.join(MXNET_MODEL_ROOTDIR, model_dir_name, 'inception-bn-blue-0000.params')
download_to_demo(relative_to_local(fp, local_root = ROOT_DIR))

fp = os.path.join(MXNET_MODEL_ROOTDIR, model_dir_name, 'inception-bn-blue-symbol.json')
download_to_demo(relative_to_local(fp, local_root = ROOT_DIR))

fp = os.path.join(MXNET_MODEL_ROOTDIR, model_dir_name, 'mean_224.npy')
download_to_demo(relative_to_local(fp, local_root = ROOT_DIR))

# Download warp/crop operation configs.
operation_configs = ['crop_orig_template',
    'from_aligned_to_none',
    'from_aligned_to_padded',
    'from_none_to_aligned_template',
    'from_none_to_padded',
    'from_none_to_wholeslice',
    'from_padded_to_brainstem_template',
    'from_padded_to_wholeslice_template',
    'from_padded_to_none',
    'from_wholeslice_to_brainstem']

for fn in operation_configs:
    download_to_demo(os.path.join('operation_configs', fn + '.ini'))

# Download brain meta data
print("Download brain DEMO998 meta data")
download_to_demo(os.path.join('brains_info', 'DEMO998.ini'))

download_to_demo(os.path.join('CSHL_data_processed', 'DEMO998', 'DEMO998_sorted_filenames.txt'))
download_to_demo(os.path.join('CSHL_data_processed', 'DEMO998', 'DEMO998_prep2_sectionLimits.ini'))

# Elastix intra-stack registration parameters
download_to_demo(os.path.join('elastix_parameters', 'Parameters_Rigid_MutualInfo_noNumberOfSpatialSamples_4000Iters.txt'))

# Rough global transform
download_to_demo(os.path.join('CSHL_simple_global_registration', 'DEMO998_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json'))
download_to_demo(os.path.join('CSHL_simple_global_registration', 'DEMO998_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.txt'))
