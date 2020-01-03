import sys
import os
from skimage.io import imread
import pickle

from pandas import read_hdf
import numpy as np

#sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import load_hdf, load_ini, one_liner_to_arr
from metadata import all_stacks, prep_str_to_id_2d, ROOT_DIR, RAW_DATA_DIR, THUMBNAIL_DATA_DIR, VOLUME_ROOTDIR, prep_id_to_str_2d

import bloscpack as bp
use_image_cache = False
image_cache = {}
metadata_cache = {}

"""
methods used in download_demo_data
1. get_sorted_filenames_filename
2. get_anchor_filename_filename
3. load_data
4. get_section_limits_filename_v2
5. get_cropbox_filename_v2
6. get_score_volume_filepath_v3
7. get_score_volume_origin_filepath_v3
8. get_image_filepath_v2
9. get_original_volume_filepath_v2
10. get_original_volume_origin_filepath_v3
"""

class DataManager(object):

    ################################################
    ##   Conversion between coordinate systems    ##
    ################################################

    ########################################################
    @staticmethod
    def get_sorted_filenames_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_sorted_filenames.txt')
        return fn
    
    @staticmethod
    def get_anchor_filename_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_anchor.txt')
        return fn
    
    @staticmethod
    def get_anchor_filename_filename_v2(stack):
        fp = os.path.join(ROOT_DIR, 'CSHL_data_processed', stack, 'operation_configs', 'from_none_to_aligned.ini')
        return fp    
    
    @staticmethod
    def get_section_limits_filename_v2(stack, anchor_fn=None, prep_id=2):
        """
        Return path to file that specified the cropping box of the given crop specifier.

        Args:
            prep_id (int or str): 2D frame specifier
        """

        if isinstance(prep_id, str):
            prep_id = prep_str_to_id_2d[prep_id]

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)

        fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_prep' + str(prep_id) + '_sectionLimits.json')
        if not os.path.exists(fp):
            fp = os.path.join(ROOT_DIR, stack, stack + '_prep' + str(prep_id) + '_sectionLimits.ini')

        return fp
    
    @staticmethod
    def get_cropbox_filename_v2(stack, anchor_fn=None, prep_id=2):
        """
        Return path to file that specified the cropping box of the given crop specifier.

        Args:
            prep_id (int or str): 2D frame specifier
        """

        if isinstance(prep_id, str) or isinstance(prep_id, type(str)):
            prep_id = prep_str_to_id_2d[prep_id]

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)

        #fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_prep' + str(prep_id) + '_cropbox.json')
        fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_cropbox.ini')
        return fp

    @staticmethod
    def get_image_dimension(stack, prep_id='alignedBrainstemCrop'):
        """
        Returns the dimensions at raw resolution for the alignedBrainstemCrop images.
        
        Returns:
            (raw image width, raw image height)
        """

        # first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
        # anchor_fn = DataManager.load_anchor_filename(stack)
        # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)

        xmin, xmax, ymin, ymax = DataManager.load_cropbox_v2(stack=stack, prep_id=prep_id)
        return (xmax - xmin + 1) * 32, (ymax - ymin + 1) * 32

    @staticmethod
    def get_original_volume_basename_v2(stack_spec):
        """
        Args:
            stack_spec (dict):
                - prep_id
                - detector_id
                - vol_type
                - structure (str or list)
                - name
                - resolution
        """

        if 'prep_id' in stack_spec:
            prep_id = stack_spec['prep_id']
        else:
            prep_id = None

        if 'detector_id' in stack_spec:
            detector_id = stack_spec['detector_id']
        else:
            detector_id = None

        if 'vol_type' in stack_spec:
            volume_type = stack_spec['vol_type']
        else:
            volume_type = None

        if 'structure' in stack_spec:
            structure = stack_spec['structure']
        else:
            structure = None

        assert 'name' in stack_spec, stack_spec
        stack = stack_spec['name']

        if 'resolution' in stack_spec:
            resolution = stack_spec['resolution']
        else:
            resolution = None

        components = []
        if prep_id is not None:
            if isinstance(prep_id, str):
                components.append(prep_id)
            elif isinstance(prep_id, int):
                components.append('prep%(prep)d' % {'prep':prep_id})
        if detector_id is not None:
            components.append('detector%(detector_id)d' % {'detector_id':detector_id})
        if resolution is not None:
            components.append(resolution)

        tmp_str = '_'.join(components)
        basename = '%(stack)s_%(tmp_str)s%(volstr)s' % \
            {'stack':stack, 'tmp_str': (tmp_str+'_') if tmp_str != '' else '', 'volstr':volume_type_to_str(volume_type)}
        if structure is not None:
            if isinstance(structure, str):
                basename += '_' + structure
            elif isinstance(structure, list):
                basename += '_' + '_'.join(sorted(structure))
            else:
                raise

        return basename


    @staticmethod
    def get_score_volume_filepath_v3(stack_spec, structure):
        basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        vol_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                              '%(basename)s',
                              'score_volumes',
                             '%(basename)s_%(struct)s.bp') % \
        {'stack':stack_spec['name'], 'basename':basename, 'struct':structure}
        return vol_fp


    @staticmethod
    def get_score_volume_origin_filepath_v3(stack_spec, structure, wrt='wholebrain'):

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                          'score_volumes',
                         '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
        {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}
        return fp

    @staticmethod
    def get_original_volume_origin_filepath_v3(stack_spec, structure, wrt='wholebrain', resolution=None):

        volume_type = stack_spec['vol_type']

        if 'resolution' not in stack_spec or stack_spec['resolution'] is None:
            assert resolution is not None
            stack_spec['resolution'] = resolution

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        if volume_type == 'score' or volume_type == 'annotationAsScore':
            origin_fp = os.path.join(VOLUME_ROOTDIR, '%(stack)s',
                          '%(basename)s',
                          'score_volumes',
                         '%(basename)s_%(struct)s_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt') % \
            {'stack':stack_spec['name'], 'basename':vol_basename, 'struct':structure}

        elif volume_type == 'intensity':
            origin_fp = os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, vol_basename + '_origin' + ('_wrt_'+wrt if wrt is not None else '') + '.txt')
        else:
            raise Exception("vol_type of %s is not recognized." % stack_spec['vol_type'])

        return origin_fp


    @staticmethod
    def get_original_volume_filepath_v2(stack_spec, structure=None, resolution=None):
        """
        Args:
            stack_spec (dict): keys are:
                                - name
                                - resolution
                                - prep_id (optional)
                                - detector_id (optional)
                                - structure (optional)
                                - vol_type
        """

        if 'resolution' not in stack_spec or stack_spec['resolution'] is None:
            assert resolution is not None
            stack_spec['resolution'] = resolution

        if 'structure' not in stack_spec or stack_spec['structure'] is None:
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec)
        else:
            stack_spec_no_structure = stack_spec.copy()
            stack_spec_no_structure['structure'] = None
            vol_basename = DataManager.get_original_volume_basename_v2(stack_spec=stack_spec_no_structure)

        vol_basename_with_structure_suffix = vol_basename + ('_' + structure) if structure is not None else ''

        if stack_spec['vol_type'] == 'score':
            return os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '.bp')
        elif stack_spec['vol_type'] == 'annotationAsScore':
            return os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '.bp')
        elif stack_spec['vol_type'] == 'intensity':
            return os.path.join(VOLUME_ROOTDIR, stack_spec['name'], vol_basename, vol_basename + '.bp')
        else:
            raise Exception("vol_type of %s is not recognized." % stack_spec['vol_type'])


    @staticmethod
    def get_image_filepath_v2(stack, prep_id, version=None, resol=None,
                           data_dir=ROOT_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR,
                           section=None, fn=None, ext=None, sorted_filenames_fp=None):
        """
        Args:
            version (str): the version string.

        Returns:
            Absolute path of the image file.
        """

#         if resol == 'lossless':
#             if stack == 'CHATM2' or stack == 'CHATM3':
#                 resol = 'raw'
#         elif resol == 'raw':
#             if stack not in ['CHATM2', 'CHATM3', 'MD661', 'DEMO999']:
#                 resol = 'lossless'

        if section is not None:

            if sorted_filenames_fp is not None:
                _, sections_to_filenames = DataManager.load_sorted_filenames(fp=sorted_filenames_fp)
                if section not in sections_to_filenames:
                    raise Exception('Section %d is not specified in sorted list.' % section)
                fn = sections_to_filenames[section]
            else:
                if section not in metadata_cache['sections_to_filenames'][stack]:
                    raise Exception('Section %d is not specified in sorted list.' % section)
                fn = metadata_cache['sections_to_filenames'][stack][section]

            if is_invalid(fn=fn, stack=stack):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None


        if prep_id is not None and (isinstance(prep_id, str) or isinstance(prep_id)):
            if prep_id == 'None':
                prep_id = None
            else:
                prep_id = prep_str_to_id_2d[prep_id]

        image_dir = DataManager.get_image_dir_v2(stack=stack, prep_id=prep_id, resol=resol, version=version, data_dir=data_dir, thumbnail_data_dir=thumbnail_data_dir)

        if version is None:
            image_name = fn + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol + '.' + 'tif'
        else:
            if ext is None:
                if version == 'mask':
                    ext = 'png'
                elif version == 'contrastStretched' or version.endswith('Jpeg') or version == 'jpeg':
                    ext = 'jpg'
                else:
                    ext = 'tif'
            image_name = fn + ('_prep%d' % prep_id if prep_id is not None else '') + '_' + resol + '_' + version + '.' + ext

        image_path = os.path.join(image_dir, image_name)
        return image_path
    
    @staticmethod
    def get_image_dir_v2(stack, prep_id=None, version=None, resol=None,
                      data_dir=ROOT_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
        """
        Args:
            version (str): version string
            data_dir: This by default is ROOT_DIR, but one can change this ad-hoc when calling the function
        Returns:
            Absolute path of the image directory.
        """

        if prep_id is not None and (isinstance(prep_id, str) or isinstance(prep_id, type(str))):
            prep_id = prep_str_to_id_2d[prep_id]

        if version is None:
            if resol == 'thumbnail' or resol == 'down64':
                image_dir = os.path.join(thumbnail_data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol)
            else:
                image_dir = os.path.join(data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol)
        else:
            if resol == 'thumbnail' or resol == 'down64':
                image_dir = os.path.join(thumbnail_data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol + '_' + version)
            else:
                image_dir = os.path.join(data_dir, stack, stack + ('_prep%d' % prep_id if prep_id is not None else '') + '_%s' % resol + '_' + version)

        return image_dir

    
    @staticmethod
    def load_anchor_filename(stack):
        fp = DataManager.get_anchor_filename_filename(stack)
        if not os.path.exists(fp):
            sys.stderr.write("No anchor.txt is found. Seems we are using the operation ini to provide anchor. Try to load operation ini.\n")
            fp = DataManager.get_anchor_filename_filename_v2(stack) # ini
            print(fp)
            print('****************************************************************')
            anchor_image_name = load_ini(fp)['anchor_image_name']
        else:
            # download_from_s3(fp, local_root=THUMBNAIL_ROOT_DIR)
            anchor_image_name = DataManager.load_data(fp, filetype='anchor')
        return anchor_image_name
    
    @staticmethod
    def load_data(filepath, filetype=None):

        if not os.path.exists(filepath):
            sys.stderr.write('File does not exist: %s\n' % filepath)

        if filetype == 'bp':
            return bp.unpack_ndarray_file(filepath)
        elif filetype == 'image':
            return imread(filepath)
        elif filetype == 'hdf':
            try:
                return load_hdf(filepath)
            except:
                print('could not load hdf')
        elif filetype == 'bbox':
            return np.loadtxt(filepath).astype(np.int)
        elif filetype == 'annotation_hdf':
            contour_df = read_hdf(filepath, 'contours')
            return contour_df
        elif filetype == 'pickle':
            return pickle.load(open(filepath, 'r'))
        elif filetype == 'file_section_map':
            with open(filepath, 'r') as f:
                fn_idx_tuples = [line.strip().split() for line in f.readlines()]
                filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
                section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}
            return filename_to_section, section_to_filename
        elif filetype == 'label_name_map':
            label_to_name = {}
            name_to_label = {}
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    name_s, label = line.split()
                    label_to_name[int(label)] = name_s
                    name_to_label[name_s] = int(label)
            return label_to_name, name_to_label
        elif filetype == 'anchor':
            with open(filepath, 'r') as f:
                anchor_fn = f.readline().strip()
            return anchor_fn
        elif filetype == 'transform_params':
            with open(filepath, 'r') as f:
                lines = f.readlines()

                global_params = one_liner_to_arr(lines[0], float)
                centroid_m = one_liner_to_arr(lines[1], float)
                xdim_m, ydim_m, zdim_m  = one_liner_to_arr(lines[2], int)
                centroid_f = one_liner_to_arr(lines[3], float)
                xdim_f, ydim_f, zdim_f  = one_liner_to_arr(lines[4], int)

            return global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f
        elif filepath.endswith('ini'):
            return load_ini(filepath) # this was originally fp
        else:
            sys.stderr.write('File type %s not recognized.\n' % filetype)

    @staticmethod
    def load_sorted_filenames(stack=None, fp=None, redownload=False):
        """
        Get the mapping between section index and image filename.
        Returns:
            filename_to_section, section_to_filename
        """

        if fp is None:
            assert stack is not None, 'Must specify stack'
            fp = DataManager.get_sorted_filenames_filename(stack=stack)

        # download_from_s3(fp, local_root=THUMBNAIL_ROOT_DIR, redownload=redownload)
        filename_to_section, section_to_filename = DataManager.load_data(fp, filetype='file_section_map')
        if 'Placeholder' in filename_to_section:
            filename_to_section.pop('Placeholder')
        return filename_to_section, section_to_filename
    
    @staticmethod
    def load_section_limits_v2(stack, anchor_fn=None, prep_id=2):
        """
        """
        d = DataManager.load_data(DataManager.get_section_limits_filename_v2(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id))
        return np.r_[d['left_section_limit'], d['right_section_limit']]

    @staticmethod
    def load_cropbox_v2(stack, anchor_fn=None, convert_section_to_z=False, prep_id=2,
                        return_origin_instead_of_bbox=False,
                       return_dict=False, only_2d=True):
        """
        Loads the cropping box for the given crop at thumbnail (downsample 32 times from raw) resolution.
        Args:
            convert_section_to_z (bool): If true, return (xmin,xmax,ymin,ymax,zmin,zmax) where z=0 is section #1; if false, return (xmin,xmax,ymin,ymax,secmin,secmax)
            prep_id (int)
        """

        if isinstance(prep_id, str) or isinstance(prep_id, type(str)):
            fp = DataManager.get_cropbox_filename_v2(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
        elif isinstance(prep_id, int):
            # fp = DataManager.get_cropbox_filename(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
            fp = DataManager.get_cropbox_filename_v2(stack=stack, anchor_fn=anchor_fn, prep_id=prep_id)
        else:
            raise Exception("prep_id %s must be either str or int" % prep_id)
        if not os.path.exists(fp):
            sys.stderr.write("Seems you are using operation INIs to provide cropbox.\n")
            if prep_id == 2 or prep_id == 'alignedBrainstemCrop':
                fp = os.path.join(ROOT_DIR, 'CSHL_data_processed', stack, 'operation_configs', 'from_padded_to_brainstem.ini')
            elif prep_id == 5 or prep_id == 'alignedWithMargin':
                fp = os.path.join(ROOT_DIR, 'CSHL_data_processed', stack, 'operation_configs', 'from_padded_to_wholeslice.ini')
            else:
                raise Exception("Not implemented")
        else:
            raise Exception("Cannot find any cropbox specification.")

            # download_from_s3(fp, local_root=THUMBNAIL_ROOT_DIR)

        if fp.endswith('.txt'):
            xmin, xmax, ymin, ymax, secmin, secmax = DataManager.load_data(fp).astype(np.int)

            if convert_section_to_z:
                zmin = int(DataManager.convert_section_to_z(stack=stack, sec=secmin, downsample=32, z_begin=0, mid=True))
                zmax = int(DataManager.convert_section_to_z(stack=stack, sec=secmax, downsample=32, z_begin=0, mid=True))

        elif fp.endswith('.json') or fp.endswith('.ini'):
            if fp.endswith('.json'):
                cropbox_dict = DataManager.load_data(fp)
            else:
                if isinstance(prep_id, str) or isinstance(prep_id, type(str)):
                    prep_id_str = prep_id
                elif isinstance(prep_id, int):
                    prep_id_str = prep_id_to_str_2d[prep_id]
                else:
                    raise

                if fp.endswith('cropbox.ini'):
                    cropbox_dict = load_ini(fp, section=prep_id_str)
                elif '_to_' in fp:
                    cropbox_dict = load_ini(fp)
                else:
                    raise Exception("Do not know how to parse %s for cropbox" % fp)

        assert cropbox_dict['resolution'] == 'thumbnail', "Provided cropbox must have thumbnail resolution."

        xmin = cropbox_dict['rostral_limit']
        xmax = cropbox_dict['caudal_limit']
        ymin = cropbox_dict['dorsal_limit']
        ymax = cropbox_dict['ventral_limit']

        if 'left_limit_section_number' in cropbox_dict:
            secmin = cropbox_dict['left_limit_section_number']
        else:
            secmin = None

        if 'right_limit_section_number' in cropbox_dict:
            secmax = cropbox_dict['right_limit_section_number']
        else:
            secmax = None

        if 'left_limit' in cropbox_dict:
            zmin = cropbox_dict['left_limit']
        else:
            zmin = None

        if 'right_limit' in cropbox_dict:
            zmax = cropbox_dict['right_limit']
        else:
            zmax = None

        if return_dict:
            if convert_section_to_z:
                cropbox_dict = {'rostral_limit': xmin,
                'caudal_limit': xmax,
                'dorsal_limit': ymin,
                'ventral_limit': ymax,
                'left_limit': zmin,
                'right_limit': zmax}
            else:
                cropbox_dict = {'rostral_limit': xmin,
                'caudal_limit': xmax,
                'dorsal_limit': ymin,
                'ventral_limit': ymax,
                'left_limit_section_number': secmin,
                'right_limit_section_number': secmax}
            return cropbox_dict

        else:
            if convert_section_to_z:
                cropbox = np.array((xmin, xmax, ymin, ymax, zmin, zmax))
                if return_origin_instead_of_bbox:
                    return cropbox[[0,2,4]].astype(np.int)
                else:
                    if only_2d:
                        return cropbox[:4].astype(np.int)
                    else:
                        return cropbox.astype(np.int)
            else:
                assert not return_origin_instead_of_bbox
                cropbox = np.array((xmin, xmax, ymin, ymax, secmin, secmax))
                if only_2d:
                    return cropbox[:4].astype(np.int)
                else:
                    return cropbox.astype(np.int)


##### added methods to fix missing
def is_invalid(fn=None, sec=None, stack=None):
    """
    Determine if a section is invalid (i.e. tagged nonexisting, rescan or placeholder in the brain labeling GUI).
    """
    if sec is not None:
        assert stack is not None, 'is_invalid: if section is given, stack must not be None.'
        if sec not in metadata_cache['sections_to_filenames'][stack]:
            return True
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    else:
        assert fn is not None, 'If sec is not provided, must provide fn'
    return fn in ['Nonexisting', 'Rescan', 'Placeholder']

def volume_type_to_str(t):
    if t == 'score':
        return 'scoreVolume'
    elif t == 'annotation':
        return 'annotationVolume'
    elif t == 'annotationAsScore':
        return 'annotationAsScoreVolume'
    elif t == 'annotationSmoothedAsScore':
        return 'annotationSmoothedAsScoreVolume'
    elif t == 'outer_contour':
        return 'outerContourVolume'
    elif t == 'intensity':
        return 'intensityVolume'
    elif t == 'intensity_metaimage':
        return 'intensityMetaImageVolume'
    else:
        raise Exception('Volume type %s is not recognized.' % t)


def generate_metadata_cache():

    global metadata_cache
    metadata_cache['image_shape'] = {}
    metadata_cache['anchor_fn'] = {}
    metadata_cache['sections_to_filenames'] = {}
    metadata_cache['filenames_to_sections'] = {}
    metadata_cache['section_limits'] = {}
    metadata_cache['cropbox'] = {}
    metadata_cache['valid_sections'] = {}
    metadata_cache['valid_filenames'] = {}
    metadata_cache['valid_sections_all'] = {}
    metadata_cache['valid_filenames_all'] = {}
    for stack in all_stacks:

        try:
            metadata_cache['anchor_fn'][stack] = DataManager.load_anchor_filename(stack)
        except Exception as e:
            sys.stderr.write("Failed to cache %s anchor: %s\n" % (stack, e))
            pass

        try:
            metadata_cache['sections_to_filenames'][stack] = DataManager.load_sorted_filenames(stack)[1]
        except Exception as e:
            sys.stderr.write("Failed to cache %s sections_to_filenames: %s\n" % (stack, e))

        try:
            metadata_cache['filenames_to_sections'][stack] = DataManager.load_sorted_filenames(stack)[0]
            if 'Placeholder' in metadata_cache['filenames_to_sections'][stack]:
                metadata_cache['filenames_to_sections'][stack].pop('Placeholder')
            if 'Nonexisting' in metadata_cache['filenames_to_sections'][stack]:
                metadata_cache['filenames_to_sections'][stack].pop('Nonexisting')
            if 'Rescan' in metadata_cache['filenames_to_sections'][stack]:
                    metadata_cache['filenames_to_sections'][stack].pop('Rescan')
        except Exception as e:
            sys.stderr.write("Failed to cache %s filenames_to_sections: %s\n" % (stack, e))

        try:
            metadata_cache['section_limits'][stack] = DataManager.load_section_limits_v2(stack, prep_id=2)
        except Exception as e:
            sys.stderr.write("Failed to cache %s section_limits: %s\n" % (stack, e))

        try:
            # alignedBrainstemCrop cropping box relative to alignedpadded
            metadata_cache['cropbox'][stack] = DataManager.load_cropbox_v2(stack, prep_id=2)
        except Exception as e:
            sys.stderr.write("Failed to cache %s cropbox: %s\n" % (stack, e))

        try:
            first_sec, last_sec = metadata_cache['section_limits'][stack]
            metadata_cache['valid_sections'][stack] = [sec for sec in range(first_sec, last_sec+1) \
                                if sec in metadata_cache['sections_to_filenames'][stack] and \
                                not is_invalid(stack=stack, sec=sec)]
            metadata_cache['valid_filenames'][stack] = [metadata_cache['sections_to_filenames'][stack][sec] for sec in
                                                       metadata_cache['valid_sections'][stack]]
        except Exception as e:
            sys.stderr.write("Failed to cache %s valid_sections/filenames: %s\n" % (stack, e))

        try:
            metadata_cache['valid_sections_all'][stack] = [sec for sec, fn in metadata_cache['sections_to_filenames'][stack].items() if not is_invalid(fn=fn)]
            metadata_cache['valid_filenames_all'][stack] = [fn for sec, fn in metadata_cache['sections_to_filenames'][stack].items() if not is_invalid(fn=fn)]
        except:
            pass

        try:
            metadata_cache['image_shape'][stack] = DataManager.get_image_dimension(stack)
        except Exception as e:
            sys.stderr.write("Failed to cache %s image_shape: %s\n" % (stack, e))
            pass

    return metadata_cache


generate_metadata_cache()

