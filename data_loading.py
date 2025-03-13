import os
import numpy as np
import pandas as pd
import h5py
from typing import List, Tuple
from tqdm import tqdm
from datetime import datetime

def periodic_table_naming_to_human_readable(channels: List[str]) -> List[str]:
    peridic_table_mapping = {
        'Cu': 'copper',
        'Fe': 'iron',
        'Mg': 'magnesium',
        'Mn': 'manganese',
        'Zn': 'zinc',
    }
    return [peridic_table_mapping[channel] for channel in channels]

def read_channels_of_dataset(root_dir: str) -> List[str]:
    cores_filename = os.path.join(root_dir, 'la-icp-ms', "aq.h5")
    with h5py.File(cores_filename, "r") as cores_file:
        cores_global_group_key = list(cores_file.keys())[1]
        core_channels = [channel.decode('ascii') for channel in cores_file[cores_global_group_key]['element_list'][:]]
        core_channels.remove('TIC')
        core_channels = periodic_table_naming_to_human_readable(core_channels)
        return core_channels

def read_metadata_records(root_dir) -> pd.DataFrame:
    metadata_records: pd.DataFrame = pd.read_excel(os.path.join(root_dir, 'la-icp-ms', 'LEAP code response data 05122023.xlsx'))
    metadata_records = metadata_records.rename(columns={
        "Extreme non-responder (death within 2 years?)": "is_extreme"
    })
    batch_info = pd.read_excel(os.path.join(root_dir, 'la-icp-ms', '240129 DeltaLeap.xlsx'))
    batch_info = batch_info[~batch_info['LEAPID'].isna()]
    batch_info['LEAPID'] = batch_info['LEAPID'].apply(lambda x: x.lower().capitalize())
    date_imaged = []
    for date, leap_id in zip(batch_info['Date Imaged'], batch_info['LEAPID']):
        if type(date) == int or type(date) == str:
            if type(date) == str and 'or' in date:
                # if screening date is unclear take the first one
                date = date.split('or')[0].strip()
            date_formatted = datetime.strptime(str(int(date)), "%y%m%d").strftime("%d-%m-%y")
        else:
            date_formatted = ''
        date_imaged.append(date_formatted)
    batch_info['Date Imaged'] = date_imaged
    batch_info = batch_info.set_index('LEAPID')
    parsed_metadata_records = {
        'leap-id': [],
        'is-responded': [],
        'is-extreme-responded': [],
        'batch-date': [],
        'leap-type': [],
        'medium-type': [],
        'is-force': [],
        'next': [],
    }
    crashed_cores = [
        'Leap052',
        'Leap053',
        'Leap054',
        'Leap056',
        'Leap057',
        # 'Leap114',
        'Leap115',
        'Leap116',
        'Leap118',
    ]
    for (_, row), is_extreme in zip(metadata_records.iterrows(), metadata_records['is_extreme'].isna()):
        if row['Response'] == 'no surgery yet':
            continue
        # remove prefix "LEAP"
        leap_ids = row['LEAP ID'][4:]
        leap_ids = leap_ids.split('/')
        medium_types = row['FFPE/FRESH/frozen'].lower()
        if ',' in medium_types:
            medium_types = medium_types.split(',')
        else:
            medium_types = [medium_types] * len(leap_ids)
        leap_types = row['COMMENTS'].lower()
        leap_types_map = dict()
        if ',' in leap_types:
            leap_types = [l.split('=')for l in leap_types.split(',')]
            for id, leap_type in leap_types:
                leap_types_map[id.zfill(3)] = leap_type
        else:
            # map everything to same type
            for id in leap_ids:
                leap_types_map[id.zfill(3)] = leap_types
        for i, (id, medium_type) in enumerate(zip(leap_ids, medium_types)):
            parsed_leap_id = f'Leap{id.zfill(3)}'
            if parsed_leap_id not in crashed_cores:
                parsed_metadata_records['leap-id'].append(parsed_leap_id)
                is_responded = row['Response'] == 'pCR' or row['Response'] == 'Responder'
                is_extreme_responded = (row['Response'] == "Non-Responder") & (not is_extreme)
                parsed_metadata_records['is-responded'].append(is_responded)
                parsed_metadata_records['is-extreme-responded'].append(is_extreme_responded)
                parsed_metadata_records['batch-date'].append(batch_info.loc[parsed_leap_id, 'Date Imaged'] if parsed_leap_id in batch_info.index else '')
                leap_type = leap_types_map[id.zfill(3)]
                parsed_metadata_records['leap-type'].append(leap_type)
                parsed_metadata_records['medium-type'].append(medium_type)
                parsed_metadata_records['is-force'].append(row['FORCE'] == 'FORCE')
                if i == 0:
                    next_leaps = [f'Leap{leap_id.zfill(3)}' for leap_id in leap_ids][1:]
                else:
                    next_leaps = []
                parsed_metadata_records['next'].append(next_leaps)
    return pd.DataFrame(parsed_metadata_records).set_index('leap-id')

class Leap:
    
    def __init__(self, **kargs) -> None:
        assert (kargs['is_responded'] and (not kargs['is_extreme_responded'])) or (not kargs['is_responded'])
        self.id: str = kargs['id']
        self.is_responded: bool = kargs['is_responded'] if 'is_responded' in kargs else None
        self.is_extreme_responded: bool = kargs['is_extreme_responded'] if 'is_extreme_responded' in kargs else None
        self.biobank_id: str = kargs['biobank_id']
        self.is_core: bool = kargs['is_core']
        self.mask: np.ndarray = kargs['mask']
        self.image: np.ndarray = kargs['image']
        self.pixels: pd.DataFrame = kargs['pixels']
        self.file_tag_id: str = kargs['file_tag_id']
        self.file_uid: str = kargs['file_uid']
        self.chunk_id: str = kargs['chunk_id']
        self.tic: np.ndarray = kargs['tic']
        self.batch_date: str = kargs['batch_date']
        self.lod: np.ndarray = kargs['lod']
        self.medium: str = kargs['medium']
        self.is_force: str = kargs['is_force']
        self.analysis_id = None
        self.next: List[Leap] = []

def extract_non_background_pixels(image: np.ndarray, channels: List[str]) -> pd.DataFrame:
    y, x = np.where(np.any(image, axis=-1))
    pixels: pd.DataFrame = pd.DataFrame(image[(y, x)], columns=channels)
    pixels.insert(0, "y", y)
    pixels.insert(0, "x", x)
    return pixels

def fetch_from_scalar_dataset(dataset):
    """fetch number form scalar dataset object of h5py lib
    Args:
        dataset (Unknown): an object that encapsulates a number (int or float)
    Returns:
        unknown: the number
    """
    return dataset[()]


def get_coordinates_from_mask(mask: np.ndarray) -> Tuple[Tuple, Tuple]:
    """generate the bounding box of the tissue in the mask without zero valued background.
    Done by removing all pixels that zero in rectuangular frame that surround the tissue.

    Args:
        mask (np.ndarray): (H,W) image mask

    Returns:
        Tuple[Tuple, Tuple]: (min_y, max_y), (min_x, max_x)
    """
    coordiantes_were_images_non_zero = np.argwhere(mask != 0)
    max_y = coordiantes_were_images_non_zero[:, 0].max()
    min_y = coordiantes_were_images_non_zero[:, 0].min()
    max_x = coordiantes_were_images_non_zero[:, 1].max()
    min_x = coordiantes_were_images_non_zero[:, 1].min()
    return (min_y, max_y), (min_x, max_x)
        
def get_coordinates_from_image(image: np.ndarray) -> Tuple[Tuple, Tuple]:
    """generate the bounding box of the tissue in the image without zero valued background.
    Done by removing all pixels that are zero in all channels in rectuangular frame that surround the tissue.

    Args:
        image (np.ndarray): (C,H,W) image

    Returns:
        Tuple[Tuple, Tuple]: (min_y, max_y), (min_x, max_x)
    """
    coordiantes_were_images_non_zero = [np.argwhere(channel != 0) for channel in image] # assume blackground is zero
    max_y = max([coordinates_of_image[:, 0].max() for coordinates_of_image in coordiantes_were_images_non_zero])
    min_y = min([coordinates_of_image[:, 0].min() for coordinates_of_image in coordiantes_were_images_non_zero])
    max_x = max([coordinates_of_image[:, 1].max() for coordinates_of_image in coordiantes_were_images_non_zero])
    min_x = min([coordinates_of_image[:, 1].min() for coordinates_of_image in coordiantes_were_images_non_zero])
    return (min_y, max_y), (min_x, max_x)

def validate_mask_image_consistency(image: np.ndarray, mask: np.ndarray) -> bool:
    """validate that image and mask afer cropping results the same (H,W) shape. 
    The validation done bu applying the cropping solely on the image and separately on the mask then compares the shapes.
    If the shapes matches then it doesn't matter which method to use, otherwise there is inconsistency and may be a bug in how the mask and the image were produced. 
    Args:
        image (np.ndarray): (C, H, W) image
        mask (np.ndarray): (H, W)

    Returns:
        bool: true iff the shapes match
    """
    image_y_bounds, image_x_bounds = get_coordinates_from_image(image)
    mask_y_bounds, mask_x_bounds = get_coordinates_from_mask(mask)
    return  image_y_bounds[0] == mask_y_bounds[0] and \
            image_y_bounds[1] == mask_y_bounds[1] and \
            image_x_bounds[0] == mask_x_bounds[0] and \
            image_x_bounds[1] == mask_x_bounds[1]
def crop_background(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """removes a rectangle frame of background where pixels are zero
    Args:
        image (np.ndarray): (C, H, W) image
        mask (np.ndarray): (H, W) bitmap of the image mask

    Returns:
        Tuple[np.ndarray, np.ndarray]: (image, mask) after cropping
    """
    (min_y, max_y), (min_x, max_x) = get_coordinates_from_mask(mask)
    image = image[:, min_y:max_y+1, min_x:max_x+1]
    mask = mask[min_y:max_y+1, min_x:max_x+1]
    return image, mask

def clip_lod(image: np.ndarray, lod: np.ndarray, mask: np.ndarray, id, is_core, mssg: bool, verbose=False) -> None:
    """sets values for each channel that are lower 
    than the channel's lod (limit of detection) to the lod value of the channel.
    This manipulation done inplace.
    
    Args:
        image (np.ndarray): (C, H, W) image
        lod (np.ndarray): (C,) list of lod per channel
    """
    # apply the lod over the tissue area
    mask = mask.astype(bool)
    where_need_to_clip = (image <= lod.reshape(lod.shape[0], 1, 1)) & np.stack([mask, mask, mask, mask, mask])
    y_background, x_background = np.where(np.logical_not(mask))
    channels = ['magnesium', 'manganese', 'iron', 'copper', 'zinc']
    for i, channel_lod in enumerate(lod):
        y_tissue, x_tissue = np.where(where_need_to_clip[i])
        if is_core and mssg and verbose:
            print(f'{id} {channels[i]} lod clipping affected {len(y_tissue)} pixels')
        if is_core and (not mssg) and verbose:
            print(f'{id} {channels[i]} clipping negative values to zero affecting {len(y_tissue)} pixels')
        image[i, y_tissue, x_tissue] = channel_lod
        image[i, y_background, x_background] = 0
    
def dataset2leaps(dataset, samples_names: List[str], channels: List[str], metadata_records: pd.DataFrame, tic_index: int, apply_lod: bool=True, verbose=False, desc='', filename = None) -> List[Leap]:
    leaps: List[Leap] = []
    for sample in tqdm(samples_names, desc=desc):
        sample_name: str = fetch_from_scalar_dataset(dataset[sample]['sample_name']).decode('ascii')
        mask: np.ndarray = dataset[sample]['mask'][:]
        # image is (C, H, W)
        lod: np.ndarray = dataset[sample]['LoD'][:] if 'LoD' in dataset[sample] else None
        if len(sample_name.split('_')) == 2:
            leap_id, biobank_id = sample_name.split('_')
        else:
            leap_id, _, biobank_id = sample_name.split('_')
        leap_id = leap_id.title()
        sample_type = metadata_records.loc[leap_id, 'leap-type']
        image: np.ndarray = dataset[sample]['processed_data'][:]
        image = image.squeeze()
        # remove suffix a or b that note the chunk id
        biobank_id: str = biobank_id[:-1]
        # delete TIC
        if lod is not None:
            lod = np.delete(lod, tic_index, axis=0)
        image = np.delete(image, tic_index, axis=0)
        if not validate_mask_image_consistency(image, mask):
            raise Exception(f'{sample_name}, {sample} has mask that inconsistent with image background')
        image, mask = crop_background(image, mask)
        if apply_lod:
            # apply clipping by lod
            clip_lod(image, lod, mask, leap_id + sample_name[-1], sample_type.lower() == 'core', True, verbose)
        else:
            clip_lod(image, np.zeros(len(channels)), mask, leap_id + sample_name[-1], sample_type.lower() == 'core', False, verbose)
        # extract pixels
        image = np.moveaxis(image, 0, -1)
        pixels = extract_non_background_pixels(image, channels)
        tic = image.sum(axis=-1)
        leap_conf = {
            'id': leap_id,
            'is_core': sample_type.lower() == 'core',
            'is_responded': metadata_records.loc[leap_id, 'is-responded'],
            'is_extreme_responded': metadata_records.loc[leap_id, 'is-extreme-responded'],
            'batch_date': metadata_records.loc[leap_id, 'batch-date'],
            'biobank_id': biobank_id,
            'mask': mask,
            'image': image,
            'pixels': pixels,
            'file_tag_id': sample,
            'file_uid': sample_name,
            'tic': tic,
            'lod': lod,
            'chunk_id': sample_name[-1],
            'medium': metadata_records.loc[leap_id, 'medium-type'],
            'is_force': metadata_records.loc[leap_id, 'is-force'],
        }
        leaps.append(Leap(**leap_conf))
    return leaps

def read_h5file(filepath, metadata_records: pd.DataFrame, apply_lod: bool=True, verbose=False):
    with h5py.File(filepath, "r") as file:
        # get first object name/key; may or may NOT be a group
        leaps_group_key = list(file.keys())[0]
        global_group_key = list(file.keys())[1]
        # If leaps_group_key is a dataset name, this gets the dataset values and returns as a list
        leap_samples_names = list(file[leaps_group_key])
        # preferred methods to get dataset values:
        leaps_dataset = file[leaps_group_key]      # returns as a h5py dataset object
        channels = [channel.decode('ascii') for channel in file[global_group_key]['element_list'][:]]
        
        tic_index = channels.index('TIC')
        channels.remove('TIC')
        
        channels = periodic_table_naming_to_human_readable(channels)
        
        if apply_lod and verbose:
            print('Applying clipping by lod')
        leaps: List[Leap] = dataset2leaps(leaps_dataset, leap_samples_names, channels, metadata_records, tic_index, apply_lod, verbose, desc=filepath, filename=filepath)
        
        leaps.sort(key=lambda x: x.id + x.chunk_id)
        return leaps, channels

def read_dataset(root_dir, apply_lod: bool=True, verbose=False):
    cores_filename = os.path.join(root_dir, 'la-icp-ms', "aq_cores_1.h5")
    cores_filename_2 = os.path.join(root_dir, 'la-icp-ms', "aq_cores_2_FF.h5")
    cores_filename_3 = os.path.join(root_dir, 'la-icp-ms', "aq_cores_2_FFPE.h5")
    resection_filename = os.path.join(root_dir, 'la-icp-ms', "resection_aq.h5")
    metadata_records: pd.DataFrame = read_metadata_records(root_dir)
    cores, core_channels = read_h5file(cores_filename, metadata_records, apply_lod=apply_lod, verbose=verbose)
    cores_2, core_channels = read_h5file(cores_filename_2, metadata_records, apply_lod=apply_lod, verbose=verbose)
    cores_3, core_channels = read_h5file(cores_filename_3, metadata_records, apply_lod=apply_lod, verbose=verbose)
    # resections, resectionn_channels = read_h5file(resection_filename, metadata_records, apply_lod=apply_lod, verbose=verbose)
    resections = []
    cores = cores + cores_2 + cores_3
    cores.sort(key=lambda x: x.id + x.chunk_id)
    return cores, resections, core_channels


def remove_manganese_from_corrupted_cores(cores, channels):
    corrupted_ids = [f'Leap{str(id).zfill(3)}' for id in range(69, 152)]
    manganese_index = channels.index('manganese')
    corrupted_leaps = [core for core in cores if core.id in corrupted_ids]
    for leap in corrupted_leaps:
        leap.pixels.drop(inplace=True, columns=['manganese'])
        leap.image[:, :, manganese_index] = np.nan

def get_core_ids_of_that_are_not_first(metadata_df):
    def get_next_ids(next_leaps):
        return [
            leap 
            for leap in next_leaps 
            if leap in metadata_df.index and metadata_df.loc[leap, 'leap-type'] == 'core'
        ]
    next_ids_per_leap = metadata_df['next'].apply(get_next_ids)
    return [e for array in next_ids_per_leap.to_list() for e in array]

def get_cores(root_dir):
    channels = read_channels_of_dataset(root_dir)
    metadata_df = read_metadata_records(root_dir)
    cores_new, resections_new, channels_new = read_dataset(root_dir, apply_lod=False, verbose=False)

    remove_manganese_from_corrupted_cores(cores_new, channels)

    core_ids_not_first = get_core_ids_of_that_are_not_first(metadata_df)
    core_ids_frozen = metadata_df.index[(metadata_df['leap-type'] == 'core') & (metadata_df['medium-type'] == 'frozen')].to_list()
    cores_to_exclude = pd.Series(core_ids_frozen + core_ids_not_first).unique().tolist()
    print('Excluded cores')
    print(cores_to_exclude)
    cores_new = [core for core in cores_new if core.id not in cores_to_exclude]
    
    new_metadata_df = pd.read_excel(os.path.join(root_dir, 'la-icp-ms', 'tnbc_dataset.xlsx'))
    ignore_count = 0
    not_same = 0
    cores_new_fixed_new_metadata = []
    for i, core in enumerate(cores_new):
        response_txt = new_metadata_df.loc[new_metadata_df['leap_id'] == core.id, 'response'].iloc[0]
        analysis_id = new_metadata_df.loc[new_metadata_df['leap_id'] == core.id, 'analysis_id'].iloc[0]
        if type(response_txt) == float and np.isnan(response_txt):
            ignore_count += 1
        else:
            is_responded = response_txt == 'pCR' or response_txt == 'Responder'
            if core.is_responded != is_responded:
                not_same += 1
                core.is_responded = is_responded
            core.analysis_id = analysis_id
            cores_new_fixed_new_metadata.append(core)

    return cores_new_fixed_new_metadata