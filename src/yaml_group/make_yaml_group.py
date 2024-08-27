
import os

import yaml
from datetime import datetime

# sys.path.append('H:\\anthony\\repos\\NWB_analysis')
import nwb_utils.utils_io as utils_io
from nwb_wrappers import nwb_reader_functions as nwb_read
from src.utils.utils_io import read_excel_db


def filter_nwb_based_on_excel_db(excel_path, nwb_path, experimenters, exclude_cols, **filters):
    
    nwb_list = read_excel_db(excel_path)
    for key, val in filters.items():
        if type(val) is list:
            nwb_list = nwb_list.loc[(nwb_list[key].isin(val))]
        else:
            nwb_list = nwb_list.loc[(nwb_list[key]==val)]

    # Remove excluded sessions.
    for col in exclude_cols:
        nwb_list = nwb_list.loc[(nwb_list[col]!='exclude')]
    
    nwb_list = list(nwb_list.session_id)
    nwb_list = [os.path.join(nwb_path, f + '.nwb') for f in nwb_list]

    if experimenters:
        nwb_list = [nwb for nwb in nwb_list if os.path.basename(nwb)[-25:-23] in experimenters]

    return nwb_list

    # nwb_metadata = {nwb: nwb_read.get_session_metadata(nwb) for nwb in nwb_list}
    # nwb_list_rew = [nwb for nwb, metadata in nwb_metadata.items()
    #                 if (os.path.basename(nwb)[-25:-20] in nwb_list)
    #                 & ('twophoton' in metadata['session_type'])
    #                 & (metadata['behavior_type'] in behavior_types)
    #                 & (metadata['day'] in days)
    #                 ]

    # def get_date(x):
    #     if x[-25:-23] in ['GF', 'MI']:
    #         return datetime.strptime(x[-19:-11], '%d%m%Y')
    #     else:
    #         return datetime.strptime(x[-19:-11], '%Y%m%d')

    # # Reorder nwb list in chronological order mouse wise (GF does not use americain dates).
    # temp = []
    # mice_list = db.subject_id.unique()
    # for mouse in mice_list:
    #     l = [nwb for nwb in nwb_list if mouse in nwb]
    #     l = sorted(l, key=get_date)
    #     temp.extend(l)
    # nwb_list = temp

if __name__ == '__main__':
    excel_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\mice_info\session_metadata.xlsx'
    nwb_path = utils_io.get_experimenter_nwb_folder('AR')
    yaml_folder = r'C:\Users\aprenard\recherches\repos\fast-learning\docs\groups'
    yaml_name = 'imaging_non_rewarded.yaml'
    session_paths = filter_nwb_based_on_excel_db(excel_path, nwb_path,
                                                 exclude_cols=['exclude', 'two_p_exclude'],
                                                 experimenters=['AR', 'GF', 'MI'],
                                                 reward_group='R+',
                                                 day=['-3', '-2', '-1', '0', '+1', '+2'],
                                                 two_p_imaging='yes')
    yaml_path = os.path.join(yaml_folder, yaml_name)
    with open(yaml_path, 'w') as fid:
        yaml.safe_dump(session_paths, fid)
        


# excluded_two_p  = ['GF264', 'GF278', 'GF208', 'GF340']
        