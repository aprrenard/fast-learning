from datetime import datetime

import pandas as pd


def read_excel_db(db_path):
    database = pd.read_excel(db_path, converters={'day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    # Change yes/no columns to booleans.
    database = database.replace('yes', True)
    database = database.replace('no', False)
    database = database.astype({'two_p_imaging': bool,
                                'optogenetic': bool,'pharmacology': bool})

    return database
