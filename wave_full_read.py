import os
import base64
import struct

import h5py
import numpy as np
import polars as pl


def ept_wavedata_decode(wave):

    data = base64.b64decode(bytes(wave, encoding="utf8"))

    cleaned_data = []

    struct_format = ">f"

    for i in range(len(data) // 4):

        cleaned_data.append(struct.unpack_from(struct_format, data, 4*i)[0])

    return np.array(cleaned_data, dtype=np.float32)


def csv2h5(filename, df):

    h5_dir = os.path.join("data", "h5")
    # exclude `id`, `wave_data`` from df.columns
    attr_columns = [
        col for col in df.columns if col not in ['id',  'wave_data']]

    h5_file = os.path.join(h5_dir, filename + '.h5')

    # check if h5_file exists, if so, print warning and return
    if os.path.exists(h5_file):
        print(f"File {h5_file} already exists, skip...")
        return

    # print(attr_columns)

    with h5py.File(h5_file, 'w') as f:

        for row in df.iter_rows(named=True):

            wave_data = ept_wavedata_decode(row['wave_data'])

            # check if dataset exists, if so, print warning and continue
            if str(row['id']) in f:
                print(f"Dataset {row['id']} already exists, skip...")
                continue

            # Create a new dataset
            dataset = f.create_dataset(str(row['id']), data=wave_data)

            for col in attr_columns:
                # print(col, row[col])
                dataset.attrs[col] = row[col] if row[col] is not None else ''


def all_csv2h5():

    data_dir = os.path.join("data", "wave-full")

    for filename in os.listdir(data_dir):

        df = pl.read_csv(os.path.join(data_dir, filename))

        df.drop_in_place('data_time_right')

        # print(df.shape)

        csv2h5(filename.split('.')[0], df)

        # break


def all_readh5():

    h5_dir = os.path.join("data", "h5")

    for filename in os.listdir(h5_dir):

        readh5(os.path.join(h5_dir, filename))

        break


def readh5(filepath):
    # Read the h5 file
    with h5py.File(filepath, 'r') as f:
        # Loop over the datasets
        for name in f:
            if isinstance(f[name], h5py.Dataset):
                ds = f[name]

                print(ds[()].shape, np.min(ds[()]), np.max(ds[()]))

                mf = ['data_time', 'device_id', 'device_name', 'profile_name', 'pump_category', 'impellers',
                      'blades', 'orientation', 'rpm_rate', 'rpm_max', 'rolling_num', 'alarm_name', 'alarm_type']

                for m in mf:
                    print(ds.attrs[m])

                break


if __name__ == "__main__":
    # all_csv2h5()
    all_readh5()
