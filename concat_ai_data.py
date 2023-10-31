import os
import datetime
from ast import literal_eval
import pickle

import numpy as np
import polars as pl
import h5py

def list_fault_code():
    """
    BaseplateUnbalance: 1
    Misaligned: 2
    Bearing: 3
    ImpellerUnbalance: 4
    Cavitation: 5
    """
    for filename in os.listdir(os.path.join("data", 'ai-service')):

        # check if `filename` is a file
        if not os.path.isfile(os.path.join("data", "ai-service", filename)):
            continue

        df = pl.read_csv(os.path.join("data", "ai-service", filename))
        # print(filename, df.groupby('label_true').count())
        filename = filename.split('.')[0]
        pump_type, fault_type, _ = filename.split('_')

        # assign a constant column pump_type to df
        # df['pump_type'] = pump_type
        df = df.with_columns(
            pl.repeat(pump_type, df.shape[0]).alias('pump_type'))

        # count items in label_true array
        print(df.groupby('label_true').count())
        print(fault_type)


def csv2pkl():

    data_dict = {}

    for filename in os.listdir(os.path.join("data", 'ai-service')):

        # check if `filename` is a file
        if not os.path.isfile(os.path.join("data", "ai-service", filename)):
            continue

        df = pl.read_csv(os.path.join("data", "ai-service", filename))
        # print(filename, df.groupby('label_true').count())
        filename = filename.split('.')[0]
        pump_type, fault_type, _ = filename.split('_')

        # assign a constant column pump_type to df
        # df['pump_type'] = pump_type
        df = df.with_columns(
            pl.repeat(pump_type, df.shape[0]).alias('pump_type'))

        # # rename `label_true` to `label_true` + `fault_type`
        # df = df.with_columns(df['label_true'].alias('label_true_' + fault_type))

        # # drop `label_pred` and `label_true`
        # df = df.drop(['label_pred', 'label_true'])
        # drop index
        if '' in df.columns:
            df.drop_in_place('')

        for row in df.iter_rows(named=True):
            # print(row)
            # print(row['id'], row['label_true_' + fault_type])
            label = int(row['label_true'])

            if row['wave_id'] not in data_dict:
                data_dict[row['wave_id']] = {
                    'data_time': datetime.datetime.strptime(row['data_time'], "%Y-%m-%d %H:%M:%S").timestamp(),
                    'rpm': int(row['rpm']),
                    'wave_data': literal_eval(row['wave_decode']),
                    'pump_type': row['pump_type'],
                    'label_true': label
                }
            else:
                if label != 0:
                    data_dict[row['wave_id']]['label_true'] = label

    # dump `data_dict` to pickle
    with open(os.path.join("data", "fault_data.pkl"), 'wb') as f:
        pickle.dump(data_dict, f)


def pkl2h5():

    with open(os.path.join("data", "fault_data.pkl"), 'rb') as f:
        data_dict = pickle.load(f)

        label_true = np.array([v['label_true'] for k, v in data_dict.items()])
        # count items in label_true array
        print(np.unique(label_true, return_counts=True))

        for k, v in data_dict.items():
            # data_dict[k]['wave_data'] = np.array(v['wave_data'])
            print(v)
            break

        wave_id = []
        data_time = []
        rpm = []
        wave_data = []
        pump_type= []
        label = []

        for k, v in data_dict.items():
            wave_id.append(k)
            data_time.append(int(v['data_time']))
            rpm.append(int(v['rpm']))
            wave_data.append(v['wave_data'])
            pump_type.append(v['pump_type'])
            label.append(int(v['label_true']))

        # wave_id = np.array(wave_id)
        data_time = np.array(data_time, dtype=np.int32)
        rpm = np.array(rpm, dtype=np.int32)
        wave_data = np.array(wave_data,dtype=np.float32)
        # pump_type = np.array(pump_type)
        label = np.array(label, dtype=np.int32)

        print(len(wave_id), data_time.shape, rpm.shape, wave_data.shape, len(pump_type), label.shape)

    # save `data_dict` to h5, filename = 'fault_data.h5'
    with h5py.File(os.path.join("data", "fault_data.h5"), 'w') as f:

        f.create_dataset('wave_id', data=wave_id, chunks=True)
        f.create_dataset('data_time', data=data_time, chunks=True)
        f.create_dataset('rpm', data=rpm, chunks=True)
        f.create_dataset('wave_data', data=wave_data, chunks=True)
        f.create_dataset('pump_type', data=pump_type, chunks=True)
        f.create_dataset('label', data=label, chunks=True)

        


def read_h5():
    with h5py.File(os.path.join("data", "fault_data.h5"), 'r') as f:

        for name in f:
            print(name)
            print(f[name][()])


read_h5()