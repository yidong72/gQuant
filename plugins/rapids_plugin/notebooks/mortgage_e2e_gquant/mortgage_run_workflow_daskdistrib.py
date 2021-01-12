'''
'''
import os

try:
    # Disable NCCL P2P. Only necessary for versions of NCCL < 2.4
    # https://rapidsai.github.io/projects/cudf/en/0.8.0/dask-xgb-10min.html#Disable-NCCL-P2P.-Only-necessary-for-versions-of-NCCL-%3C-2.4
    os.environ["NCCL_P2P_DISABLE"] = "1"
except Exception:
    pass

import json

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
# from distributed import Client

from mortgage_common import (
    mortgage_etl_workflow_def, generate_mortgage_gquant_run_params_list,
    MortgageTaskNames)


def main():

    memory_limit = 128e9
    threads_per_worker = 4
    cluster = LocalCUDACluster(
        memory_limit=memory_limit,
        threads_per_worker=threads_per_worker)
    client = Client(cluster)
    sched_info = client.scheduler_info()

    print('CLIENT: {}'.format(client))
    print('SCHEDULER INFO:\n{}'.format(json.dumps(sched_info, indent=2)))

    # Importing here in case RMM is used later on. Must start client prior
    # to importing cudf stuff if using RMM.
    from gquant.dataframe_flow import (TaskSpecSchema, TaskGraph)

    # workers_names = \
    #     [iw['name'] for iw in client.scheduler_info()['workers'].values()]
    # nworkers = len(workers_names)

    _basedir = os.path.dirname(__file__)
    # mortgage_data_path = '/datasets/rapids_data/mortgage'
    mortgage_data_path = os.path.join(_basedir, 'mortgage_data')

    # Using some default csv files for testing.
    # csvfile_names = os.path.join(mortgage_data_path, 'names.csv')
    # acq_data_path = os.path.join(mortgage_data_path, 'acq')
    # perf_data_path = os.path.join(mortgage_data_path, 'perf')
    # csvfile_acqdata = os.path.join(acq_data_path, 'Acquisition_2000Q1.txt')
    # csvfile_perfdata = \
    #     os.path.join(perf_data_path, 'Performance_2000Q1.txt_0')
    # mortgage_etl_workflow_def(
    #     csvfile_names, csvfile_acqdata, csvfile_perfdata)

    gquant_task_spec_list = mortgage_etl_workflow_def()

    start_year = 2000
    end_year = 2001  # end_year is inclusive
    # end_year = 2016  # end_year is inclusive
    # part_count = 16  # the number of data files to train against

    # create_dmatrix_serially - When False on same node if not enough host RAM
    # then it's a race condition when creating the dmatrix. Make sure enough
    # host RAM otherwise set to True.
    # create_dmatrix_serially = False

    # able to do 18 with create_dmatrix_serially set to True
    part_count = 18  # the number of data files to train against
    create_dmatrix_serially = True
    # part_count = 4  # the number of data files to train against

    # Use RAPIDS Memory Manager. Seems to work fine without it.
    use_rmm = False

    # Clean up intermediate dataframes in the xgboost training task.
    delete_dataframes = True

    mortgage_run_params_dict_list = generate_mortgage_gquant_run_params_list(
        mortgage_data_path, start_year, end_year, part_count,
        gquant_task_spec_list)

    _basedir = os.path.dirname(__file__)
    mortgage_lib_module = os.path.join(_basedir, 'mortgage_gquant_plugins.py')

    filter_dask_logger = False

    mortgage_workflow_runner_task = {
        TaskSpecSchema.task_id:
            MortgageTaskNames.dask_mortgage_workflow_runner_task_name,
        TaskSpecSchema.node_type: 'DaskMortgageWorkflowRunner',
        TaskSpecSchema.conf: {
            'mortgage_run_params_dict_list': mortgage_run_params_dict_list,
            'client': client,
            'use_rmm': use_rmm,
            'filter_dask_logger': filter_dask_logger,
        },
        TaskSpecSchema.inputs: [],
        TaskSpecSchema.filepath: mortgage_lib_module
    }

    dxgb_gpu_params = {
        'nround': 100,
        'max_depth': 8,
        'max_leaves': 2 ** 8,
        'alpha': 0.9,
        'eta': 0.1,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'subsample': 1,
        'reg_lambda': 1,
        'scale_pos_weight': 2,
        'min_child_weight': 30,
        'tree_method': 'gpu_hist',
        'n_gpus': 1,
        'distributed_dask': True,
        'loss': 'ls',
        # 'objective': 'gpu:reg:linear',
        'objective': 'reg:squarederror',
        'max_features': 'auto',
        'criterion': 'friedman_mse',
        'grow_policy': 'lossguide',
        'verbose': True
    }

    dxgb_trainer_task = {
        TaskSpecSchema.task_id: MortgageTaskNames.dask_xgb_trainer_task_name,
        TaskSpecSchema.node_type: 'DaskXgbMortgageTrainer',
        TaskSpecSchema.conf: {
            'create_dmatrix_serially': create_dmatrix_serially,
            'delete_dataframes': delete_dataframes,
            'dxgb_gpu_params': dxgb_gpu_params,
            'client': client,
            'filter_dask_logger': filter_dask_logger
        },
        TaskSpecSchema.inputs: [
            MortgageTaskNames.dask_mortgage_workflow_runner_task_name
        ],
        TaskSpecSchema.filepath: mortgage_lib_module
    }

    task_spec_list = [mortgage_workflow_runner_task, dxgb_trainer_task]

    out_list = [MortgageTaskNames.dask_xgb_trainer_task_name]
    task_graph = TaskGraph(task_spec_list)
    (bst,) = task_graph.run(out_list)

    print('XGBOOST BOOSTER:\n', bst)


if __name__ == '__main__':
    main()
