#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gquant.dataframe_flow import TaskGraph
import cupy
import sys
import time


def collector_data(seed):
    taskgraph = TaskGraph.load_taskgraph('./option_simulation.gq.yaml')
    iterator = taskgraph.run(replace={"parameters": {
        "conf": {
            "seed": seed
        }
    }})[0]
    number = 102400
    block = 10
    print('seed number', seed)
    for bid in range(block):
        paras = []
        targets = []
        for i in range(number):
            if i % (number//100) == 0:
                print('currently', i/number*100, 'percent')
            sim_result = next(iterator)
            para = sim_result[0]
            target = sim_result[1]
            paras.append(para)
            targets.append(target)
        cupy.save('para_seed{}_block{}'.format(seed, bid),
                  cupy.concatenate(paras))
        cupy.save('taget_seed{}_block{}'.format(seed, bid),
                  cupy.concatenate(targets))


if __name__ == "__main__":
    collector_data(int(sys.argv[1]))
