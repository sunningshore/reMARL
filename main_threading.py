import multiprocessing
import os
# from multiprocessing import Pool
# from time import sleep
# from random import randint
# from test_just_test1_swarm_contour_mue import *


def execute(process):
    os.system(f'python {process}')

#
# def task(id_):
#     for _ in range(10):
#         stime = randint(1, 5)
#         print(f"Task {id_}: Woke up. Now sleep {stime}")
#         sleep(stime)


if __name__ == '__main__':
    # freeze_support()

    hyper_agent_num = str(3)  # Only takes odd numbers, one in the center.
    agent_num, obs_num = str(1), str(1)
    all_processes1 = ('test_just_test1_swarm_contour_mue.py 0' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 1' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 2' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 3' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 4' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num
                      )
    hyper_agent_num = str(5)  # Only takes odd numbers, one in the center.
    agent_num, obs_num = str(1), str(1)
    all_processes2 = ('test_just_test1_swarm_contour_mue.py 0' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 1' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 2' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 3' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 4' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num
                      )
    hyper_agent_num = str(7)  # Only takes odd numbers, one in the center.
    agent_num, obs_num = str(1), str(1)
    all_processes3 = ('test_just_test1_swarm_contour_mue.py 0' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 1' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 2' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 3' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 4' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num
                      )
    hyper_agent_num = str(10)  # Only takes odd numbers, one in the center.
    agent_num, obs_num = str(1), str(1)
    all_processes4 = ('test_just_test1_swarm_contour_mue.py 0' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 1' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 2' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 3' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num,
                      'test_just_test1_swarm_contour_mue.py 4' + ' ' + hyper_agent_num + ' ' + agent_num + ' ' + obs_num
                      )

    all_processes = all_processes1 + all_processes2

    process_pool = multiprocessing.Pool(processes=10)
    process_pool.map(execute, all_processes)

    # pool = Pool()
    # try:
    #     # for i in range(1, 3):
    #     #     pool.apply_async(run, args=[i,])
    #     pool.apply_async(run, args=[i, ])
    # finally:
    #     pool.close()
    #     pool.join()

