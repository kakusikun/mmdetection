import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import time
import uuid
import queue
import logging
import cv2


def test_input_output(idx, qin, qout):
    print(f'{idx} start')
    duration = 0
    count = 0
    while True:
        start = time.time()
        tensor = qin.get()
        print(f"{idx} get")
        tensor = torch.nn.functional.interpolate(tensor, size=(640,352))
        qout.put(tensor)
        print(f"{idx} put")
        duration += time.time()-start
        count += 1
        if count == 10:
            break
    time.sleep(3)
    print(f'{idx} avg {(duration/count)*1000:.1f}ms')

def test_output(idx, qout):
    print(f'{idx} start')
    duration = 0
    for _ in range(10):
        start = time.time()
        data = np.ones((1920,1080,3), dtype=np.uint8)
        data = cv2.resize(data, (352,640))
        tensor = torch.from_numpy(data).cuda()
        tensor = tensor.permute(2,0,1).unsqueeze(0).share_memory_()
        qout.put(tensor)
        del tensor
        duration += time.time()-start
    print(f'{idx} avg {(duration/10)*1000:.1f}ms')
    time.sleep(60)
        

def main_input_output():
    qins = []
    qouts = []
    ps = []
    inputs = []
    for i in range(5):
        qin = mp.Queue(10)
        qout = mp.Queue(10)
        p = mp.Process(target=test_input_output, args=(i, qin, qout,))
        a = torch.rand(1,3,1920,1080).cuda()
        p.start()
        ps.append(p)
        qins.append(qin)
        qouts.append(qout)
        inputs.append(a)

    time.sleep(1)
    start = time.time()
    for _ in range(10):
        for a, qin in zip(inputs, qins):
            a.share_memory_()
            qin.put(a)
        for qout in qouts:
            tensor = qout.get()
            
            # print('shared', a.is_shared())
            # print("put")
    print('put total time', (time.time() - start)/10)

    for p in ps:
        p.join()
        p.close()

def main_output():
    qouts = []
    ps = []
    for i in range(5):
        qout = mp.Queue(10)
        p = mp.Process(target=test_output, args=(i, qout,))
        p.start()
        ps.append(p)
        qouts.append(qout)

    time.sleep(1)
    start = time.time()
    count = 0
    while True:
        for qout in qouts:
            try:
                tensor = qout.get_nowait()
                count += 1
            except queue.Empty:
                pass
        if count == 50:
            break
    print('get total time', (time.time() - start)/50)

    for p in ps:
        p.join()
        p.close()

if __name__ == "__main__":    
    main_output()