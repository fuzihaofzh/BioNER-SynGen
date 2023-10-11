from __future__ import absolute_import
import multiprocessing
#multiprocessing.set_start_method('spawn', True) #Slow, debug only
from multiprocessing import Process, Queue, Pool, Manager
import time
import sys
import queue
import datetime
import traceback
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
STREAM_LINE_TAG = "[Streamline]"

def func_wrapper(func, q_in, q_out, layer, proc_num, finished, batch_size, args, kwargs):
    if q_in is None:
        bucket = []
        for d in func(*args, **kwargs):
            bucket.append(d)
            tmp = finished[layer]
            if len(bucket) >= batch_size:
                q_out.put(bucket)
                with finished.lock:
                    finished[layer] += len(bucket)
                bucket = []
        if len(bucket) > 0:
            q_out.put(bucket)
            with finished.lock:
                finished[layer] += len(bucket)
    else:
        while True:
            try:
                bucket = []
                input_data = q_in.get(timeout = 2)
                for d in input_data:
                    bucket.append(func(d, *args, **kwargs))
                with finished.lock:
                    finished[layer] += len(input_data)
                q_out.put(bucket)
            except queue.Empty:
                with proc_num.lock:
                    #print("empty!", func.__name__, q_in.qsize(), proc_num[layer - 1])
                    if q_in.qsize() == 0 and proc_num[layer - 1] <= 0:
                        logging.info(f"{STREAM_LINE_TAG} break")
                        break
            except Exception as error:
                q_in.put(input_data)#return back
                logging.error(f"{STREAM_LINE_TAG} {func.__name__} Error!")
                logging.error(f"{STREAM_LINE_TAG} {traceback.print_exc()}")

    with proc_num.lock:            
        proc_num[layer] -= 1
    logging.info(f"{STREAM_LINE_TAG} Finish {str(func)}")

def show_progress(proc_num, modules, buffers, finished, batch_sizes):
    try:
        finished_prev = list(finished)
        st = time.time()
        st0 = st
        while sum(proc_num) > 0:
            log_str = ""
            has_progress = False
            for i in range(len(modules)):
                finish_inc = finished[i] - finished_prev[i]
                has_progress = True if finish_inc > 0 else False
                if modules[i] is not None:
                    log_str += "%s: %d, %s, %.1f/s; "%(modules[i].__name__, finished[i], str(buffers[i].qsize() * batch_sizes[i]) if buffers[i] and batch_sizes[i] else 'N/A', (finish_inc) / (time.time() - st))
            finished_prev = list(finished)
            st = time.time()
            if not has_progress:
                continue
            logging.info(f"{STREAM_LINE_TAG} {log_str}")
            time.sleep(1)
            #sys.stdout.write('\x1b[2K\r')
            #print("", end='\r')
            #print(proc_num, finished, [b.qsize() if b is not None else 'na' for b in buffers])
            logging.info(f"{STREAM_LINE_TAG} {str(proc_num)}")
            remain_seconds = (st - st0) / max(1, finished[-1]) * (finished[0] - finished[-1])
            #remain_time = time.strftime('%H:%M:%S', time.gmtime(remain_seconds))
            current_time = datetime.timedelta(seconds = int(st - st0))
            remain_time = datetime.timedelta(seconds = int(remain_seconds))
            logging.info(f"{STREAM_LINE_TAG} ETA: {current_time} < {remain_time}")
    except Exception as error:
        logging.error(f"{STREAM_LINE_TAG} show_progress Error!")
        
def _add_data_func(items):
    for item in items:
        yield item

class StreamLine():
    def __init__(self, batch_size = 1, debug = False):
        self.manager = Manager()
        self.modules = self._get_locked_list(self.manager)
        self.args = []
        self.kwargs = []
        self.buffers = [None]
        self.processes = []
        self.proc_num = self._get_locked_list(self.manager)
        self.finished = self._get_locked_list(self.manager)
        self.batch_sizes = self._get_locked_list(self.manager)
        self.default_batch_size = batch_size
        self.debug = debug

    def _get_locked_list(self, manager):
        l = manager.list()
        l.lock = manager.Lock()
        return l

    def add_module(self, func, proc_num = 1, batch_size = None, args = [], kwargs = {}):
        self.modules.append(func)
        self.proc_num.append(proc_num)
        self.buffers.append(self.manager.Queue())
        self.finished.append(0)
        if batch_size is None:
            batch_size = self.default_batch_size
        self.batch_sizes.append(batch_size)
        self.args.append(args)
        self.kwargs.append(kwargs)

    def add_data(self, items):
        self.add_module(_add_data_func, args=[items])

    def run(self):
        if self.debug:
            self.run_serial()
        else:
            self.run_parallel()

    def run_parallel(self):
        self.run_mode = "run"
        for i in range(len(self.modules)):
            for _ in range(self.proc_num[i]):
                c = Process(target=func_wrapper, args=(self.modules[i], self.buffers[i], self.buffers[i + 1], i, self.proc_num, self.finished, self.batch_sizes[i], self.args[i], self.kwargs[i]))
                c.start()
                self.processes.append(c)

        c = Process(target=show_progress, args = (self.proc_num, self.modules, self.buffers, self.finished, self.batch_sizes))
        c.start()
        self.processes.append(c)
        
    def run_serial(self):
        self.run_mode = "run_serial"
        cnt = 0
        for d in self.modules[0](*self.args[0], **self.kwargs[0]):
            for i in range(1, len(self.modules)):
                #print(self.modules[i].__name__, len(d))
                d = self.modules[i](d, *self.args[i], **self.kwargs[i])
            self.buffers[-1].put(d)
            logging.info(f"{STREAM_LINE_TAG} Finish: {cnt}")
            cnt += 1
        
    def join(self):
        for p in self.processes:
            p.join()
            logging.info(f"{STREAM_LINE_TAG} {str(p)} Finish")

    def get_results(self):
        results = []
        while self.buffers[-1].qsize() > 0:
            res = self.buffers[-1].get()
            if self.run_mode == "run_serial":
                results.append(res)
            else:
                results += res
        return results #[r[0] for r in results]

    def get_finish_count(self):
        return self.buffers[-1].qsize() * self.batch_sizes[-1]



    
#=============================TEST=============================

def f1():
    import time
    for i in range(10):
        #time.sleep(0.00000000000000000002)
        yield i * 2

def f2(n, add, third = 0.01):
    #time.sleep(0.000000000000000001)
    return n + add + third

def f3(n):
    #time.sleep(0.00000000000000000001)
    return n + 1

if __name__ == "__main__":
    sl = StreamLine(debug = False)
    sl.add_module(f1, 1)
    sl.add_module(f2, 2, args = [0.5], kwargs = {'third' : 0.02})
    sl.add_module(f3, 2)
    sl.run()
    sl.join()
    show_progress(sl.proc_num, sl.modules, sl.buffers, sl.finished, sl.batch_sizes)
    print(sl.get_results())

