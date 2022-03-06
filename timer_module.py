import time


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__:<35} {"executed in":^20} {(time.time()-start_time):>.5f}s')
        return result
    return wrap_func