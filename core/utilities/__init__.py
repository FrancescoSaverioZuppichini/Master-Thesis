import time

def run_for(seconds, *args, **kwargs):
    def run_for_seconds(func, *args, **kwargs):
        elapsed = 0
        start = time.time()
        while elapsed <= seconds:
            elapsed = time.time() - start
            func(*args, **kwargs)

    return run_for_seconds

