""" Tools for creating pools of worker processes
"""
import multiprocessing.pool
import functools
from multiprocessing import TimeoutError, cpu_count, get_context
import types
import signal
import atexit
import logging

logger = logging.getLogger('pycbc.pool')

def is_main_process():
    """ Check if this is the main control process and may handle one time tasks
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        return rank == 0
    except (ImportError, ValueError, RuntimeError):
        return True

# Allow the pool to be interrupted, need to disable the children processes
# from intercepting the keyboard interrupt
def _noint(init, *args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if init is not None:
        return init(*args)

_process_lock = None
_numdone = None
def _lockstep_fcn(values):
    """ Wrapper to ensure that all processes execute together """
    numrequired, fcn, args = values
    with _process_lock:
        _numdone.value += 1
    # yep this is an ugly busy loop, do something better please
    # when we care about the performance of this call and not just the
    # guarantee it provides (ok... maybe never)
    while 1:
        if _numdone.value == numrequired:
            return fcn(args)

def _shutdown_pool(p):
    p.terminate()
    p.join()

class BroadcastPool(multiprocessing.pool.Pool):
    """ Multiprocessing pool with a broadcast method
    """
    def __init__(self, processes=None, initializer=None, initargs=(),
                 context=None, **kwds):
        global _process_lock
        global _numdone
        _process_lock = multiprocessing.Lock()
        _numdone = multiprocessing.Value('i', 0)
        noint = functools.partial(_noint, initializer)

        # Default is fork to preserve child memory inheritance and
        # copy on write
        if context is None:
            context = get_context("fork")
        super(BroadcastPool, self).__init__(processes, noint, initargs,
                                            context=context, **kwds)
        atexit.register(_shutdown_pool, self)

    def __len__(self):
        return len(self._pool)

    def broadcast(self, fcn, args):
        """ Do a function call on every worker.

        Parameters
        ----------
        fcn: funtion
            Function to call.
        args: tuple
            The arguments for Pool.map
        """
        results = self.map(_lockstep_fcn, [(len(self), fcn, args)] * len(self))
        _numdone.value = 0
        return results

    def allmap(self, fcn, args):
        """ Do a function call on every worker with different arguments

        Parameters
        ----------
        fcn: funtion
            Function to call.
        args: tuple
            The arguments for Pool.map
        """
        results = self.map(_lockstep_fcn,
                           [(len(self), fcn, arg) for arg in args])
        _numdone.value = 0
        return results

    def map(self, func, items, chunksize=None):
        """ Catch keyboard interrupts to allow the pool to exit cleanly.

        Parameters
        ----------
        func: function
            Function to call
        items: list of tuples
            Arguments to pass
        chunksize: int, Optional
            Number of calls for each process to handle at once
        """
        results = self.map_async(func, items, chunksize)
        while True:
            try:
                return results.get(1800)
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate()
                self.join()
                raise KeyboardInterrupt

    def close_pool(self):
        """ Close the pool and remove the reference
        """
        self.close()
        self.join()
        atexit.unregister(_shutdown_pool)

def _dummy_broadcast(self, f, args):
    self.map(f, [args] * self.size)

class SinglePool(object):

    def __init__(self, **_):
        pass

    def broadcast(self, fcn, args):
        return self.map(fcn, [args])

    def map(self, f, items):
        return [f(a) for a in items]

    # This is single core, so imap and map
    # would not behave differently. This is defined
    # so that the general pool interfaces can use
    # imap irrespective of the pool type. 
    imap = map
    imap_unordered = map

    def close_pool(self):
        ''' Dummy function to be consistent with BroadcastPool
        '''
        pass

def use_mpi(require_mpi=False, log=True):
    """ Get whether MPI is enabled and if so the current size and rank
    """
    use_mpi = False
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if size > 1:
            use_mpi = True
            if log:
                logger.info(
                    'Running under mpi with size: %s, rank: %s',
                    size, rank
                )
    except ImportError as e:
        if require_mpi:
            print(e)
            raise ValueError("Failed to load mpi, ensure mpi4py is installed")
    if not use_mpi:
        size = rank = 0
    return use_mpi, size, rank


def choose_pool(processes, mpi=False, **kwargs):
    """ Get processing pool.

    Keyword arguments are passed to the pool constructor.
    """
    do_mpi, size, rank = use_mpi(require_mpi=mpi)
    if do_mpi:
        try:
            import schwimmbad
            pool = schwimmbad.choose_pool(mpi=do_mpi,
                                          processes=(size - 1),
                                          **kwargs)
            pool.broadcast = types.MethodType(_dummy_broadcast, pool)
            atexit.register(pool.close)

            if processes:
                logger.info('NOTE: that for MPI process size determined by '
                            'MPI launch size, not the processes argument')

            if do_mpi and not mpi:
                logger.info('NOTE: using MPI as this process was launched'
                            'under MPI')
        except ImportError:
            raise ValueError("Failed to start up an MPI pool, "
                             "install mpi4py / schwimmbad")
    elif processes == 1:
        pool = SinglePool(**kwargs)
    else:
        if processes == -1:
            processes = cpu_count()
        pool = BroadcastPool(processes, **kwargs)

    pool.size = processes
    if size:
        pool.size = size
    return pool

