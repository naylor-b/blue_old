"""Various profiling functions."""


import cProfile
import pstats
import subprocess
from contextlib import contextmanager

from openmdao.devtools.pstats_viewer import view_pstats

@contextmanager
def profiling(fname='prof_out', port=8009, view=True):
    """
    Turns on profiling for a section of code.

    Parameters
    ----------
    fname : str
        Name of file to write profile data to.
    port : int
        Port used by profile viewer.
    view : bool
        If True, pop up interactive profile viewer
    """
    prof = cProfile.Profile()
    prof.enable()
    yield prof
    prof.disable()

    prof.dump_stats(fname)

    if view:
        view_pstats(fname, port)
