"""Define the NewtonSolver class."""
from __future__ import print_function

from openmdao.solvers.solver import NonlinearSolver


class NewtonSolver(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the ln_solver in the containing system.

    Attributes
    ----------
    ln_solver : <LinearSolver>
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : <NonlinearSolver>
        Line search algorithm. Default is None for no line search.
    options : <OptionsDictionary>
        options dictionary.
    _system : <System>
        pointer to the owning system.
    _depth : int
        how many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        list of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        number of iterations for the current invocation of the solver.
    """

    SOLVER = 'NL: Newton'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(NewtonSolver, self).__init__(**kwargs)

        # Slot for linear solver
        self.ln_solver = None

        # Slot for linesearch
        self.linesearch = None

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(NewtonSolver, self)._setup_solvers(system, depth)

        if self.ln_solver is not None:
            self.ln_solver._setup_solvers(self._system, self._depth + 1)
        else:
            self.ln_solver = system.ln_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(self._system, self._depth + 1)

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.ln_solver is not None:
            self.ln_solver._linearize()

        if self.linesearch is not None:
            self.linesearch._linearize()

    def _iter_initialize(self):
        system = self._system

        for isub, subsys in enumerate(system._subsystems_allprocs):
            system._transfers['fwd', isub](system._inputs,
                                           system._outputs, 'fwd')

            if subsys in system._subsystems_myproc:
                subsys._solve_nonlinear()

        return super(NewtonSolver, self)._iter_initialize()

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        system._vectors['residual']['linear'].set_vec(system._residuals)
        system._vectors['residual']['linear'] *= -1.0
        system._linearize()
        self.ln_solver.solve(['linear'], 'fwd')
        if self.linesearch:
            self.linesearch.solve()
        else:
            system._outputs += system._vectors['output']['linear']

        for isub, subsys in enumerate(system._subsystems_allprocs):
            system._transfers['fwd', isub](system._inputs,
                                           system._outputs, 'fwd')

            if subsys in system._subsystems_myproc:
                subsys._solve_nonlinear()

        # print(system.pathname)
        # outputs = system._vectors['output']['nonlinear']
        # resids = system._vectors['residual']['nonlinear']
        # import numpy as np
        # super(NewtonSolver, self)._iter_initialize()
        # for var_name in outputs:
        #     print(var_name, np.sum(outputs[var_name]), np.linalg.norm(resids[var_name]))
        # print()
