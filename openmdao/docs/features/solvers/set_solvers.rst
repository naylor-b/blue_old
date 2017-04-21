:orphan:

.. _set-solvers:

Setting Nonlinear and Linear Solvers
=====================================

A nonlinear solver, like :ref:`NonlinearBlockGS <usr_openmdao.solvers.nl_bgs.py>` or :ref:`Newton <usr_openmdao.solvers.nl_newton.py>`,
is used to converge the nonlinear analysis. A nonlinear solver is needed whenever this is either a cyclic dependency between components in your model.
It might also be needed if you have an :ref:`ImplicitComponent <usr_openmdao.core.implicitcomponent.py>` in your model that expects the framework to handle its convergence.

Whenever you use a nonlinear solver on a :ref:`Group <usr_openmdao.core.group.py>` or :ref:`Component <usr_openmdao.core.component.py>`, if you're going to be working with analytic derivatives,
you will also need a linear solver.
A linear solver, like :ref:`LinearBlockGS <usr_openmdao.solvers.ln_bgs.py>` or :ref:`DirectSolver <usr_openmdao.solvers.ln_direct.py>`,
is used to solve the linear system that provides total derivatives across the model.

You can add nonlinear and linear solvers at any level of the model hierarchy,
letting you build a hierarchical solver setup to efficiently converge your model and solve for total derivatives across it.


Solvers for the Sellar Problem
----------------------------------

The Sellar Problem has two components with a cyclic dependency, so the appropriate nonlinear solver is necessary.
We'll use the :ref:`Newton <usr_openmdao.solvers.nl_newton.py>` nonlinear solver,
which requires derivatives so we'll also use the :ref:`Direct <usr_openmdao.solvers.ln_direct.py>` linear solver

.. embed-test::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_solver

----

Some models have more complex coupling. There could be top level cycles between groups as well as
lower level groups that have cycles of their own. The openmdao.test_suite.components.double_sellar.DoubleSellar (TODO: Link to problem page)
is a simple example of this kind of model structure. In these problems, you might want to specify a more complex hierarchical solver structure for both nonlinear and linear solvers.

.. embed-test::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_subgroup_solvers


.. note::
    Preconditioning for iterative linear solvers is complex topic.
    The structure of the preconditioner should follow the model hierarchy itself,
    but developing an effective and efficient preconditioner is not trivial.
    If you're having trouble converging the linear solves with an iterative solver,
    you should try using the :ref:`Direct <usr_openmdao.solvers.ln_direct.py>` solver instead.
    But first, verify that all your partials derivatives are correct with the check_partial_derivs method.


----

You can also specify solvers as part of the initialization of a Group

.. embed-code::
    openmdao.test_suite.components.double_sellar.DoubleSellar

.. tags:: Solver