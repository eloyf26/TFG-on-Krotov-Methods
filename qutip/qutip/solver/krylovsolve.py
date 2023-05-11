__all__ = ['krylovsolve']

from .. import QobjEvo
from .sesolve import SESolver


def krylovsolve(
    H, psi0, tlist, krylov_dim, e_ops=None, args=None, options=None
):
    """
    Schrodinger equation evolution of a state vector for time independent
    Hamiltonians using Krylov method.

    Evolve the state vector ("psi0") finding an approximation for the time
    evolution operator of Hamiltonian ("H") by obtaining the projection of
    the time evolution operator on a set of small dimensional Krylov
    subspaces (m << dim(H)).

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    krylov_dim: int
        Dimension of Krylov approximation subspaces used for the time
        evolution approximation.

    e_ops : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : None / dict
        Dictionary of options for the solver.

        - store_final_state : bool, [False]
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, [None]
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - normalize_output : bool, [True]
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}, ["text"]
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict, [{"chunk_size": 10}]
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - atol: float [1e-7]
          Absolute and relative tolerance of the ODE integrator.
        - nsteps : int [100]
          Maximum number of (internally defined) steps allowed in one ``tlist``
          step.
        - min_step, max_step : float, [1e-5, 1e5]
          Miniumum and maximum lenght of one internal step.
        - always_compute_step: bool [False]
          If True, the step lenght is computed each time a new Krylov
          subspace is computed. Otherwise it is computed only once when
          creating the integrator.
        - sub_system_tol: float, [1e-7]
          Tolerance to detect an happy breakdown. An happy breakdown happens
          when the initial ket is in a subspace of the Hamiltonian smaller
          than ``krylov_dim``.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        a *list of array* `result.expect` of expectation values for the times
        specified by `tlist`, and/or a *list* `result.states` of state vectors
        or density matrices corresponding to the times in `tlist` [if `e_ops`
        is an empty list of `store_states=True` in options].
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    options = options or {}
    options["method"] = "krylov"
    options["krylov_dim"] = krylov_dim
    solver = SESolver(H, options=options)
    return solver.run(psi0, tlist, e_ops=e_ops)
