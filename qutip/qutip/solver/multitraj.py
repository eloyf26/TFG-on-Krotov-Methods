from .. import Qobj, QobjEvo
from .result import Result, MultiTrajResult
from .parallel import _get_map
from time import time
from .solver_base import Solver
import numpy as np
from copy import copy

__all__ = ["MultiTrajSolver", "TrajectorySolver"]


class MultiTrajSolver(Solver):
    """
    Basic class for multi-trajectory evolutions.

    As :class:`Solver` it can ``run`` or ``step`` evolution.
    It manages the random seed for each trajectory.

    The actual evolution is done by a single trajectory solver::
        ``_traj_solver_class``

    Parameters
    ----------
    rhs : Qobj, QobjEvo
        Right hand side of the evolution::
            d state / dt = rhs @ state

    options : dict
        Options for the solver.
    """
    name = "generic multi trajectory"
    resultclass = MultiTrajResult
    _avail_integrators = {}

    # Class of option used by the solver
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
    }

    def __init__(self, rhs, *, options=None):
        self.rhs = rhs
        self.options = options
        self.seed_sequence = np.random.SeedSequence()
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()

    def start(self, state, t0, seed=None):
        """
        Set the initial state and time for a step evolution.

        Parameters
        ----------
        state : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        seed : int, SeedSequence, list, {None}
            Seed for the random number generator. It can be a single seed used
            to spawn seeds for each trajectory or a list of seed, one for each
            trajectory.

        ..note ::
            When using step evolution, only one trajectory can be computed at
            once.
        """
        seeds = self._read_seed(seed, 1)
        generator = self._get_generator(seeds[0])
        self._integrator.set_state(t0, self._prepare_state(state), generator)

    def step(self, t, *, args=None, copy=True):
        """
        Evolve the state to ``t`` and return the state as a :class:`Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        args : dict, optional {None}
            Update the ``args`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``args`` can slow the evolution.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.
        """
        if not self._integrator._is_set:
            raise RuntimeError("The `start` method must called first.")
        self._argument(args)
        _, state = self._integrator.integrate(t, copy=False)
        return self._restore_state(state, copy=copy)

    def run(self, state, tlist, ntraj=1, *,
            args=None, e_ops=(), timeout=None, target_tol=None, seed=None):
        """
        Do the evolution of the Quantum system.

        For a ``state`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :cls:`Result`. The evolution method and stored
        results are determined by ``options``.

        Parameters
        ----------
        state : :class:`Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Time in the list must be in increasing order, but does
            not need to be uniformly distributed.

        ntraj : int
            Number of trajectories to add.

        args : dict, optional {None}
            Change the ``args`` of the rhs for the evolution.

        e_ops : list
            list of Qobj or QobjEvo to compute the expectation values.
            Alternatively, function[s] with the signature f(t, state) -> expect
            can be used.

        timeout : float, optional [1e8]
            Maximum time in seconds for the trajectories to run. Once this time
            is reached, the simulation will end even if the number
            of trajectories is less than ``ntraj``. The map function, set in
            options, can interupt the running trajectory or wait for it to
            finish. Set to an arbitrary high number to disable.

        target_tol : {float, tuple, list}, optional [None]
            Target tolerance of the evolution. The evolution will compute
            trajectories until the error on the expectation values is lower
            than this tolerance. The maximum number of trajectories employed is
            given by ``ntraj``. The error is computed using jackknife
            resampling. ``target_tol`` can be an absolute tolerance or a pair
            of absolute and relative tolerance, in that order. Lastly, it can
            be a list of pairs of (atol, rtol) for each e_ops.

        seed : {int, SeedSequence, list} optional
            Seed or list of seeds for each trajectories.

        Return
        ------
        results : :class:`qutip.solver.MultiTrajResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.

        .. note:
            The simulation will end when the first end condition is reached
            between ``ntraj``, ``timeout`` and ``target_tol``.
        """
        start_time = time()
        self._argument(args)
        stats = self._initialize_stats()
        seeds = self._read_seed(seed, ntraj)

        result = self.resultclass(
            e_ops, self.options, solver=self.name, stats=stats
        )
        result.add_end_condition(ntraj, target_tol)

        map_func = _get_map[self.options['map']]
        map_kw = {
            'timeout': timeout,
            'job_timeout': self.options['job_timeout'],
            'num_cpus': self.options['num_cpus'],
        }
        state0 = self._prepare_state(state)
        stats['preparation time'] += time() - start_time

        start_time = time()
        map_func(
            self._run_one_traj, seeds,
            (state0, tlist, e_ops),
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] = time() - start_time
        return result

    def _run_one_traj(self, seed, state, tlist, e_ops):
        """
        Run one trajectory and return the result.
        """
        result = Result(e_ops, self.options)
        generator = self._get_generator(seed)
        self._integrator.set_state(tlist[0], state, generator)
        result.add(tlist[0], self._restore_state(state, copy=False))
        for t in tlist[1:]:
            t, state = self._integrator.step(t, copy=False)
            result.add(t, self._restore_state(state, copy=False))
        return seed, result

    def _read_seed(self, seed, ntraj):
        """
        Read user provided seed(s) and produce one for each trajectory.
        Let numpy raise error for inputs that cannot be seeds.
        """
        if seed is None:
            seeds = self.seed_sequence.spawn(ntraj)
        elif isinstance(seed, np.random.SeedSequence):
            seeds = seed.spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = np.random.SeedSequence(seed).spawn(ntraj)
        elif len(seed) >= ntraj:
            seeds = [
                seed_ if (isinstance(seed_, np.random.SeedSequence)
                          or hasattr(seed_, 'random'))
                else np.random.SeedSequence(seed_)
                for seed_ in seed[:ntraj]
            ]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def _argument(self, args):
        """Update the args, for the `rhs` and `c_ops` and other operators."""
        if args:
            self.rhs.arguments(args)

    def _get_generator(self, seed):
        """
        Read the seed and create the random number generator.
        If the ``seed`` has a ``random`` method, it will be used as the
        generator.
        """
        if hasattr(seed, 'random'):
            # We check for the method, not the type to accept pseudo non-random
            # generator for debug/testing purpose.
            return seed

        if self.options['bitgenerator']:
            bit_gen = getattr(np.random, self.options['bitgenerator'])
            generator = np.random.Generator(bit_gen(seed))
        else:
            generator = np.random.default_rng(seed)
        return generator

    @classmethod
    def avail_integrators(cls):
        return cls._avail_integrators
