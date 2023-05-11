from qutip import liouvillian, lindblad_dissipator, Qobj, qzero_like, qeye_like
from qutip import vector_to_operator, operator_to_vector
from qutip import settings
import qutip.core.data as _data
import numpy as np
import scipy.sparse.csgraph
import scipy.sparse.linalg
from warnings import warn


__all__ = ["steadystate", "steadystate_floquet", "pseudo_inverse"]


def _permute_wbm(L, b):
    perm = scipy.sparse.csgraph.maximum_bipartite_matching(L.as_scipy())
    L = _data.permute.indices(L, perm, None)
    b = _data.permute.indices(b, perm, None)
    return L, b


def _permute_rcm(L, b):
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(L.as_scipy())
    L = _data.permute.indices(L, perm, perm)
    b = _data.permute.indices(b, perm, None)
    return L, b, perm


def _reverse_rcm(rho, perm):
    rev_perm = np.argsort(perm)
    rho = _data.permute.indices(rho, rev_perm, None)
    return rho


def steadystate(A, c_ops=[], *, method='direct', solver=None, **kwargs):
    """
    Calculates the steady state for quantum evolution subject to the supplied
    Hamiltonian or Liouvillian operator and (if given a Hamiltonian) a list of
    collapse operators.

    If the user passes a Hamiltonian then it, along with the list of collapse
    operators, will be converted into a Liouvillian operator in Lindblad form.

    Parameters
    ----------
    A : :obj:`~Qobj`
        A Hamiltonian or Liouvillian operator.

    c_op_list : list
        A list of collapse operators.

    method : str, default='direct'
        The allowed methods are composed of 2 parts, the steadystate method:
        - "direct": Solving ``L(rho_ss) = 0``
        - "eigen" : Eigenvalue problem
        - "svd" : Singular value decomposition
        - "power" : Inverse-power method

    solver : str, default=None
        'direct' and 'power' methods only.
        Solver to use when solving the ``L(rho_ss) = 0`` equation.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve"
          sparse solver by mkl.

        Extension to qutip, such as qutip-tensorflow, can use come with their
        own solver. When ``A`` and ``c_ops`` use these data backends, see the
        corresponding libraries ``linalg`` for available solver.

        Extra options for these solver can be passed in ``**kw``.

    use_rcm : bool, default False
        Use reverse Cuthill-Mckee reordering to minimize fill-in in the LU
        factorization of the Liouvillian.
        Used with 'direct' or 'power' method.

    use_wbm : bool, default False
        Use Weighted Bipartite Matching reordering to make the Liouvillian
        diagonally dominant.  This is useful for iterative preconditioners
        only. Used with 'direct' or 'power' method.

    weight : float, optional
        Sets the size of the elements used for adding the unity trace condition
        to the linear solvers.  This is set to the average abs value of the
        Liouvillian elements if not specified by the user.
        Used with 'direct' method.

    power_tol : float, default 1e-12
        Tolerance for the solution when using the 'power' method.

    power_maxiter : int, default 10
        Maximum number of iteration to use when looking for a solution when
        using the 'power' method.

    power_eps: double, default 1e-15
        Small weight used in the "power" method.

    sparse: bool
        Whether to use the sparse eigen solver with the "eigen" method
        (default sparse).  With "direct" and "power" method, when the solver is
        not specified, it is used to set whether "solve" or "spsolve" is
        used as default solver.

    **kwargs :
        Extra options to pass to the linear system solver. See the
        documentation of the used solver in ``numpy.linalg`` or
        ``scipy.sparse.linalg`` to see what extra arguments are supported.

    Returns
    -------
    dm : qobj
        Steady state density matrix.
    info : dict, optional
        Dictionary containing solver-specific information about the solution.

    .. note::

        The SVD method works only for dense operators (i.e. small systems).

    """
    if not A.issuper and not c_ops:
        raise TypeError('Cannot calculate the steady state for a ' +
                        'non-dissipative system.')
    if not A.issuper:
        A = liouvillian(A)
    for op in c_ops:
        A += lindblad_dissipator(op)

    if "-" in method:
        # to support v4's "power-gmres" method
        method, solver = method.split("-")

    if solver == "mkl":
        solver = "mkl_spsolve"

    # Keys supported in v4, but removed in v5
    if kwargs.pop("return_info", False):
        warn("Steadystate no longer supports return_info", DeprecationWarning)
    if "mtol" in kwargs and "power_tol" not in kwargs:
        kwargs["power_tol"] = kwargs["mtol"]
    kwargs.pop("mtol", None)

    if method == "eigen":
        return _steadystate_eigen(A, **kwargs)
    if method == "svd":
        return _steadystate_svd(A, **kwargs)

    # We want to be able to use this without having to know what data type the
    # liouvillian uses. For extra data types (tensorflow) we can expect
    # the users to know they are using them and choose an appropriate solver
    sparse_solvers = ["spsolve", "mkl_spsolve", "gmres", "lgmres", "bicgstab"]
    if not isinstance(A.data, (_data.CSR, _data.Dense)):
        # Tensorflow, jax, etc. data type
        pass
    elif isinstance(A.data, _data.CSR) and solver in ["solve", "lstsq"]:
        A = A.to("dense")
    elif isinstance(A.data, _data.Dense) and solver in sparse_solvers:
        A = A.to("csr")
    elif solver is None and kwargs.get("sparse", False):
        A = A.to("csr")
        solver = "mkl_spsolve" if settings.has_mkl else "spsolve"
    elif solver is None and (kwargs.get("sparse", None) is False):
        # sparse is explicitly set to false, v4 tag to use `numpy.linalg.solve`
        A = A.to("dense")
        solver = "solve"

    if method in ["direct", "iterative"]:
        # Remove unused kwargs, so only used and pass-through ones are included
        kwargs.pop("power_tol", 0)
        kwargs.pop("power_maxiter", 0)
        kwargs.pop("power_eps", 0)
        kwargs.pop("sparse", 0)
        return _steadystate_direct(A, kwargs.pop("weight", 0),
                                   method=solver, **kwargs)

    elif method == "power":
        # Remove unused kwargs, so only used and pass-through ones are included
        kwargs.pop("weight", 0)
        kwargs.pop("sparse", 0)
        return _steadystate_power(A, method=solver, **kwargs)
    else:
        raise ValueError(f"method {method} not supported.")


def _steadystate_direct(A, weight, **kw):
    # Find the weight, no good dispatched function available...
    if weight:
        pass
    elif isinstance(A.data, _data.CSR):
        weight = np.mean(np.abs(A.data.as_scipy().data))
    else:
        A_np = np.abs(A.full())
        weight = np.mean(A_np[A_np > 0])

    # Add weight to the Liouvillian
    # A[:, 0] = vectorized(eye * weight)
    # We don't have a function to overwrite part of an array, so
    N = A.shape[0]
    n = int(N**0.5)
    dtype = type(A.data)
    weight_vec = _data.column_stack(_data.diag([weight] * n, 0, dtype=dtype))
    weight_mat = _data.kron(
        weight_vec.transpose(),
        _data.one_element[dtype]((N, 1), (0, 0), 1)
    )
    L = _data.add(weight_mat, A.data)
    b = _data.one_element[dtype]((N, 1), (0, 0), weight)

    # Permutation are part of scipy.sparse, thus only supported for CSR.
    if kw.pop("use_wbm", False):
        if isinstance(L, _data.CSR):
            L, b = _permute_wbm(L, b)
        else:
            warn("Only sparse matrice can be permuted.", RuntimeWarning)
    use_rcm = False
    if kw.pop("use_rcm", False):
        if isinstance(L, _data.CSR):
            L, b, perm = _permute_rcm(L, b)
            use_rcm = True
        else:
            warn("Only sparse matrice can be permuted.", RuntimeWarning)
    if kw.pop("use_precond", False):
        if isinstance(L, _data.CSR):
            kw["M"] = _compute_precond(L, kw)
        else:
            warn("Only sparse solver use preconditioners.", RuntimeWarning)


    method = kw.pop("method", None)
    steadystate = _data.solve(L, b, method, options=kw)

    if use_rcm:
        steadystate = _reverse_rcm(steadystate, perm)

    rho_ss = _data.column_unstack(steadystate, n)
    rho_ss = _data.add(rho_ss, rho_ss.adjoint()) * 0.5

    return Qobj(rho_ss, dims=A.dims[0], isherm=True)


def _steadystate_eigen(L, **kw):
    val, vec = (L.dag() @ L).eigenstates(
        eigvals=1,
        sort="low",
        # v4's implementation only uses sparse eigen solver
        sparse=kw.pop("sparse", True)
    )
    rho = vector_to_operator(vec[0])
    return rho / rho.tr()


def _steadystate_svd(L, **kw):
    u, s, vh = _data.svd(L.data, True)
    vec = Qobj(_data.split_columns(vh.adjoint())[-1], dims=[L.dims[0],[1]])
    rho = vector_to_operator(vec)
    return rho / rho.tr()


def _steadystate_power(A, **kw):
    A += kw.pop("power_eps", 1e-15)
    L = A.data
    N = L.shape[1]
    y = _data.Dense([1]*N)

    # Permutation are part of scipy.sparse, thus only supported for CSR.
    if kw.pop("use_wbm", False):
        if isinstance(L, _data.CSR):
            L, y = _permute_wbm(L, y)
        else:
            warn("Only sparse matrice can be permuted.", RuntimeWarning)
    use_rcm = False
    if kw.pop("use_rcm", False):
        if isinstance(L, _data.CSR):
            L, y, perm = _permute_rcm(L, y)
            use_rcm = True
        else:
            warn("Only sparse matrice can be permuted.", RuntimeWarning)
    if kw.pop("use_precond", False):
        if isinstance(L, _data.CSR):
            kw["M"] = _compute_precond(L, kw)
        else:
            warn("Only sparse solver use preconditioners.", RuntimeWarning)

    it = 0
    maxiter = kw.pop("power_maxiter", 10)
    tol = kw.pop("power_tol", 1e-12)
    method = kw.pop("method", None)
    while it < maxiter and _data.norm.max(L @ y) > tol:
        y = _data.solve(L, y, method, options=kw)
        y = y / _data.norm.max(y)
        it += 1

    if it >= maxiter:
        raise Exception('Failed to find steady state after ' +
                        str(maxiter) + ' iterations')

    if use_rcm:
        y = _reverse_rcm(y, perm)

    rho_ss = Qobj(_data.column_unstack(y, N**0.5), dims=A.dims[0])
    rho_ss = rho_ss + rho_ss.dag()
    rho_ss = rho_ss / rho_ss.tr()
    rho_ss.isherm = True
    return rho_ss


def steadystate_floquet(H_0, c_ops, Op_t, w_d=1.0, n_it=3, sparse=False,
                        solver=None, **kwargs):
    """
    Calculates the effective steady state for a driven
     system with a time-dependent cosinusoidal term:

    .. math::

        \\mathcal{\\hat{H}}(t) = \\hat{H}_0 +
         \\mathcal{\\hat{O}} \\cos(\\omega_d t)

    Parameters
    ----------
    H_0 : :obj:`~Qobj`
        A Hamiltonian or Liouvillian operator.

    c_ops : list
        A list of collapse operators.

    Op_t : :obj:`~Qobj`
        The the interaction operator which is multiplied by the cosine

    w_d : float, default 1.0
        The frequency of the drive

    n_it : int, default 3
        The number of iterations for the solver

    sparse : bool, default False
        Solve for the steady state using sparse algorithms.

    solver : str, default=None
        Solver to use when solving the linear system.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve"
          sparse solver by mkl.

        Extensions to qutip, such as qutip-tensorflow, may provide their own solvers.
        When ``H_0`` and ``c_ops`` use these data backends, see their documentation
        for the names and details of additional solvers they may provide.

    **kwargs:
        Extra options to pass to the linear system solver. See the
        documentation of the used solver in ``numpy.linalg`` or
        ``scipy.sparse.linalg`` to see what extra arguments are supported.

    Returns
    -------
    dm : qobj
        Steady state density matrix.

    .. note::

        See: Sze Meng Tan,
        https://copilot.caltech.edu/documents/16743/qousersguide.pdf,
        Section (10.16)

    """

    L_0 = liouvillian(H_0, c_ops)
    L_m = L_p = 0.5 * liouvillian(Op_t)
    # L_p and L_m correspond to the positive and negative
    # frequency terms respectively.
    # They are independent in the model, so we keep both names.
    Id = qeye_like(L_0)
    S = T = qzero_like(L_0)

    if isinstance(H_0.data, _data.CSR) and not sparse:
        L_0 = L_0.to("Dense")
        L_m = L_m.to("Dense")
        L_p = L_p.to("Dense")
        Id = Id.to("Dense")

    for n_i in np.arange(n_it, 0, -1):
        L = L_0 - 1j * n_i * w_d * Id + L_m @ S
        S.data = - _data.solve(L.data, L_p.data, solver, kwargs)
        L = L_0 - 1j * n_i * w_d * Id + L_p @ T
        T.data = - _data.solve(L.data, L_m.data, solver, kwargs)

    M_subs = L_0 + L_m @ S + L_p @ T
    return steadystate(M_subs, solver=solver, **kwargs)


def pseudo_inverse(L, rhoss=None, w=None, method='splu', *, use_rcm=False,
                   **kwargs):
    """
    Compute the pseudo inverse for a Liouvillian superoperator, optionally
    given its steady state density matrix (which will be computed if not
    given).

    Parameters
    ----------
    L : Qobj
        A Liouvillian superoperator for which to compute the pseudo inverse.

    rhoss : Qobj
        A steadystate density matrix as Qobj instance, for the Liouvillian
        superoperator L.

    w : double
        frequency at which to evaluate pseudo-inverse.  Can be zero for dense
        systems and large sparse systems. Small sparse systems can fail for
        zero frequencies.

    sparse : bool
        Flag that indicate whether to use sparse or dense matrix methods when
        computing the pseudo inverse.

    method : string
        Method used to compte matrix inverse.
        Choice are 'pinv' to use scipy's function of the same name, or a linear
        system solver.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab", "splu"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve",
          sparse solver by mkl.

        Extension to qutip, such as qutip-tensorflow, can use come with their
        own solver. When ``L`` use these data backends, see the corresponding
        libraries ``linalg`` for available solver.

    kwargs : dictionary
        Additional keyword arguments for setting parameters for solver methods.

    Returns
    -------
    R : Qobj
        Returns a Qobj instance representing the pseudo inverse of L.

    .. note::

        In general the inverse of a sparse matrix will be dense.  If you
        are applying the inverse to a density matrix then it is better to
        cast the problem as an Ax=b type problem where the explicit calculation
        of the inverse is not required. See page 67 of "Electrons in
        nanostructures" C. Flindt, PhD Thesis available online:
        https://orbit.dtu.dk/fedora/objects/orbit:82314/datastreams/
        file_4732600/content

        Note also that the definition of the pseudo-inverse herein is different
        from numpys pinv() alone, as it includes pre and post projection onto
        the subspace defined by the projector Q.

    """
    if rhoss is None:
        rhoss = steadystate(L)

    sparse = kwargs.pop("sparse", False)
    if method == "direct":
        method = "splu" if sparse else "pinv"
    sparse_solvers = ["splu", "mkl_spsolve", "spilu"]
    dense_solvers = ["solve", "lstsq", "pinv"]
    if isinstance(L.data, _data.CSR) and method in dense_solvers:
        L = L.to("dense")
    elif isinstance(L.data, _data.Dense) and method in sparse_solvers:
        L = L.to("csr")

    N = np.prod(L.dims[0][0])
    dtype = type(L.data)
    rhoss_vec = operator_to_vector(rhoss)

    tr_op = qeye_like(rhoss)
    tr_op_vec = operator_to_vector(tr_op)

    P = _data.kron(rhoss_vec.data, tr_op_vec.data.transpose(), dtype=dtype)
    I = _data.identity_like(P)
    Q = _data.sub(I, P)

    if w in [None, 0.0]:
        L += 1e-15j
    else:
        L += 1.0j*w

    use_rcm = use_rcm and isinstance(L.data, _data.CSR)

    if use_rcm:
        perm = scipy.sparse.csgraph.reverse_cuthill_mckee(L.data.as_scipy())
        A = _data.permute.indices(L.data, perm, perm)
        Q = _data.permute.indices(Q, perm, perm, dtype=_data.CSR)
    else:
        A = L.data

    if method in ["pinv", "numpy", "scipy", "scipy2"]:
        # from scipy 1.7.0, they all use the same algorithm.
        LI = _data.Dense(scipy.linalg.pinv(A.to_array()), copy=False)
        LIQ = _data.matmul(LI, Q)
    elif method == "spilu":
        if not isinstance(A, _data.CSR):
            raise TypeError("'spilu' method can only be used with sparse data")
        ILU = scipy.sparse.linalg.spilu(A.as_scipy().tocsc(), **kwargs)
        LIQ = _data.Dense(ILU.solve(Q.to_array()))
    else:
        LIQ = _data.solve(A, Q, method, options=kwargs)

    R = _data.matmul(Q, LIQ)

    if use_rcm:
        rev_perm = np.argsort(perm)
        R = _data.permute.indices(R, rev_perm, rev_perm)

    return Qobj(R, dims=L.dims)


def _compute_precond(L, args):
    spilu_keys = {
        'permc_spec',
        'drop_tol',
        'diag_pivot_thresh',
        'fill_factor',
        'options',
    }
    ss_args = {
        key: args.pop(key)
        for key in spilu_keys
        if key in args
    }
    P = scipy.sparse.linalg.spilu(L.as_scipy().tocsc(), **ss_args)
    return scipy.sparse.linalg.LinearOperator(L.shape, matvec=P.solve)
