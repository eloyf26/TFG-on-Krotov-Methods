from ..settings import settings

__all__ = ["CoreOptions"]


class QutipOptions:
    """
    Class for basic functionality for qutip's options.

    Define basic method to wrap an ``options`` dict.
    Default options are in a class _options dict.
    """
    _options = {}
    _settings_name = None  # Where the default is in settings

    def __init__(self, **options):
        self.options = self._options.copy()
        for key in set(options) & set(self.options):
            self[key] = options.pop(key)
        if options:
            raise KeyError(f"Options {set(options)} are not supported.")

    def __contains__(self, key):
        return key in self.options

    def __getitem__(self, key):
        # Let the dict catch the KeyError
        return self.options[key]

    def __setitem__(self, key, value):
        # Let the dict catch the KeyError
        self.options[key] = value

    def __repr__(self, full=True):
        out = [f"<{self.__class__.__name__}("]
        for key, value in self.options.items():
            if full or value != self._options[key]:
                out += [f"    '{key}': {repr(value)},"]
        out += [")>"]
        if len(out)-2:
            return "\n".join(out)
        else:
            return "".join(out)

    def __enter__(self):
        self._backup = getattr(settings, self._settings_name)
        setattr(settings, self._settings_name, self)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        setattr(settings, self._settings_name, self._backup)


class CoreOptions(QutipOptions):
    """
    Options used by the core of qutip such as the tolerance of :class:`Qobj`
    comparison or coefficient's format.

    Values can be changed in ``qutip.settings.core`` or by using context:
    ``with CoreOptions(atol=1e-6): ...``.

    Options
    -------
    auto_tidyup : bool
        Whether to tidyup during sparse operations.

    auto_tidyup_dims : bool [True]
        Use auto tidyup dims on multiplication. (Not used yet)

    atol : float {1e-12}
        General absolute tolerance

    rtol : float {1e-12}
        General relative tolerance
        Used to choose QobjEvo.expect output type

    auto_tidyup_atol : float {1e-14}
        The absolute tolerance used in automatic tidyup (see the ``auto_tidyup``
        parameter above) and the default value of ``atol`` used in
        :method:`Qobj.tidyup`.

    function_coefficient_style : str {"auto"}
        The signature expected by function coefficients. The options are:

        - "pythonic": the signature should be ``f(t, ...)`` where ``t``
          is the time and the ``...`` are the remaining arguments passed
          directly into the function. E.g. ``f(t, w, b=5)``.

        - "dict": the signature shoule be ``f(t, args)`` where ``t`` is
          the time and ``args`` is a dict containing the remaining arguments.
          E.g. ``f(t, {"w": w, "b": 5})``.

        - "auto": select automatically between the two options above based
          on the signature of the supplied function. If the function signature
          is exactly ``f(t, args)`` then ``dict`` is used. Otherwise
          ``pythonic`` is used.

    default_dtype : Nonetype, str, type {None}
        When set, functions creating :class:`Qobj`, such as :func:"qeye" or
        :func:"rand_herm", will use the specified data type. Any data-layer
        known to ``qutip.data.to`` is accepted. When ``None``, these functions
        will default to a sensible data type.
    """
    _options = {
        # use auto tidyup
        "auto_tidyup": True,
        # use auto tidyup dims on multiplication
        "auto_tidyup_dims": True,
        # general absolute tolerance
        "atol": 1e-12,
        # general relative tolerance
        "rtol": 1e-12,
        # use auto tidyup absolute tolerance
        "auto_tidyup_atol": 1e-14,
        # signature style expected by function coefficients
        "function_coefficient_style": "auto",
        # Default Qobj dtype for Qobj create function
        "default_dtype": None,
    }
    _settings_name = "core"


# Creating the instance of core options to use everywhere.
settings.core = CoreOptions()
