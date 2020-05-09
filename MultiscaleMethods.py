from Tools.Utils import act_on_functions
from Tools.Registry import Registry


MULTISCALE_METHODS = Registry()


class AbstractMultiscaleApproximationMethod(object):
    """
    TODO: instead of original_funcitom grid_size, and resolution.,
    The approximation method should reeieve the Sampler.
    """
    def __init__(self,
                 manifold,
                 original_function,
                 grid_size,
                 resolution,
                 rbf,
                 number_of_scales,
                 scaled_interpolation_method):
        self._manifold = manifold
        self._original_function = original_function
        self._grid_size = grid_size
        self._resolution = resolution
        self._rbf = rbf
        self._number_of_scales = number_of_scales
        self._scaled_interpolation_method = scaled_interpolation_method

        self._init_step()

    def _init_step(self):
        self._f_j = manifold.zero_func
        self._e_j = act_on_functions(self._manifold.log, self._f_j, self._original_function)

    @property
    @abc.abstract_method
    def is_approximating_on_tangent(self):
        pass

    @property
    @abc.abstract_method
    def _function_to_interpolate(self):
        pass

    @property
    @abc.abstract_method
    def _function_added_to_f_j(self):
        pass

    def run_all_scales(self):
        for scale_index in range(1, self._number_of_scales + 1):
            scale = scaling_factor ** scale_index
            print("NEW SCALE: {}".format(scale))
            current_grid_parameters = [
                ('Grid', symmetric_grid_params(grid_size + 1, scale / resolution)),
                # Can add here more grids (borders)
            ]

            self._s_j = scaled_interpolation_method(
                self._manifold,
                self._get_function_to_interpolate,
                current_grid_parameters,
                self._rbf,
                scale,
                # TODO: remove this
                self.is_approximating_on_tangent).approximation
            print("interpolated!")

            self._f_j = act_on_functions(self._manifold.exp, self._f_j, self._function_added_to_f_j)
            self._e_j = act_on_functions(self._manifold.log, self._f_j, self._original_function)
            yield self._f_j


@MULTISCALE_METHODS.register('tangent')
class Tangent(AbstractMultiscaleApproximationMethod):
    @property
    def is_approximating_on_tangent(self):
        return True

    @property
    def _function_to_interpolate(self):
        return self._e_j

    @property
    def _function_added_to_f_j(self):
        return self._s_j


@MULTISCALE_METHODS.register('adaptive')
class Adaptive(AbstractMultiscaleApproximationMethod):
    @property
    def is_approximating_on_tangent(self):
        return False

    @property
    def _function_to_interpolate(self):
        return (e_j, act_on_functions(self._manifold.exp, self._manifold.zero_func, self._e_j))

    @property
    def _function_added_to_f_j(self):
        return self._s_j


@MULTISCALE_METHODS.register('intrinsic')
class Intrinsic(AbstractMultiscaleApproximationMethod):
    @property
    def is_approximating_on_tangent(self):
        return False

    @property
    def _function_to_interpolate(self):
        return act_on_functions(self._manifold.exp, self._manifold.zero_func, self._e_j)

    @property
    def _function_added_to_f_j(self):
        return act_on_functions(self._manifold.log, self._manifold.zero_func, self._s_j)
