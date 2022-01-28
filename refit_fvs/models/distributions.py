from jax import numpy as jnp
from jax import random, lax
from numpyro.distributions.util import is_prng_key, promote_shapes
from numpyro.distributions import constraints, Normal
from numpyro.distributions.distribution import Distribution


class AsymmetricLaplace(Distribution):
    """Asymmetric version of the Laplace Distribution, intended for quantile
    regression."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive,
                       "quantile": constraints.interval(0.,1.)}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, quantile=0.5, validate_args=None):
        self.loc, self.scale, self.quantile = promote_shapes(loc, scale,
                                                             quantile)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale),
                                           jnp.shape(quantile))
        super(AsymmetricLaplace, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def log_prob(self, y):
        # following Yu and Moyeed (2001)
        if self._validate_args:
            self._validate_sample(value)

        µ, σ, p = self.loc, self.scale, self.quantile
        const = p*(1-p)/σ
        z = (y - µ)/σ
        check = z * jnp.where(y <= µ, -(1-p), p)
        return jnp.log(const*jnp.exp(-check))

    def sample(self, key, sample_shape=()):
        # mixture of exponentials per Kozumi and Kobyashi (2009)
        assert is_prng_key(key)
        µ, σ, p = self.loc, self.scale, self.quantile
        shape = (2,) + sample_shape + self.batch_shape + self.event_shape
        u, v = random.exponential(key, shape=shape) * σ
        e = u/p - v/(1-p)
        return µ + e

    @property
    def mean(self):
        µ, σ, p = self.loc, self.scale, self.quantile
        return µ + (1-2*p)/(p*(1-p))*σ

    @property
    def variance(self):
        σ, p = self.scale, self.quantile
        return (1-2*p+2*p**2)/(p**2*(1-p)**2) * σ**2


class NegativeHalfNormal(Distribution):
    """Half-Normal distribution located at zero and constrained to negative
    values."""

    reparametrized_params = ["scale"]
    support = constraints.less_than(0)
    arg_constraints = {"scale": constraints.positive}

    def __init__(self, scale=1.0, validate_args=None):
        self._normal = Normal(0.0, scale)
        self.scale = scale
        super(NegativeHalfNormal, self).__init__(
            batch_shape=jnp.shape(scale), validate_args=validate_args
        )


    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return -jnp.abs(self._normal.sample(key, sample_shape))


    @validate_sample
    def log_prob(self, value):
        return self._normal.log_prob(value) + jnp.log(2)

    def cdf(self, value):
        return self._normal.cdf(value) * 2


    def icdf(self, q):
        return -self._normal.icdf((q + 1) / 2)


    @property
    def mean(self):
        return -jnp.sqrt(2 / jnp.pi) * self.scale

    @property
    def variance(self):
        return (1 - 2 / jnp.pi) * self.scale ** 2
