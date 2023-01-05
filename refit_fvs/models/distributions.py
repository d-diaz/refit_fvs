from jax import numpy as jnp
from jax import random, lax
from numpyro.distributions.util import (is_prng_key, promote_shapes,
                                        validate_sample)

from numpyro.distributions import constraints, Normal, Beta
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.transforms import AffineTransform

from jax.scipy.special import gammaln, gammainc


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


class NegativeGamma(Distribution):
    arg_constraints = {
        "concentration": constraints.less_than(0),
        "rate": constraints.positive,
    }
    support = constraints.less_than(0)
    reparametrized_params = ["concentration", "rate"]

    def __init__(self, concentration, rate=1.0, validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        batch_shape = lax.broadcast_shapes(jnp.shape(concentration),
                                           jnp.shape(rate))
        super(NegativeGamma, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape
        return -random.gamma(key, -self.concentration, shape=shape) / self.rate

    @validate_sample
    def log_prob(self, value):
        concentration = -self.concentration
        value = -value
        normalize_term = gammaln(concentration) - concentration * jnp.log(
            self.rate
        )
        return (
            (concentration-1) * jnp.log(value)
            - self.rate * value
            - normalize_term
        )

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return -self.concentration / jnp.power(self.rate, 2)

    def cdf(self, x):
        return 1 - gammainc(-self.concentration, self.rate * -x)

    
class AffineBeta(TransformedDistribution):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    reparametrized_params = ["concentration1", "concentration0"]
    
    def __init__(self, concentration1, concentration0, loc, scale, validate_args=None):
        self.concentration1, self.concentration0, self.loc, self.scale = promote_shapes(
            concentration1, concentration0, loc, scale
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1), jnp.shape(concentration0), jnp.shape(loc), jnp.shape(scale),
        )
        base_dist = Beta(concentration1, concentration0)
        super(AffineBeta, self).__init__(
            base_dist,
            AffineTransform(loc=loc, scale=scale, domain=constraints.unit_interval),
            validate_args=validate_args,
        )
    
    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)
    
    @property
    def low(self):
        return self.loc

    @property
    def high(self):
        return self.loc + self.scale

    @property
    def mean(self):
        return self.loc + self.scale * self.base_dist.mean

    @property
    def variance(self):
        return self.scale**2 * self.base_dist.variance