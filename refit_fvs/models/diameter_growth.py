import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from numpyro.contrib.control_flow import scan

from refit_fvs.models.distributions import (
    AffineBeta,
    NegativeHalfNormal,
    NegativeLogNormal,
)


def wykoff_model(
    data,
    num_cycles,
    bark_b1,
    bark_b2,
    num_variants,
    num_locations,
    num_plots,
    y=None,
    target="dg",
    pooling="unpooled",
    loc_random=False,
    plot_random=False,
):
    """A recursive model that predicts diameter growth or basal area increment
    for individual trees following the general form of Wykoff (1990), with
    annualization inspired by Cao (2000, 2004), and mixed effects approach
    illustrated by Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The model can be fit using an arbitrary number
    of time steps, and with three alternatives to incorporate hierarchical model
    structure across ecoregions: fully pooled, fully unpooled, and partially
    pooled. The likelihood for the model is calculated from the periodic
    outside-bark diameter growth using a Cauchy distribution to as a form of
    robust regression to help reduce the influence of extreme growth
    observations (both negative and positive) compared to a Normal likelihood.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
          1. variant_index
          2. location_index
          3. plot_index
          4. site_index
          5. slope, in percent, where 100% slope = 1.0
          6. elevation
          7. diameter at breast height
          8. crown_ratio_start, as a proportion, where 100% = 1.0
          9. crown_ratio_end, as a proportion, where 100% = 1.0
          10. competition_treelevel_start
          11. competition_treelevel_end
          12. competition_standlevel_start
          13. competition_standlevel_end
    num_cycles : int
      number of steps (or cycles) of growth to simulate
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    y : scalar or list-like
      observed values for target (basal area increment or diameter growth)
    target : str
      type of target variable, may be 'bai' or 'dg'
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'.
    """
    (
        variant,
        loc,
        plot,
        site_index,
        slope,
        elev,
        dbh,
        cr_start,
        cr_end,
        comp_tree_start,
        comp_tree_end,
        comp_stand_start,
        comp_stand_end,
    ) = data

    dbh = jnp.asarray(dbh).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1)
    loc = jnp.asarray(loc).reshape(-1)
    plot = jnp.asarray(plot).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)
    crown_ratio = jnp.linspace(cr_start, cr_end, num_cycles)
    comp_tree = jnp.linspace(comp_tree_start, comp_tree_end, num_cycles)
    comp_stand = jnp.linspace(comp_stand_start, comp_stand_end, num_cycles)

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            elev,
            cr_start,
            comp_tree_start,
            comp_stand_start,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(0.9, 2.0))
        b1z = numpyro.sample("b1z", dist.Normal(0.7, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(-0.1, 1.0))  # dbh**2
        b3z = numpyro.sample("b3z", dist.Normal(0.3, 1.0))  # ln(site_index)
        b4z = numpyro.sample("b4z", dist.Normal(-0.04, 1.0))  # slope
        b5z = numpyro.sample("b5z", dist.Normal(-0.1, 1.0))  # elev
        b6z = numpyro.sample("b6z", dist.Normal(0.4, 1.0))  # crown_ratio
        b7z = numpyro.sample(
            "b7z", dist.Normal(-0.4, 1.0)
        )  # comp_tree  # BAL / ln(dbh+1)
        b8z = numpyro.sample("b8z", dist.Normal(0.0, 1.0))  # comp_stand  # ln(BA)

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(0.9, 2.0))
            b1z = numpyro.sample("b1z", dist.Normal(0.7, 1.0))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(-0.1, 1.0))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(0.3, 1.0))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(-0.04, 1.0))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(-0.1, 1.0))  # elev
            b6z = numpyro.sample("b6z", dist.Normal(0.4, 1.0))  # crown_ratio
            b7z = numpyro.sample(
                "b7z", dist.Normal(-0.4, 1.0)
            )  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample("b8z", dist.Normal(0.0, 1.0))  # comp_stand  # ln(BA)

    elif pooling == "partial":
        b0z_mu = numpyro.sample("b0z_mu", dist.Normal(0.9, 0.5))
        b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0.7, 0.3))
        b2z_mu = numpyro.sample("b2z_mu", dist.Normal(-0.1, 0.1))
        b3z_mu = numpyro.sample("b3z_mu", dist.Normal(0.3, 0.1))
        b4z_mu = numpyro.sample("b4z_mu", dist.Normal(-0.04, 0.1))
        b5z_mu = numpyro.sample("b5z_mu", dist.Normal(-0.1, 0.1))
        b6z_mu = numpyro.sample("b6z_mu", dist.Normal(0.4, 0.1))
        b7z_mu = numpyro.sample("b7z_mu", dist.Normal(-0.4, 0.1))
        b8z_mu = numpyro.sample("b8z_mu", dist.Normal(0.0, 0.1))
        b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(0.05))
        b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(0.05))
        b3z_sd = numpyro.sample("b3z_sd", dist.HalfNormal(0.05))
        b4z_sd = numpyro.sample("b4z_sd", dist.HalfNormal(0.05))
        b5z_sd = numpyro.sample("b5z_sd", dist.HalfNormal(0.05))
        b6z_sd = numpyro.sample("b6z_sd", dist.HalfNormal(0.05))
        b7z_sd = numpyro.sample("b7z_sd", dist.HalfNormal(0.05))
        b8z_sd = numpyro.sample("b8z_sd", dist.HalfNormal(0.05))

        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(b3z_mu, b3z_sd))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(b4z_mu, b4z_sd))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(b5z_mu, b5z_sd))  # elev
            b6z = numpyro.sample("b6z", dist.Normal(b6z_mu, b6z_sd))  # crown_ratio
            b7z = numpyro.sample(
                "b7z", dist.Normal(b7z_mu, b7z_sd)
            )  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample(
                "b8z", dist.Normal(b8z_mu, b8z_sd)
            )  # comp_stand  # ln(BA)
    else:
        raise (
            ValueError(
                "valid options for pooling are 'unpooled', 'pooled', or 'partial'"
            )
        )

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    b5 = numpyro.deterministic("b5", b5z / X_sd[4])
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])

    adjust = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = numpyro.deterministic("eloc_last", -eloc_.sum())
        eloc = jnp.concatenate([eloc_, eloc_last.reshape(-1)])
    else:
        eloc = 0 * loc

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = numpyro.deterministic("eplot_last", -eplot_.sum())
        eplot = jnp.concatenate([eplot_, eplot_last.reshape(-1)])
    else:
        eplot = 0 * plot

    def step(dbh, step_covars):
        crown_ratio, comp_tree, comp_stand = step_covars
        if pooling == "pooled":
            size = b1 * jnp.log(dbh) + b2 * dbh**2
            site = b3 * jnp.log(site_index) + b4 * slope + b5 * elev
            comp = b6 * crown_ratio + b7 * comp_tree + b8 * comp_stand

            ln_dds = b0 + size + site + comp + eloc[loc] + eplot[plot]

        else:
            size = b1[variant] * jnp.log(dbh) + b2[variant] * dbh**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope
                + b5[variant] * elev
            )
            comp = (
                b6[variant] * crown_ratio
                + b7[variant] * comp_tree
                + b8[variant] * comp_stand
            )

            ln_dds = b0[variant] + size + site + comp + eloc[loc] + eplot[plot]

        dds = jnp.exp(ln_dds)
        dib_start = bark_b1 * dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end / bark_b1) ** (1 / bark_b2)
        dg_ob = dbh_end - dbh

        return dbh_end, dg_ob

    step_covars = (crown_ratio, comp_tree, comp_stand)
    dbh_end, growth = scan(step, dbh, step_covars, length=num_cycles)
    dg_pred = numpyro.deterministic("dg_pred", dbh_end - dbh)
    bai_pred = numpyro.deterministic("bai_pred", jnp.pi / 4 * (dbh_end**2 - dbh**2))

    if target.lower() == "bai":
        etree_bai = numpyro.sample("etree_bai", dist.Gamma(4.0, 0.1))
        obs = numpyro.sample("obs", dist.Cauchy(bai_pred, etree_bai), obs=y)
    elif target.lower() == "dg":
        etree_dg = numpyro.sample("etree_dg", dist.InverseGamma(2.0, 0.25))
        obs = numpyro.sample("obs", dist.Laplace(dg_pred, etree_dg), obs=y)

    if y is not None:
        if target.lower() == "bai":
            bai_obs = y
            dbh_end_obs = jnp.sqrt(4 / jnp.pi * bai_obs + dbh**2)
            dg_obs = dbh_end_obs - dbh

        elif target.lower() == "dg":
            dg_obs = y
            bai_obs = jnp.pi / 4 * ((dbh + dg_obs) ** 2 - dbh**2)

        bai_resid = numpyro.deterministic("bai_resid", bai_pred - bai_obs)
        bai_ss_res = ((bai_obs - bai_pred) ** 2).sum()
        bai_ss_tot = ((bai_obs - bai_obs.mean()) ** 2).sum()
        bai_r2 = numpyro.deterministic("bai_r2", 1 - bai_ss_res / bai_ss_tot)
        bai_bias = numpyro.deterministic("bai_bias", bai_resid.mean())
        bai_mae = numpyro.deterministic("bai_mae", jnp.abs(bai_resid).mean())
        bai_rmse = numpyro.deterministic(
            "bai_rmse", jnp.sqrt((bai_resid**2).sum() / num_trees)
        )
        dg_resid = numpyro.deterministic("dg_resid", dg_pred - dg_obs)
        dg_ss_res = ((dg_obs - dg_pred) ** 2).sum()
        dg_ss_tot = ((dg_obs - dg_obs.mean()) ** 2).sum()
        dg_r2 = numpyro.deterministic("dg_r2", 1 - dg_ss_res / dg_ss_tot)
        dg_bias = numpyro.deterministic("dg_bias", dg_resid.mean())
        dg_mae = numpyro.deterministic("dg_mae", jnp.abs(dg_resid).mean())
        dg_rmse = numpyro.deterministic(
            "dg_rmse", jnp.sqrt((dg_resid**2).sum() / num_trees)
        )


def potential_growth_model(
    data,
    num_cycles,
    bark_b1,
    bark_b2,
    num_variants,
    num_locations,
    num_plots,
    y=None,
    target="dg",
    pooling="pooled",
    loc_random=False,
    plot_random=False,
    quantile=0.95,
):
    """A recursive model that predicts potential diameter growth or
    basal area increment for individual trees adapted from the general
    form of Wykoff (1990), with annualization inspired by Cao (2000, 2004),
    and optional mixed effects following the approach illustrated by
    Weiskittel et al., (2007).

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
          1. variant_index
          2. location_index
          3. plot_index
          4. dbh
    num_cycles : int
      number of steps (or cycles) of growth to simulate
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    y : scalar or list-like
      observed outside-bark diameter growth or basal area increment
    target : str
      type of target variable, may be 'bai' or 'dg'
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'.
    """
    (variant, loc, plot, dbh) = data

    dbh = jnp.asarray(dbh).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1)
    loc = jnp.asarray(loc).reshape(-1)
    plot = jnp.asarray(plot).reshape(-1)

    X = jnp.array([jnp.log(dbh), dbh**2])
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(0.0, 1.0))
        b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(0.0, 1.0))
            b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2

    elif pooling == "partial":
        b0z_mu = numpyro.sample("b0z_mu", dist.Normal(0.0, 1.0))
        b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0.0, 1.0))
        b2z_mu = numpyro.sample("b2z_mu", dist.Normal(0.0, 1.0))
        b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(1.0))
        b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(1.0))

        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2

    else:
        raise (
            ValueError(
                "valid options for pooling are 'unpooled', 'pooled', or 'partial'"
            )
        )

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    adjust = b1 + b2
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = numpyro.deterministic("eloc_last", -eloc_.sum())
        eloc = jnp.concatenate([eloc_, eloc_last.reshape(-1)])
    else:
        eloc = 0 * loc

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = numpyro.deterministic("eplot_last", -eplot_.sum())
        eplot = jnp.concatenate([eplot_, eplot_last.reshape(-1)])
    else:
        eplot = 0 * plot

    def step(dbh, step_covars=None):
        if pooling == "pooled":
            ln_dds = b0 + b1 * jnp.log(dbh) + b2 * dbh**2 + eloc[loc] + eplot[plot]

        else:
            ln_dds = (
                b0
                + b1[variant] * jnp.log(dbh)
                + b2[variant] * dbh**2
                + eloc[loc]
                + eplot[plot]
            )

        dds = jnp.exp(ln_dds)
        dib_start = bark_b1 * dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end / bark_b1) ** (1 / bark_b2)
        dg_ob = dbh_end - dbh

        return dbh_end, dg_ob

    step_covars = None
    dbh_end, growth = scan(step, dbh, step_covars, length=num_cycles)

    dg_pred = numpyro.deterministic("dg_pred", dbh_end - dbh)
    bai_pred = numpyro.deterministic("bai_pred", jnp.pi / 4 * (dbh_end**2 - dbh**2))

    if target.lower() == "bai":
        etree_bai = numpyro.sample("etree_bai", dist.Gamma(4.0, 0.1))
        obs = numpyro.sample(
            "obs",
            dist.AsymmetricLaplaceQuantile(bai_pred, etree_bai, quantile=quantile),
            obs=y,
        )
    elif target.lower() == "dg":
        etree_dg = numpyro.sample("etree_dg", dist.InverseGamma(2.0, 0.25))
        obs = numpyro.sample(
            "obs",
            dist.AsymmetricLaplaceQuantile(dg_pred, etree_dg, quantile=quantile),
            obs=y,
        )


def potential_modified_model(
    data,
    num_cycles,
    bark_b1,
    bark_b2,
    num_variants,
    num_locations,
    num_plots,
    y=None,
    target="dg",
    pooling="pooled",
    loc_random=False,
    plot_random=False,
):
    """A recursive model that predicts diameter growth or basal area increment
    for individual trees adapted from the general form of Wykoff (1990), with
    annualization inspired by Cao (2000, 2004), and optional mixed effects
    following the approach illustrated by Weiskittel et al., (2007).

    Fixed effects are transformed such that increasing magnitude of the
    predictor variable should correspond to decreased growth to approximate
    a POTENTIAL*MODIFIER growth form.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
          1. variant_index
          2. location_index
          3. plot_index
          4. site_index
          5. slope, in percent, where 100% slope = 1.0
          6. elevation
          7. dbh
          8. crown_ratio_start, as a proportion, where 100% = 1.0
          9. crown_ratio_end, as a proportion, where 100% = 1.0
          10. competition_treelevel_start
          11. competition_treelevel_end
          12. competition_standlevel_start
          13. competition_standlevel_end
    num_cycles : int
      number of steps (or cycles) of growth to simulate
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    y : scalar or list-like
      observed outside-bark diameter growth
    target : str
      type of target variable, may be 'bai' or 'dg'
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'.
    """
    (
        variant,
        loc,
        plot,
        site_index,
        slope,
        elev,
        dbh,
        cr_start,
        cr_end,
        comp_tree_start,
        comp_tree_end,
        comp_stand_start,
        comp_stand_end,
    ) = data

    dbh = jnp.asarray(dbh).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1)
    loc = jnp.asarray(loc).reshape(-1)
    plot = jnp.asarray(plot).reshape(-1)
    site_index = 250 / jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)
    crown_ratio = jnp.linspace(1 - cr_start, 1 - cr_end, num_cycles)
    comp_tree = jnp.linspace(comp_tree_start, comp_tree_end, num_cycles)
    comp_stand = jnp.linspace(comp_stand_start, comp_stand_end, num_cycles)

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            elev,
            1 - cr_start,
            comp_tree_start,
            comp_stand_start,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(0, 1.0))
        b1z = numpyro.sample("b1z", dist.Normal(0, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(0, 1.0))  # dbh**2
        b3z = numpyro.sample("b3z", NegativeLogNormal(0, 2.0))  # ln(site_index)
        b4z = numpyro.sample("b4z", NegativeLogNormal(0, 2.0))  # slope
        b5z = numpyro.sample("b5z", NegativeLogNormal(0, 2.0))  # elev
        b6z = numpyro.sample("b6z", NegativeLogNormal(0, 2.0))  # crown_ratio
        b7z = numpyro.sample("b7z", NegativeLogNormal(0, 2.0))  # comp_tree
        # BAL / ln(dbh+1)
        b8z = numpyro.sample("b8z", NegativeLogNormal(0, 2.0))  # comp_stand
        # ln(BA)

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(0, 1.0))
            b1z = numpyro.sample("b1z", dist.Normal(0, 1.0))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(0, 1.0))  # dbh**2
            b3z = numpyro.sample("b3z", NegativeLogNormal(0, 2.0))  # ln(site_index)
            b4z = numpyro.sample("b4z", NegativeLogNormal(0, 2.0))  # slope
            b5z = numpyro.sample("b5z", NegativeLogNormal(0, 2.0))  # elev
            b6z = numpyro.sample("b6z", NegativeLogNormal(0, 2.0))  # crown_ratio
            b7z = numpyro.sample("b7z", NegativeLogNormal(0, 2.0))  # comp_tree
            # BAL / ln(dbh+1)
            b8z = numpyro.sample("b8z", NegativeLogNormal(0, 2.0))  # comp_stand
            # ln(BA)

    elif pooling == "partial":
        b0z_mu = numpyro.sample("b0z_mu", dist.Normal(0, 1.0))
        b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0, 1.0))
        b2z_mu = numpyro.sample("b2z_mu", dist.Normal(0, 1.0))
        b3z_mu = numpyro.sample("b3z_mu", dist.Normal(0, 1.0))
        b4z_mu = numpyro.sample("b4z_mu", dist.Normal(0, 1.0))
        b5z_mu = numpyro.sample("b5z_mu", dist.Normal(0, 1.0))
        b6z_mu = numpyro.sample("b6z_mu", dist.Normal(0, 1.0))
        b7z_mu = numpyro.sample("b7z_mu", dist.Normal(0, 1.0))
        b8z_mu = numpyro.sample("b8z_mu", dist.Normal(0, 1.0))

        b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(1.0))
        b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(1.0))
        b3z_sd = numpyro.sample("b3z_sd", dist.HalfNormal(1.0))
        b4z_sd = numpyro.sample("b4z_sd", dist.HalfNormal(1.0))
        b5z_sd = numpyro.sample("b5z_sd", dist.HalfNormal(1.0))
        b6z_sd = numpyro.sample("b6z_sd", dist.HalfNormal(1.0))
        b7z_sd = numpyro.sample("b7z_sd", dist.HalfNormal(1.0))
        b8z_sd = numpyro.sample("b8z_sd", dist.HalfNormal(1.0))

        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample(
                "b3z", NegativeLogNormal(b3z_mu, b3z_sd)
            )  # ln(site_index)
            b4z = numpyro.sample("b4z", NegativeLogNormal(b4z_mu, b4z_sd))  # slope
            b5z = numpyro.sample("b5z", NegativeLogNormal(b5z_mu, b5z_sd))  # elev
            b6z = numpyro.sample(
                "b6z", NegativeLogNormal(b6z_mu, b6z_sd)
            )  # crown_ratio
            b7z = numpyro.sample(
                "b7z", NegativeLogNormal(b7z_mu, b7z_sd)
            )  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample(
                "b8z", NegativeLogNormal(b8z_mu, b8z_sd)
            )  # comp_stand  # ln(BA)
    else:
        raise (
            ValueError(
                "valid options for pooling are 'unpooled', 'pooled', or 'partial'"
            )
        )

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    b5 = numpyro.deterministic("b5", b5z / X_sd[4])
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])

    adjust = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = numpyro.deterministic("eloc_last", -eloc_.sum())
        eloc = jnp.concatenate([eloc_, eloc_last.reshape(-1)])
    else:
        eloc = 0 * loc

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = numpyro.deterministic("eplot_last", -eplot_.sum())
        eplot = jnp.concatenate([eplot_, eplot_last.reshape(-1)])
    else:
        eplot = 0 * plot

    def step(dbh, step_covars):
        crown_ratio, comp_tree, comp_stand = step_covars
        if pooling == "pooled":
            size = b1 * jnp.log(dbh) + b2 * dbh**2
            site = b3 * jnp.log(site_index) + b4 * slope + b5 * elev
            comp = b6 * crown_ratio + b7 * comp_tree + b8 * comp_stand

            ln_dds = b0 + size + site + comp + eloc[loc] + eplot[plot]

        else:
            size = b1[variant] * jnp.log(dbh) + b2[variant] * dbh**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope
                + b5[variant] * elev
            )
            comp = (
                b6[variant] * crown_ratio
                + b7[variant] * comp_tree
                + b8[variant] * comp_stand
            )

            ln_dds = b0[variant] + size + site + comp + eloc[loc] + eplot[plot]

        dds = jnp.exp(ln_dds)
        dib_start = bark_b1 * dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end / bark_b1) ** (1 / bark_b2)
        dg_ob = dbh_end - dbh

        return dbh_end, dg_ob

    step_covars = (crown_ratio, comp_tree, comp_stand)
    dbh_end, growth = scan(step, dbh, step_covars, length=num_cycles)

    dg_pred = numpyro.deterministic("dg_pred", dbh_end - dbh)
    bai_pred = numpyro.deterministic("bai_pred", jnp.pi / 4 * (dbh_end**2 - dbh**2))

    if target.lower() == "bai":
        etree_bai = numpyro.sample("etree_bai", dist.Gamma(4.0, 0.1))
        obs = numpyro.sample("obs", dist.Cauchy(bai_pred, etree_bai), obs=y)
    elif target.lower() == "dg":
        etree_dg = numpyro.sample("etree_dg", dist.InverseGamma(2.0, 0.25))
        obs = numpyro.sample("obs", dist.Laplace(dg_pred, etree_dg), obs=y)

    if y is not None:
        if target.lower() == "bai":
            bai_obs = y
            dbh_end_obs = jnp.sqrt(4 / jnp.pi * bai_obs + dbh**2)
            dg_obs = dbh_end_obs - dbh

        elif target.lower() == "dg":
            dg_obs = y
            bai_obs = jnp.pi / 4 * ((dbh + dg_obs) ** 2 - dbh**2)

        bai_resid = numpyro.deterministic("bai_resid", bai_pred - bai_obs)
        bai_ss_res = ((bai_obs - bai_pred) ** 2).sum()
        bai_ss_tot = ((bai_obs - bai_obs.mean()) ** 2).sum()
        bai_r2 = numpyro.deterministic("bai_r2", 1 - bai_ss_res / bai_ss_tot)
        bai_bias = numpyro.deterministic("bai_bias", bai_resid.mean())
        bai_mae = numpyro.deterministic("bai_mae", jnp.abs(bai_resid).mean())
        bai_rmse = numpyro.deterministic(
            "bai_rmse", jnp.sqrt((bai_resid**2).sum() / num_trees)
        )
        dg_resid = numpyro.deterministic("dg_resid", dg_pred - dg_obs)
        dg_ss_res = ((dg_obs - dg_pred) ** 2).sum()
        dg_ss_tot = ((dg_obs - dg_obs.mean()) ** 2).sum()
        dg_r2 = numpyro.deterministic("dg_r2", 1 - dg_ss_res / dg_ss_tot)
        dg_bias = numpyro.deterministic("dg_bias", dg_resid.mean())
        dg_mae = numpyro.deterministic("dg_mae", jnp.abs(dg_resid).mean())
        dg_rmse = numpyro.deterministic(
            "dg_rmse", jnp.sqrt((dg_resid**2).sum() / num_trees)
        )


def vslite_model(
    data,
    num_cycles,
    bark_b1,
    bark_b2,
    num_variants,
    num_locations,
    num_plots,
    y=None,
    target="dg",
    pooling="unpooled",
    loc_random=False,
    plot_random=False,
):
    """A recursive model that predicts diameter growth or basal area increment
    for individual trees adapted from the general form of Wykoff (1990), with
    annualization inspired by Cao (2000, 2004), and mixed effects approach
    illustrated by Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The Wykoff model has been modified to follow
    a "POTENTIAL * MODIFIER" form where potential diameter growth is now
    estimated from the intercept and fixed effects of tree size. All other
    fixed effects have been transformed such that increasing magnitude of the
    predictor variable should correspond to decreased growth. The coefficients
    for these features are then constrained to be negative.

    The model can be fit using an arbitrary number of time steps, and with
    three alternatives to incorporate hierarchical model structure across
    ecoregions: fully pooled, fully unpooled, and partially pooled. The
    likelihood for the model is calculated from the periodic outside-bark
    diameter growth using a Cauchy distribution to as a form of robust
    regression to help reduce the influence of extreme growth observations
    (both negative and positive) compared to a Normal likelihood.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
          1. variant_index
          2. location_index
          3. plot_index
          4. site_index
          5. slope, in percent, where 100% slope = 1.0
          6. elevation
          7. dbh
          8. crown_ratio_start, as a proportion, where 100% = 1.0
          9. crown_ratio_end, as a proportion, where 100% = 1.0
          10. competition_treelevel_start
          11. competition_treelevel_end
          12. competition_standlevel_start
          13. competition_standlevel_end
          14. solar radiation (monthly)
          15. soil moisture (monthly)
          16. average temperature (monthly)
    num_cycles : int
      number of steps (or cycles) of growth to simulate
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    y : scalar or list-like
      observed outside-bark diameter growth
    target : str
      type of target variable, may be 'bai' or 'dg'
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'.
    """

    (
        variant,
        loc,
        plot,
        site_index,
        slope,
        elev,
        dbh,
        cr_start,
        cr_end,
        comp_tree_start,
        comp_tree_end,
        comp_stand_start,
        comp_stand_end,
        solar,
        moisture,
        temp,
    ) = data

    dbh = jnp.asarray(dbh).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1)
    loc = jnp.asarray(loc).reshape(-1)
    plot = jnp.asarray(plot).reshape(-1)
    site_index = 250 / jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)
    crown_ratio = jnp.linspace(1 - cr_start, 1 - cr_end, num_cycles)
    comp_tree = jnp.linspace(comp_tree_start, comp_tree_end, num_cycles)
    comp_stand = jnp.linspace(comp_stand_start, comp_stand_end, num_cycles)
    solar = jnp.moveaxis(solar, 0, -1)
    moisture = jnp.moveaxis(moisture, 0, -1)
    temp = jnp.moveaxis(temp, 0, -1)

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            elev,
            1 - cr_start,
            comp_tree_start,
            comp_stand_start,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(0.0, 1.0))
        b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2
        b3z = numpyro.sample("b3z", NegativeLogNormal(0.0, 2.0))  # ln(site_index)
        b4z = numpyro.sample("b4z", NegativeLogNormal(0.0, 2.0))  # slope
        # b5z = numpyro.sample('b5z', dist.Normal(0., 1.0))  # elev
        b6z = numpyro.sample("b6z", NegativeLogNormal(0.0, 2.0))  # crown_ratio
        b7z = numpyro.sample(
            "b7z", NegativeLogNormal(0.0, 2.0)
        )  # comp_tree  # BAL / ln(dbh+1)
        b8z = numpyro.sample("b8z", NegativeLogNormal(0.0, 2.0))  # comp_stand  # ln(BA)

        bclim = numpyro.sample("bclim", dist.LogNormal(0.0, 0.5))
        m1 = numpyro.sample("m1", AffineBeta(1.5, 2.8, 0.0, 0.1))
        m2 = numpyro.sample("m2", AffineBeta(1.5, 2.5, 0.1, 0.7))
        t1 = numpyro.sample("t1", AffineBeta(9.0, 5.0, 0.0, 9.0))
        t2 = numpyro.sample("t2", AffineBeta(3.5, 3.5, 10.0, 14.0))

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(0.0, 1.0))
            b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2
            b3z = numpyro.sample("b3z", NegativeLogNormal(0.0, 2.0))  # ln(site_index)
            b4z = numpyro.sample("b4z", NegativeLogNormal(0.0, 2.0))  # slope
            # b5z = numpyro.sample('b5z', NegativeHalfNormal(2))  # elev
            b6z = numpyro.sample("b6z", NegativeLogNormal(0.0, 2.0))  # crown_ratio
            b7z = numpyro.sample(
                "b7z", dist.Normal(0.0, 1.0)
            )  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample(
                "b8z", NegativeLogNormal(0.0, 2.0)
            )  # comp_stand  # ln(BA)

            bclim = numpyro.sample("bclim", dist.LogNormal(0.0, 0.5))
            m1 = numpyro.sample("m1", AffineBeta(1.5, 2.8, 0.0, 0.1))
            m2 = numpyro.sample("m2", AffineBeta(1.5, 2.5, 0.1, 0.7))
            t1 = numpyro.sample("t1", AffineBeta(9.0, 5.0, 0.0, 7.0))
            t2 = numpyro.sample("t2", AffineBeta(3.5, 3.5, 7.0, 17.0))

    elif pooling == "partial":
        b0z_mu = numpyro.sample("b0z_mu", dist.Normal(0, 1.0))
        b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0, 1.0))
        b2z_mu = numpyro.sample("b2z_mu", dist.Normal(0, 1.0))
        b3z_mu = numpyro.sample("b3z_mu", dist.Normal(0, 1.0))
        b4z_mu = numpyro.sample("b4z_mu", dist.Normal(0, 1.0))
        b5z_mu = numpyro.sample("b5z_mu", dist.Normal(0, 1.0))
        b6z_mu = numpyro.sample("b6z_mu", dist.Normal(0, 1.0))
        b7z_mu = numpyro.sample("b7z_mu", dist.Normal(0, 1.0))
        b8z_mu = numpyro.sample("b8z_mu", dist.Normal(0, 1.0))

        b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(1.0))
        b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(1.0))
        b3z_sd = numpyro.sample("b3z_sd", dist.HalfNormal(1.0))
        b4z_sd = numpyro.sample("b4z_sd", dist.HalfNormal(1.0))
        b5z_sd = numpyro.sample("b5z_sd", dist.HalfNormal(1.0))
        b6z_sd = numpyro.sample("b6z_sd", dist.HalfNormal(1.0))
        b7z_sd = numpyro.sample("b7z_sd", dist.HalfNormal(1.0))
        b8z_sd = numpyro.sample("b8z_sd", dist.HalfNormal(1.0))

        bclim_mu = numpyro.sample("bclim_mu", dist.Normal(0, 1.0))
        bclim_sd = numpyro.sample("bclim_sd", dist.HalfNormal(1.0))
        m1_conc1 = numpyro.sample("m1_conc1", dist.Gamma(100, 67))
        m1_conc2 = numpyro.sample("m1_conc2", dist.Gamma(350, 125))
        m2_conc1 = numpyro.sample("m2_conc1", dist.Gamma(100, 67))
        m2_conc2 = numpyro.sample("m2_conc2", dist.Gamma(275, 110))
        t1_conc1 = numpyro.sample("t1_conc1", dist.Gamma(81.0, 9.0))
        t1_conc2 = numpyro.sample("t1_conc2", dist.Gamma(25.0, 5.0))
        t2_conc1 = numpyro.sample("t2_conc1", dist.Gamma(12.25, 3.5))
        t2_conc2 = numpyro.sample("t2_conc2", dist.Gamma(12.25, 3.5))

        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample(
                "b3z", NegativeLogNormal(b3z_mu, b3z_sd)
            )  # ln(site_index)
            b4z = numpyro.sample("b4z", NegativeLogNormal(b4z_mu, b4z_sd))  # slope
            b5z = numpyro.sample("b5z", NegativeLogNormal(b5z_mu, b5z_sd))  # elev
            b6z = numpyro.sample(
                "b6z", NegativeLogNormal(b6z_mu, b6z_sd)
            )  # crown_ratio
            b7z = numpyro.sample(
                "b7z", NegativeLogNormal(b7z_mu, b7z_sd)
            )  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample(
                "b8z", NegativeLogNormal(b8z_mu, b8z_sd)
            )  # comp_stand  # ln(BA)

            bclim = numpyro.sample("bclim", dist.LogNormal(bclim_mu, bclim_sd))
            m1 = numpyro.sample("m1", AffineBeta(m1_conc1, m1_conc2, 0.0, 0.2))
            m2 = numpyro.sample("m2", AffineBeta(m2_conc1, m2_conc2, 0.2, 0.7))
            t1 = numpyro.sample("t1", AffineBeta(t1_conc1, t1_conc2, 0.0, 7.0))
            t2 = numpyro.sample("t2", AffineBeta(t2_conc1, t2_conc2, 7.0, 17.0))
    else:
        raise (
            ValueError(
                "valid options for pooling are 'unpooled', 'pooled', or 'partial'"
            )
        )

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    # b5 = numpyro.deterministic('b5', b5z/X_sd[4])
    b5 = 0.0
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])

    adjust = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = numpyro.deterministic("eloc_last", -eloc_.sum())
        eloc = jnp.concatenate([eloc_, eloc_last.reshape(-1)])
    else:
        eloc = 0 * loc

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = numpyro.deterministic("eplot_last", -eplot_.sum())
        eplot = jnp.concatenate([eplot_, eplot_last.reshape(-1)])
    else:
        eplot = 0 * plot

    def step(dbh, step_covars):
        crown_ratio, comp_tree, comp_stand, solar, moisture, temp = step_covars
        if pooling == "pooled":
            size = b1 * jnp.log(dbh) + b2 * dbh**2
            site = b3 * jnp.log(site_index) + b4 * slope  # + b5 * elev # drop elevation
            comp = b6 * crown_ratio + b7 * comp_tree + b8 * comp_stand

            fm = (moisture - m1) / (m2 - m1)
            fm = 1 / (1 + jnp.exp(-6 * (fm - 0.5)))
            ft = (temp - t1) / (t2 - t1)
            ft = 1 / (1 + jnp.exp(-6 * (ft - 0.5)))
            clim = jnp.log(
                bclim * (solar * jnp.minimum(ft, fm)).sum(axis=0) / solar.sum(axis=0)
            )

            ln_dds = b0 + size + site + comp + clim + eloc[loc] + eplot[plot]

        else:
            size = b1[variant] * jnp.log(dbh) + b2[variant] * dbh**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope  # +
                # b5[variant] * elev  # drop elevation
            )
            comp = (
                b6[variant] * crown_ratio
                + b7[variant] * comp_tree
                + b8[variant] * comp_stand
            )

            fm = (moisture - m1[variant]) / (m2[variant] - m1[variant])
            fm = 1 / (1 + jnp.exp(-6 * (fm - 0.5)))
            ft = (temp - t1[variant]) / (t2[variant] - t1[variant])
            ft = 1 / (1 + jnp.exp(-6 * (ft - 0.5)))
            clim = jnp.log(
                bclim[variant]
                * (solar * jnp.minimum(ft, fm)).sum(axis=0)
                / solar.sum(axis=0)
            )

            ln_dds = b0[variant] + size + site + comp + clim + eloc[loc] + eplot[plot]

        dds = jnp.exp(ln_dds)
        dib_start = bark_b1 * dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end / bark_b1) ** (1 / bark_b2)
        dg_ob = dbh_end - dbh

        return dbh_end, dg_ob

    step_covars = (crown_ratio, comp_tree, comp_stand, solar, moisture, temp)
    dbh_end, growth = scan(step, dbh, step_covars, length=num_cycles)

    dg_pred = numpyro.deterministic("dg_pred", dbh_end - dbh)
    bai_pred = numpyro.deterministic("bai_pred", jnp.pi / 4 * (dbh_end**2 - dbh**2))

    if target.lower() == "bai":
        etree_bai = numpyro.sample("etree_bai", dist.Gamma(4.0, 0.1))
        obs = numpyro.sample("obs", dist.Cauchy(bai_pred, etree_bai), obs=y)
    elif target.lower() == "dg":
        etree_dg = numpyro.sample("etree_dg", dist.InverseGamma(2.0, 0.25))
        obs = numpyro.sample("obs", dist.Laplace(dg_pred, etree_dg), obs=y)

    if y is not None:
        if target.lower() == "bai":
            bai_obs = y
            dbh_end_obs = jnp.sqrt(4 / jnp.pi * bai_obs + dbh**2)
            dg_obs = dbh_end_obs - dbh

        elif target.lower() == "dg":
            dg_obs = y
            bai_obs = jnp.pi / 4 * ((dbh + dg_obs) ** 2 - dbh**2)

        bai_resid = numpyro.deterministic("bai_resid", bai_pred - bai_obs)
        bai_ss_res = ((bai_obs - bai_pred) ** 2).sum()
        bai_ss_tot = ((bai_obs - bai_obs.mean()) ** 2).sum()
        bai_r2 = numpyro.deterministic("bai_r2", 1 - bai_ss_res / bai_ss_tot)
        bai_bias = numpyro.deterministic("bai_bias", bai_resid.mean())
        bai_mae = numpyro.deterministic("bai_mae", jnp.abs(bai_resid).mean())
        bai_rmse = numpyro.deterministic(
            "bai_rmse", jnp.sqrt((bai_resid**2).sum() / num_trees)
        )
        dg_resid = numpyro.deterministic("dg_resid", dg_pred - dg_obs)
        dg_ss_res = ((dg_obs - dg_pred) ** 2).sum()
        dg_ss_tot = ((dg_obs - dg_obs.mean()) ** 2).sum()
        dg_r2 = numpyro.deterministic("dg_r2", 1 - dg_ss_res / dg_ss_tot)
        dg_bias = numpyro.deterministic("dg_bias", dg_resid.mean())
        dg_mae = numpyro.deterministic("dg_mae", jnp.abs(dg_resid).mean())
        dg_rmse = numpyro.deterministic(
            "dg_rmse", jnp.sqrt((dg_resid**2).sum() / num_trees)
        )


def threepg_model(
    data,
    num_cycles,
    bark_b1,
    bark_b2,
    num_variants,
    num_locations,
    num_plots,
    y=None,
    target="dg",
    pooling="unpooled",
):
    """A recursive model that predicts diameter growth or basal area increment
    for individual trees adapted from the general form of Wykoff (1990), with
    annualization inspired by Cao (2000, 2004), and mixed effects approach
    illustrated by Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The Wykoff model has been modified to follow
    a "POTENTIAL * MODIFIER" form where potential diameter growth is now
    estimated from the intercept and fixed effects of tree size. All other
    fixed effects have been transformed such that increasing magnitude of the
    predictor variable should correspond to decreased growth. The coefficients
    for these features are then constrained to be negative.

    The model can be fit using an arbitrary number of time steps, and with
    three alternatives to incorporate hierarchical model structure across
    ecoregions: fully pooled, fully unpooled, and partially pooled. The
    likelihood for the model is calculated from the periodic outside-bark
    diameter growth using a Cauchy distribution to as a form of robust
    regression to help reduce the influence of extreme growth observations
    (both negative and positive) compared to a Normal likelihood.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
          1. variant_index
          2. location_index
          3. plot_index
          4. site_index
          5. slope, in percent, where 100% slope = 1.0
          6. elevation
          7. dbh
          8. crown_ratio_start, as a proportion, where 100% = 1.0
          9. crown_ratio_end, as a proportion, where 100% = 1.0
          10. competition_treelevel_start
          11. competition_treelevel_end
          12. competition_standlevel_start
          13. competition_standlevel_end
          14. solar radiation (monthly)
          15. soil moisture (monthly)
          16. average temperature (monthly)
          17. vapor pressure deficit (monthly)
          18. number of frost free days (monthly)
    num_cycles : int
      number of steps (or cycles) of growth to simulate
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    y : scalar or list-like
      observed outside-bark diameter growth
    target : str
      type of target variable, may be 'bai' or 'dg'
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'.
    """

    (
        variant,
        loc,
        plot,
        site_index,
        slope,
        elev,
        dbh,
        cr_start,
        cr_end,
        comp_tree_start,
        comp_tree_end,
        comp_stand_start,
        comp_stand_end,
        solar,
        moisture,
        temp,
        vpd,
        nffd,
    ) = data

    dbh = jnp.asarray(dbh).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1)
    loc = jnp.asarray(loc).reshape(-1)
    plot = jnp.asarray(plot).reshape(-1)
    site_index = 250 / jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)
    crown_ratio = jnp.linspace(1 - cr_start, 1 - cr_end, num_cycles)
    comp_tree = jnp.linspace(comp_tree_start, comp_tree_end, num_cycles)
    comp_stand = jnp.linspace(comp_stand_start, comp_stand_end, num_cycles)
    solar = jnp.moveaxis(solar, 0, -1)
    moisture = jnp.moveaxis(moisture, 0, -1)
    temp = jnp.moveaxis(temp, 0, -1)
    vpd = jnp.moveaxis(vpd, 0, -1)
    nffd = jnp.moveaxis(nffd, 0, -1)

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            elev,
            1 - cr_start,
            comp_tree_start,
            comp_stand_start,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)
    bsolar = numpyro.sample("bsolar", dist.Normal(0.0, 1.0))

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(0.7, 2.0))
        b1z = numpyro.sample("b1z", dist.Normal(0.8, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(-0.1, 1.0))  # dbh**2
        b3z = numpyro.sample("b3z", NegativeHalfNormal(2))  # ln(site_index)
        b4z = numpyro.sample("b4z", NegativeHalfNormal(2))  # slope
        # b5z = numpyro.sample('b5z', NegativeHalfNormal(2))  # elev
        b6z = numpyro.sample("b6z", NegativeHalfNormal(2))  # crown_ratio
        b7z = numpyro.sample(
            "b7z", NegativeHalfNormal(2)
        )  # comp_tree  # BAL / ln(dbh+1)
        b8z = numpyro.sample("b8z", NegativeHalfNormal(2))  # comp_stand  # ln(BA)

        m1 = numpyro.sample("m1", AffineBeta(1.5, 2.8, 0.0, 0.2))
        m2 = numpyro.sample("m2", AffineBeta(1.5, 2.5, 0.2, 0.7))
        tmin = numpyro.sample("tmin", AffineBeta(9.0, 5.0, 0.0, 9.0))
        topt = numpyro.sample("tmax", AffineBeta(3.5, 3.5, 10.0, 14.0))
        tmax = numpyro.sample("topt", AffineBeta(3.5, 3.5, 10.0, 14.0))
        vpd_coef = numpyro.sample("vpd_coef", AffineBeta(3.5, 3.5, 10.0, 14.0))
        sw_const = numpyro.sample("sw_const", AffineBeta(3.5, 3.5, 10.0, 14.0))
        sw_pow = numpyro.sample("sw_pow", AffineBeta(3.5, 3.5, 10.0, 14.0))

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            pass
            # TO DO

    elif pooling == "partial":
        pass
        # TO DO

        with numpyro.plate("variants", num_variants):
            pass
            # TO DO
    else:
        raise (
            ValueError(
                "valid options for pooling are 'unpooled', 'pooled', or 'partial'"
            )
        )

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    # b5 = numpyro.deterministic('b5', b5z/X_sd[4])
    b5 = 0.0
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])

    adjust = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8
    b0 = numpyro.deterministic("b0", b0z - adjust)

    with numpyro.plate("locations", num_locations):
        eloc = numpyro.sample("eloc", dist.Normal(0, 1.0))  # random effect of location

    if y is not None:
        with numpyro.plate("plots", num_plots):
            eplot = numpyro.sample(
                "eplot", dist.Normal(0, 1.0)
            )  # random effect of plot
    else:
        eplot = 0 * plot

    def step(dbh, step_covars):
        crown_ratio, comp_tree, comp_stand, solar, moisture, temp = step_covars
        if pooling == "pooled":
            size = b1 * jnp.log(dbh) + b2 * dbh**2
            site = b3 * jnp.log(site_index) + b4 * slope  # + b5 * elev # drop elevation
            comp = b6 * crown_ratio + b7 * comp_tree + b8 * comp_stand

            # ft = calc_modifier_temp(temp, tmin, tmax, topt)
            # fd = calc_modifier_vpd(vpd, vpd_coef)
            # fs = calc_modifier_soilwater(moisture, sw_const, sw_pow)
            clim = 0

            ln_dds = b0 + size + site + comp + clim + eloc[loc] + eplot[plot]

        else:
            size = b1[variant] * jnp.log(dbh) + b2[variant] * dbh**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope  # +
                # b5[variant] * elev  # drop elevation
            )
            comp = (
                b6[variant] * crown_ratio
                + b7[variant] * comp_tree
                + b8[variant] * comp_stand
            )

            # ft = calc_modifier_temp(temp, tmin, tmax, topt)
            # fd = calc_modifier_vpd(vpd, vpd_coef)
            # fs = calc_modifier_soilwater(moisture, sw_const, sw_pow)
            clim = 0

            ln_dds = b0[variant] + size + site + comp + clim + eloc[loc] + eplot[plot]

        dds = jnp.exp(ln_dds)
        dib_start = bark_b1 * dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end / bark_b1) ** (1 / bark_b2)
        dg_ob = dbh_end - dbh

        return dbh_end, dg_ob

    step_covars = (crown_ratio, comp_tree, comp_stand, solar, moisture, temp)
    dbh_end, growth = scan(step, dbh, step_covars, length=num_cycles)

    dg_pred = numpyro.deterministic("dg_pred", dbh_end - dbh)
    bai_pred = numpyro.deterministic("bai_pred", jnp.pi / 4 * (dbh_end**2 - dbh**2))

    if target.lower() == "bai":
        etree_bai = numpyro.sample("etree_bai", dist.Gamma(4.0, 0.1))
        obs = numpyro.sample("obs", dist.Cauchy(bai_pred, etree_bai), obs=y)
    elif target.lower() == "dg":
        etree_dg = numpyro.sample("etree_dg", dist.InverseGamma(2.0, 0.25))
        obs = numpyro.sample("obs", dist.Laplace(dg_pred, etree_dg), obs=y)

    if y is not None:
        if target.lower() == "bai":
            bai_obs = y
            dbh_end_obs = jnp.sqrt(4 / jnp.pi * bai_obs + dbh**2)
            dg_obs = dbh_end_obs - dbh

        elif target.lower() == "dg":
            dg_obs = y
            bai_obs = jnp.pi / 4 * ((dbh + dg_obs) ** 2 - dbh**2)

        bai_resid = numpyro.deterministic("bai_resid", bai_pred - bai_obs)
        bai_ss_res = ((bai_obs - bai_pred) ** 2).sum()
        bai_ss_tot = ((bai_obs - bai_obs.mean()) ** 2).sum()
        bai_r2 = numpyro.deterministic("bai_r2", 1 - bai_ss_res / bai_ss_tot)
        bai_bias = numpyro.deterministic("bai_bias", bai_resid.mean())
        bai_mae = numpyro.deterministic("bai_mae", jnp.abs(bai_resid).mean())
        bai_rmse = numpyro.deterministic(
            "bai_rmse", jnp.sqrt((bai_resid**2).sum() / num_trees)
        )
        dg_resid = numpyro.deterministic("dg_resid", dg_pred - dg_obs)
        dg_ss_res = ((dg_obs - dg_pred) ** 2).sum()
        dg_ss_tot = ((dg_obs - dg_obs.mean()) ** 2).sum()
        dg_r2 = numpyro.deterministic("dg_r2", 1 - dg_ss_res / dg_ss_tot)
        dg_bias = numpyro.deterministic("dg_bias", dg_resid.mean())
        dg_mae = numpyro.deterministic("dg_mae", jnp.abs(dg_resid).mean())
        dg_rmse = numpyro.deterministic(
            "dg_rmse", jnp.sqrt((dg_resid**2).sum() / num_trees)
        )


def calc_modifier_temp(t, tmin, tmax, topt):
    res = ((t - tmin) / (topt - tmin)) * ((tmax - t) / (tmax - topt)) ** (
        (tmax - topt) / (topt - tmin)
    )
    return jnp.clip(res, 0, 1)


def calc_modifier_vpd(vpd, coef):
    return jnp.exp(-1 * coef * vpd)


def calc_modifier_soilwater(moisture, const=0.7, power=9):
    return 1 / (1 + ((1 - moisture) / const) ** power)


def calc_modifier_frost(nffd, k):
    return 1 - k * (nffd / 30.0)
