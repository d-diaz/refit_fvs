import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from jax.random import PRNGKey
from numpyro.contrib.control_flow import scan


def wykoff_simulator(
    bark_b0,
    bark_b1,
    bark_b2,
    beta_mean,
    beta_sd,
    etree_mean,
    etree_sd,
    data,
    dbh_err_func,
    rad_err_func,
    tree_comp,
    stand_comp,
    loc_sd=None,
    plot_sd=None,
):
    assert tree_comp in ["bal", "ballndbh", "relht"]
    assert stand_comp in ["ba", "lnba", "ccf"]
    (
        _,  # var_idx,
        loc_idx,
        plot_idx,
        site_index,
        slope,
        asp,
        elev,
        dbh,
        crown_ratio,
        bal,
        relht,
        bapa,
        ccf,
    ) = data

    # num_trees = dbh.size
    # variant = jnp.asarray(var_idx).reshape(-1)
    location = jnp.asarray(loc_idx).reshape(-1)
    plot = jnp.asarray(plot_idx).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    asp = jnp.asarray(asp).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)

    dbh = jnp.asarray(dbh).reshape(-1)
    crown_ratio = jnp.asarray(crown_ratio)
    bal = jnp.asarray(bal)
    relht = jnp.asarray(relht)
    bapa = jnp.asarray(bapa)
    ccf = jnp.asarray(ccf)

    if tree_comp == "bal":
        X_tree = bal[:, 0]
    elif tree_comp == "relht":
        X_tree = relht[:, 0]
    else:  # tree_comp == 'ballndbh':
        X_tree = bal[:, 0] / jnp.log(dbh + 1.0)

    if stand_comp == "ba":
        X_stand = bapa[:, 0]
    elif stand_comp == "lnba":
        X_stand = jnp.log(bapa[:, 0])
    else:  # stand_comp == 'ccf':
        X_stand = ccf[:, 0]

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            slope**2,
            slope * jnp.sin(asp),
            slope * jnp.cos(asp),
            elev,
            elev**2,
            crown_ratio[:, 0],
            crown_ratio[:, 0] ** 2,
            X_tree,
            X_stand,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    bz = dist.Normal(beta_mean, beta_sd).sample(PRNGKey(0))
    b0z, b1z, b2z, b3z, b4z, b5z, b6z, b7z, b8z, b9z, b10z, b11z, b12z, b13z = bz
    etree_conc, etree_rate = etree_mean**2 / etree_sd**2, etree_mean / etree_sd**2
    etree = dist.Gamma(etree_conc, etree_rate).sample(PRNGKey(0))  # periodic (5-yr err)

    b_ = bz[1:] / X_sd
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = b_

    adjust = (
        b1 * X_mu[0]
        + b2 * X_mu[1]
        + b3 * X_mu[2]
        + b4 * X_mu[3]
        + b5 * X_mu[4]
        + b6 * X_mu[5]
        + b7 * X_mu[6]
        + b8 * X_mu[7]
        + b9 * X_mu[8]
        + b10 * X_mu[9]
        + b11 * X_mu[10]
        + b12 * X_mu[11]
        + b13 * X_mu[12]
    )
    b0 = b0z - adjust

    num_locations = location.max() + 1
    if loc_sd is not None:
        eloc = dist.Normal(0, loc_sd).sample(PRNGKey(0), sample_shape=(num_locations,))
    else:
        eloc = jnp.zeros(num_locations)
    num_plots = plot.max() + 1
    if plot_sd is not None:
        eplot = dist.Normal(0, plot_sd).sample(PRNGKey(0), sample_shape=(num_plots,))
    else:
        eplot = jnp.zeros(num_plots)

    def step(dbh0, t):
        cr_ = crown_ratio[:, t]
        if tree_comp == "bal":
            tree_var = bal[:, t]
        elif tree_comp == "ballndbh":
            tree_var = bal[:, t] / jnp.log(dbh0 + 1.0)
        else:  # tree_comp == 'relht':
            tree_var = relht[:, t]

        if stand_comp == "ba":
            stand_var = bapa[:, t]
        elif stand_comp == "lnba":
            stand_var = jnp.log(bapa[:, t])
        else:  # stand_comp == 'ccf':
            stand_var = ccf[:, t]

        size = b1 * jnp.log(dbh0) + b2 * dbh0**2
        site = (
            b3 * jnp.log(site_index)
            + b4 * slope
            + b5 * slope**2
            + b6 * slope * jnp.sin(asp)
            + b7 * slope * jnp.cos(asp)
            + b8 * elev
            + b9 * elev**2
        )
        comp = b10 * cr_ + b11 * cr_**2 + b12 * tree_var + b13 * stand_var

        ln_dds_ = b0 + size + site + comp + eloc[location] + eplot[plot]
        ln_dds = dist.Normal(ln_dds_, etree).sample(PRNGKey(0))

        dds = jnp.exp(ln_dds)
        dib0 = bark_b0 + bark_b1 * dbh0**bark_b2
        dib1 = jnp.sqrt(dib0**2 + dds)
        radial_increment = (dib1 - dib0) / 2
        dbh1 = ((dib1 - bark_b0) / bark_b1) ** (1 / bark_b2)

        return dbh1, (radial_increment, dbh1)

    # incorporate measurement error into DBH records
    dbh_start = dist.Laplace(dbh, dbh_err_func(dbh)).sample(PRNGKey(0))
    num_cycles = 2
    dbh_end, (radial_growth, dbh_series) = scan(step, dbh_start, jnp.arange(num_cycles))

    real_dbh = jnp.hstack((dbh_start.reshape(-1, 1), dbh_series.T))

    meas_dbh_next = dist.Laplace(real_dbh[:, 2], dbh_err_func(real_dbh[:, 2])).sample(
        PRNGKey(0)
    )
    meas_5yr = dist.Normal(
        radial_growth[1, :], rad_err_func(radial_growth[1, :])
    ).sample(PRNGKey(0))
    meas_10yr = dist.Normal(
        radial_growth[:2, :].sum(axis=0), rad_err_func(radial_growth[:2, :].sum(axis=0))
    ).sample(PRNGKey(0))

    return dbh_end, meas_5yr, meas_10yr


def wykoff_forward(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_variants,
    num_locations,
    num_plots,
    num_cycles,
    data,
    dbh_next=None,
    exist_5yr=None,
    obs_5yr=None,
    exist_10yr=None,
    obs_10yr=None,
):
    """A recursive model that predicts diameter growth time series
    for individual trees following the general form of Wykoff (1990)
    utilizing a five-year timestep along with an optional mixed effects approach
    for localities and plots following Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The model can be fit using an arbitrary number
    of time steps, and with three alternatives to incorporate hierarchical model
    structure across ecoregions: fully pooled, fully unpooled, and partially
    pooled.

    Data likelihoods for the model can accomodate three different types of
    observations: outside bark diameter measurements (DBH), and radial
    growth increments measured from years 0-10 or years 5-10.

    Measurement error for both DBH and radial increment are incorporated,
    resulting in a Bayesian state space model form.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
      ... TO DO ...
    tree_comp : str
      tree_level competion variable to use, options are:
      'bal', 'relht', or 'ballndbh'.
    stand_comp : str
      stand_level competition variable to use, options are:
      'ba', 'lnba', or 'ccf'
    num_cycles : int
      number of five-year steps (or cycles) of growth to simulate
    bark_b0, bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b0 + b1*(DBH**b2)
    obs_dbh : scalar or list-like,
        observed DBH measurements at 10 years since first measurement
    exist_5yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 5-yr radial increment
        recorded
    obs_5yr : scalar or list-like with shape (num_trees,)
        observed five-year radial increments
    exist_10yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 10-yr radial increment
        recorded
    obs_10yr : scalar or list-like with shape (num_trees,)
        observed ten-year radial increments
    """
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["bal", "ballndbh", "relht"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    (
        var_idx,
        loc_idx,
        plot_idx,
        site_index,
        slope,
        asp,
        elev,
        dbh,
        crown_ratio,
        bal,
        relht,
        bapa,
        ccf,
    ) = data

    num_trees = dbh.size
    variant = jnp.asarray(var_idx).reshape(-1)
    location = jnp.asarray(loc_idx).reshape(-1)
    plot = jnp.asarray(plot_idx).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    asp = jnp.asarray(asp).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)

    dbh = jnp.asarray(dbh).reshape(-1)
    dbh_next = jnp.asarray(dbh_next).reshape(-1)
    crown_ratio = jnp.asarray(crown_ratio)
    bal = jnp.asarray(bal)
    relht = jnp.asarray(relht)
    bapa = jnp.asarray(bapa)
    ccf = jnp.asarray(ccf)

    if pooling == "pooled":
        b0 = numpyro.sample("b0", dist.Normal(2.8, 1.0))
        b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))  # dbh**2
        b3 = numpyro.sample("b3", dist.Normal(0.0, 1.0))  # ln(site_index)
        b4 = numpyro.sample("b4", dist.Normal(0.0, 1.0))  # slope
        b5 = numpyro.sample("b5", dist.Normal(0.0, 1.0))  # slope**2
        b6 = numpyro.sample("b6", dist.Normal(0.0, 1.0))  # slsinasp
        b7 = numpyro.sample("b7", dist.Normal(0.0, 1.0))  # slcosasp
        b8 = numpyro.sample("b8", dist.Normal(0.0, 1.0))  # elev
        b9 = numpyro.sample("b9", dist.Normal(0.0, 1.0))  # elev**2
        b10 = numpyro.sample("b10", dist.Normal(0.0, 1.0))  # crown_ratio
        b11 = numpyro.sample("b11", dist.Normal(0.0, 1.0))  # crown_ratio**2
        b12 = numpyro.sample("b12", dist.Normal(0.0, 1.0))  # tree_comp
        b13 = numpyro.sample("b13", dist.Normal(0.0, 1.0))  # stand_comp
        etree = numpyro.sample(
            "etree", dist.InverseGamma(10.0, 5.0)
        )  # periodic (5-yr error)

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0 = numpyro.sample("b0", dist.Normal(2.8, 1.0))
            b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0))  # ln(dbh)
            b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))  # dbh**2
            b3 = numpyro.sample("b3", dist.Normal(0.0, 1.0))  # ln(site_index)
            b4 = numpyro.sample("b4", dist.Normal(0.0, 1.0))  # slope
            b5 = numpyro.sample("b5", dist.Normal(0.0, 1.0))  # slope**2
            b6 = numpyro.sample("b6", dist.Normal(0.0, 1.0))  # slsinasp
            b7 = numpyro.sample("b7", dist.Normal(0.0, 1.0))  # slcosasp
            b8 = numpyro.sample("b8", dist.Normal(0.0, 1.0))  # elev
            b9 = numpyro.sample("b9", dist.Normal(0.0, 1.0))  # elev**2
            b10 = numpyro.sample("b10", dist.Normal(0.0, 1.0))  # crown_ratio
            b11 = numpyro.sample("b11", dist.Normal(0.0, 1.0))  # crown_ratio**2
            b12 = numpyro.sample("b12", dist.Normal(0.0, 1.0))  # tree_comp
            b13 = numpyro.sample("b13", dist.Normal(0.0, 1.0))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    else:  # pooling == 'partial'
        b0_mu = numpyro.sample("b0_mu", dist.Normal(2.5, 0.1))
        b1_mu = numpyro.sample("b1_mu", dist.Normal(0.7, 0.1))  # ln(dbh)
        b2_mu = numpyro.sample("b2_mu", dist.Normal(-0.2, 0.1))  # dbh**2
        b3_mu = numpyro.sample("b3_mu", dist.Normal(0.25, 0.1))  # ln(site_index)
        b4_mu = numpyro.sample("b4_mu", dist.Normal(0.0, 0.1))  # slope
        b5_mu = numpyro.sample("b5_mu", dist.Normal(0.0, 0.11))  # slope**2
        b6_mu = numpyro.sample("b6_mu", dist.Normal(0.0, 0.05))  # slsinasp
        b7_mu = numpyro.sample("b7_mu", dist.Normal(0.0, 0.05))  # slcosasp
        b8_mu = numpyro.sample("b8_mu", dist.Normal(-0.3, 0.1))  # elev
        b9_mu = numpyro.sample("b9_mu", dist.Normal(0.15, 0.1))  # elev**2
        b10_mu = numpyro.sample("b10_mu", dist.Normal(0.8, 0.1))  # crown_ratio
        b11_mu = numpyro.sample("b11_mu", dist.Normal(-0.4, 0.1))  # crown_ratio**2
        b12_mu = numpyro.sample("b12_mu", dist.Normal(0.0, 0.25))  # tree_comp
        b13_mu = numpyro.sample("b13_mu", dist.Normal(0.0, 0.25))  # stand_comp
        b0_sd = numpyro.sample("b0_sd", dist.HalfNormal(1.0))
        b1_sd = numpyro.sample("b1_sd", dist.HalfNormal(0.1))
        b2_sd = numpyro.sample("b2_sd", dist.HalfNormal(0.1))
        b3_sd = numpyro.sample("b3_sd", dist.HalfNormal(0.1))
        b4_sd = numpyro.sample("b4_sd", dist.HalfNormal(0.1))
        b5_sd = numpyro.sample("b5_sd", dist.HalfNormal(0.1))
        b6_sd = numpyro.sample("b6_sd", dist.HalfNormal(0.1))
        b7_sd = numpyro.sample("b7_sd", dist.HalfNormal(0.1))
        b8_sd = numpyro.sample("b8_sd", dist.HalfNormal(0.1))
        b9_sd = numpyro.sample("b9_sd", dist.HalfNormal(0.1))
        b10_sd = numpyro.sample("b10_sd", dist.HalfNormal(0.1))
        b11_sd = numpyro.sample("b11_sd", dist.HalfNormal(0.1))
        b12_sd = numpyro.sample("b12_sd", dist.HalfNormal(0.1))
        b13_sd = numpyro.sample("b13_sd", dist.HalfNormal(0.1))

        with numpyro.plate("variants", num_variants):
            b0 = numpyro.sample("b0", dist.Normal(b0_mu, b0_sd))
            b1 = numpyro.sample("b1", dist.Normal(b1_mu, b1_sd))  # ln(dbh)
            b2 = numpyro.sample("b2", dist.Normal(b2_mu, b2_sd))  # dbh**2
            b3 = numpyro.sample("b3", dist.Normal(b3_mu, b3_sd))  # ln(site_index)
            b4 = numpyro.sample("b4", dist.Normal(b4_mu, b4_sd))  # slope
            b5 = numpyro.sample("b5", dist.Normal(b5_mu, b5_sd))  # slope**2
            b6 = numpyro.sample("b6", dist.Normal(b6_mu, b6_sd))  # slsinasp
            b7 = numpyro.sample("b7", dist.Normal(b7_mu, b7_sd))  # slcosasp
            b8 = numpyro.sample("b8", dist.Normal(b8_mu, b8_sd))  # elev
            b9 = numpyro.sample("b9", dist.Normal(b9_mu, b9_sd))  # elev**2
            b10 = numpyro.sample("b10", dist.Normal(b10_mu, b10_sd))  # crown_ratio
            b11 = numpyro.sample("b11", dist.Normal(b11_mu, b11_sd))  # crown_ratio**2
            b12 = numpyro.sample("b12", dist.Normal(b12_mu, b12_sd))  # tree_comp
            b13 = numpyro.sample("b13", dist.Normal(b13_mu, b13_sd))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc_", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = -eloc_.sum()
        eloc = numpyro.deterministic(
            "eloc", jnp.concatenate([eloc_, eloc_last.reshape(-1)])
        )
    else:
        eloc = 0 * location

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot_", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = -eplot_.sum()
        eplot = numpyro.deterministic(
            "eplot", jnp.concatenate([eplot_, eplot_last.reshape(-1)])
        )
    else:
        eplot = 0 * plot

    def step(dbh0, t):
        cr_ = crown_ratio[:, t]
        if tree_comp == "bal":
            tree_var = bal[:, t]
        elif tree_comp == "ballndbh":
            tree_var = bal[:, t] / jnp.log(dbh0 + 1.0)
        else:  # tree_comp == 'relht':
            tree_var = relht[:, t]

        if stand_comp == "ba":
            stand_var = bapa[:, t]
        elif stand_comp == "lnba":
            stand_var = jnp.log(bapa[:, t])
        else:  # stand_comp == 'ccf':
            stand_var = ccf[:, t]

        if pooling == "pooled":
            size = b1 * jnp.log(dbh0) + b2 * dbh0**2
            site = (
                b3 * jnp.log(site_index)
                + b4 * slope
                + b5 * slope**2
                + b6 * slope * jnp.sin(asp)
                + b7 * slope * jnp.cos(asp)
                + b8 * elev
                + b9 * elev**2
            )
            comp = b10 * cr_ + b11 * cr_**2 + b12 * tree_var + b13 * stand_var

            ln_dds_ = b0 + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree))

        else:
            size = b1[variant] * jnp.log(dbh0) + b2[variant] * dbh0**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope
                + b5[variant] * slope**2
                + b6[variant] * slope * jnp.sin(asp)
                + b7[variant] * slope * jnp.cos(asp)
                + b8[variant] * elev
                + b9[variant] * elev**2
            )
            comp = (
                b10[variant] * cr_
                + b11[variant] * cr_**2
                + b12[variant] * tree_var
                + b13[variant] * stand_var
            )

            ln_dds_ = b0[variant] + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree[variant]))

        dds = jnp.exp(ln_dds)
        dib0 = bark_b0 + bark_b1 * dbh0**bark_b2
        dib1 = jnp.sqrt(dib0**2 + dds)
        radial_increment = (dib1 - dib0) / 2
        dbh1 = ((dib1 - bark_b0) / bark_b1) ** (1 / bark_b2)

        return dbh1, (radial_increment, dbh1)

    # incorporate measurement error into DBH records
    # assuming FIA meets measurement quality objectives
    # dbh_start = numpyro.sample('dbh_start', dist.Normal(dbh, dbh/20.*0.1/1.65))

    # best estimate of DBH meas error in PNW from Melson et al. (2002)
    # Melson estimate is about 3.75 times larger than FIA MQO
    dbh_start = numpyro.sample("dbh_start", dist.Normal(dbh, 0.01 * dbh))

    dbh_end, (radial_growth, dbh_series) = scan(step, dbh_start, jnp.arange(num_cycles))

    real_dbh = numpyro.deterministic(
        "real_dbh", jnp.hstack((dbh_start.reshape(-1, 1), dbh_series.T))
    )

    meas_dbh_next = numpyro.sample(
        "meas_dbh_next",
        # assuming FIA meets measurement quality objectives
        # dist.Normal(real_dbh[:, 2], real_dbh[:, 2]/20.*0.1/1.65),
        # best estimate of DBH meas error in PNW from Melson et al. (2002)
        # Melson estimate is about 3.75 times larger than FIA MQO
        dist.Normal(real_dbh[:, 2], 0.01 * real_dbh[:, 2]),
        obs=dbh_next
        # to convert into annualized timestep where observations may or
        # may not be available in each year, use something like this...
        # dist.Normal(
        #     dbh_series[year_obs, tree_obs],
        #     dbh_series[year_obs, tree_obs]*.01
        # ),
        # obs=dbh_obs[tree_obs, year_obs]
    )

    meas_5yr = numpyro.sample(
        "meas_5yr",
        dist.Normal(
            radial_growth[1, exist_5yr], radial_growth[1, exist_5yr] / 20.0 / 1.65
        ),
        obs=obs_5yr[exist_5yr],
    )

    meas_10yr = numpyro.sample(
        "meas_10yr",
        dist.Normal(
            radial_growth[:2, exist_10yr].sum(axis=0),
            radial_growth[:2, exist_10yr].sum(axis=0) / 20.0 / 1.65,
        ),
        obs=obs_10yr[exist_10yr],
    )


def simpler_wykoff_forward(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_variants,
    num_locations,
    num_plots,
    num_cycles,
    data,
    dbh_next=None,
    exist_5yr=None,
    obs_5yr=None,
    exist_10yr=None,
    obs_10yr=None,
):
    """A recursive model that predicts diameter growth time series
    for individual trees following the general form of Wykoff (1990)
    utilizing a five-year timestep along with an optional mixed effects approach
    for localities and plots following Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The model can be fit using an arbitrary number
    of time steps, and with three alternatives to incorporate hierarchical model
    structure across ecoregions: fully pooled, fully unpooled, and partially
    pooled.

    Data likelihoods for the model can accomodate three different types of
    observations: outside bark diameter measurements (DBH), and radial
    growth increments measured from years 0-10 or years 5-10.

    Measurement error for both DBH and radial increment are incorporated,
    resulting in a Bayesian state space model form.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
      ... TO DO ...
    tree_comp : str
      tree_level competion variable to use, options are:
      'bal', 'relht', or 'ballndbh'.
    stand_comp : str
      stand_level competition variable to use, options are:
      'ba', 'lnba', or 'ccf'
    num_cycles : int
      number of five-year steps (or cycles) of growth to simulate
    bark_b0, bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b0 + b1*(DBH**b2)
    obs_dbh : scalar or list-like,
        observed DBH measurements at 10 years since first measurement
    exist_5yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 5-yr radial increment
        recorded
    obs_5yr : scalar or list-like with shape (num_trees,)
        observed five-year radial increments
    exist_10yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 10-yr radial increment
        recorded
    obs_10yr : scalar or list-like with shape (num_trees,)
        observed ten-year radial increments
    """
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["bal", "ballndbh", "relht"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    (
        var_idx,
        loc_idx,
        plot_idx,
        site_index,
        slope,
        asp,
        elev,
        dbh,
        crown_ratio,
        bal,
        relht,
        bapa,
        ccf,
    ) = data

    num_trees = dbh.size
    variant = jnp.asarray(var_idx).reshape(-1)
    location = jnp.asarray(loc_idx).reshape(-1)
    plot = jnp.asarray(plot_idx).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    asp = jnp.asarray(asp).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)
    dbh = jnp.asarray(dbh).reshape(-1)
    dbh_next = jnp.asarray(dbh_next).reshape(-1)
    crown_ratio = jnp.asarray(crown_ratio)
    bal = jnp.asarray(bal)
    relht = jnp.asarray(relht)
    bapa = jnp.asarray(bapa)
    ccf = jnp.asarray(ccf)

    if pooling == "pooled":
        b0 = numpyro.sample("b0", dist.Normal(2.5, 1.0))
        b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))  # dbh**2
        b3 = numpyro.sample("b3", dist.Normal(0.0, 1.0))  # ln(site_index)
        b4 = numpyro.sample("b4", dist.Normal(0.0, 1.0))  # slope
        b5 = numpyro.sample("b5", dist.Normal(0.0, 1.0))  # elev
        b6 = numpyro.sample("b6", dist.Normal(0.0, 1.0))  # crown_ratio
        b7 = numpyro.sample("b7", dist.Normal(0.0, 1.0))  # tree_comp
        b8 = numpyro.sample("b8", dist.Normal(0.0, 1.0))  # stand_comp
        etree = numpyro.sample(
            "etree", dist.InverseGamma(10.0, 5.0)
        )  # periodic (5-yr error)

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0 = numpyro.sample("b0", dist.Normal(2.5, 1.0))
            b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0))  # ln(dbh)
            b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))  # dbh**2
            b3 = numpyro.sample("b3", dist.Normal(0.0, 1.0))  # ln(site_index)
            b4 = numpyro.sample("b4", dist.Normal(0.0, 1.0))  # slope
            b5 = numpyro.sample("b5", dist.Normal(0.0, 1.0))  # elev
            b6 = numpyro.sample("b6", dist.Normal(0.0, 1.0))  # crown_ratio
            b7 = numpyro.sample("b7", dist.Normal(0.0, 1.0))  # tree_comp
            b8 = numpyro.sample("b8", dist.Normal(0.0, 1.0))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    else:  # pooling == 'partial'
        b0_mu = numpyro.sample("b0_mu", dist.Normal(2.5, 0.1))
        b1_mu = numpyro.sample("b1_mu", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2_mu = numpyro.sample("b2_mu", dist.Normal(0.0, 1.0))  # dbh**2
        b3_mu = numpyro.sample("b3_mu", dist.Normal(0.0, 1.0))  # ln(site_index)
        b4_mu = numpyro.sample("b4_mu", dist.Normal(0.0, 1.0))  # slope
        b5_mu = numpyro.sample("b5_mu", dist.Normal(0.0, 1.0))  # elev
        b6_mu = numpyro.sample("b6_mu", dist.Normal(0.0, 1.0))  # crown_ratio
        b7_mu = numpyro.sample("b7_mu", dist.Normal(0.0, 1.0))  # tree_comp
        b8_mu = numpyro.sample("b8_mu", dist.Normal(0.0, 1.0))  # stand_comp
        b0_sd = numpyro.sample("b0_sd", dist.HalfNormal(1.0))
        b1_sd = numpyro.sample("b1_sd", dist.HalfNormal(0.1))
        b2_sd = numpyro.sample("b2_sd", dist.HalfNormal(0.1))
        b3_sd = numpyro.sample("b3_sd", dist.HalfNormal(0.1))
        b4_sd = numpyro.sample("b4_sd", dist.HalfNormal(0.1))
        b5_sd = numpyro.sample("b5_sd", dist.HalfNormal(0.1))
        b6_sd = numpyro.sample("b6_sd", dist.HalfNormal(0.1))
        b7_sd = numpyro.sample("b7_sd", dist.HalfNormal(0.1))
        b8_sd = numpyro.sample("b8_sd", dist.HalfNormal(0.1))

        with numpyro.plate("variants", num_variants):
            b0 = numpyro.sample("b0", dist.Normal(b0_mu, b0_sd))
            b1 = numpyro.sample("b1", dist.Normal(b1_mu, b1_sd))  # ln(dbh)
            b2 = numpyro.sample("b2", dist.Normal(b2_mu, b2_sd))  # dbh**2
            b3 = numpyro.sample("b3", dist.Normal(b3_mu, b3_sd))  # ln(site_index)
            b4 = numpyro.sample("b4", dist.Normal(b4_mu, b4_sd))  # slope
            b5 = numpyro.sample("b5", dist.Normal(b5_mu, b5_sd))  # slope**2
            b6 = numpyro.sample("b6", dist.Normal(b6_mu, b6_sd))  # slsinasp
            b7 = numpyro.sample("b7", dist.Normal(b7_mu, b7_sd))  # slcosasp
            b8 = numpyro.sample("b8", dist.Normal(b8_mu, b8_sd))  # elev
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc_", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = -eloc_.sum()
        eloc = numpyro.deterministic(
            "eloc", jnp.concatenate([eloc_, eloc_last.reshape(-1)])
        )
    else:
        eloc = 0 * location

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot_", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = -eplot_.sum()
        eplot = numpyro.deterministic(
            "eplot", jnp.concatenate([eplot_, eplot_last.reshape(-1)])
        )
    else:
        eplot = 0 * plot

    def step(dbh0, t):
        cr_ = crown_ratio[:, t]
        if tree_comp == "bal":
            tree_var = bal[:, t]
        elif tree_comp == "ballndbh":
            tree_var = bal[:, t] / jnp.log(dbh0 + 1.0)
        else:  # tree_comp == 'relht':
            tree_var = relht[:, t]

        if stand_comp == "ba":
            stand_var = bapa[:, t]
        elif stand_comp == "lnba":
            stand_var = jnp.log(bapa[:, t])
        else:  # stand_comp == 'ccf':
            stand_var = ccf[:, t]

        if pooling == "pooled":
            size = b1 * jnp.log(dbh0) + b2 * dbh0**2
            site = b3 * jnp.log(site_index) + b4 * slope + b5 * elev**2
            comp = b6 * cr_ + b7 * tree_var + b8 * stand_var

            ln_dds_ = b0 + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree))

        else:
            size = b1[variant] * jnp.log(dbh0) + b2[variant] * dbh0**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope
                + b5[variant] * elev
            )
            comp = b6[variant] * cr_ + b7[variant] * tree_var + b8[variant] * stand_var

            ln_dds_ = b0[variant] + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree[variant]))

        dds = jnp.exp(ln_dds)
        dib0 = bark_b0 + bark_b1 * dbh0**bark_b2
        dib1 = jnp.sqrt(dib0**2 + dds)
        radial_increment = (dib1 - dib0) / 2
        dbh1 = ((dib1 - bark_b0) / bark_b1) ** (1 / bark_b2)

        return dbh1, (radial_increment, dbh1)

    # incorporate measurement error into DBH records
    # assuming FIA meets measurement quality objectives
    # dbh_start = numpyro.sample('dbh_start', dist.Normal(dbh, dbh/20.*0.1/1.65))

    # best estimate of DBH meas error in PNW from Melson et al. (2002)
    # Melson estimate is about 3.75 times larger than FIA MQO
    dbh_start = numpyro.sample("dbh_start", dist.Normal(dbh, 0.01 * dbh))

    dbh_end, (radial_growth, dbh_series) = scan(step, dbh_start, jnp.arange(num_cycles))

    real_dbh = numpyro.deterministic(
        "real_dbh", jnp.hstack((dbh_start.reshape(-1, 1), dbh_series.T))
    )

    meas_dbh_next = numpyro.sample(
        "meas_dbh_next",
        # assuming FIA meets measurement quality objectives
        # dist.Normal(real_dbh[:, 2], real_dbh[:, 2]/20.*0.1/1.65),
        # best estimate of DBH meas error in PNW from Melson et al. (2002)
        # Melson estimate is about 3.75 times larger than FIA MQO
        dist.Normal(real_dbh[:, 2], 0.01 * real_dbh[:, 2]),
        obs=dbh_next
        # to convert into annualized timestep where observations may or
        # may not be available in each year, use something like this...
        # dist.Normal(
        #     dbh_series[year_obs, tree_obs],
        #     dbh_series[year_obs, tree_obs]*.01
        # ),
        # obs=dbh_obs[tree_obs, year_obs]
    )

    meas_5yr = numpyro.sample(
        "meas_5yr",
        dist.Normal(
            radial_growth[1, exist_5yr], radial_growth[1, exist_5yr] / 20.0 / 1.65
        ),
        obs=obs_5yr[exist_5yr],
    )

    meas_10yr = numpyro.sample(
        "meas_10yr",
        dist.Normal(
            radial_growth[:2, exist_10yr].sum(axis=0),
            radial_growth[:2, exist_10yr].sum(axis=0) / 20.0 / 1.65,
        ),
        obs=obs_10yr[exist_10yr],
    )


def wykoff_model(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_variants,
    num_locations,
    num_plots,
    num_cycles,
    data,
    dbh_next=None,
    exist_5yr=None,
    obs_5yr=None,
    exist_10yr=None,
    obs_10yr=None,
):
    """A recursive model that predicts diameter growth time series
    for individual trees following the general form of Wykoff (1990)
    utilizing a five-year timestep along with an optional mixed effects approach
    for localities and plots following Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The model can be fit using an arbitrary number
    of time steps, and with three alternatives to incorporate hierarchical model
    structure across ecoregions: fully pooled, fully unpooled, and partially
    pooled.

    Data likelihoods for the model can accomodate three different types of
    observations: outside bark diameter measurements (DBH), and radial
    growth increments measured from years 0-10 or years 5-10.

    Measurement error for both DBH and radial increment are incorporated,
    resulting in a Bayesian state space model form.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
      ... TO DO ...
    tree_comp : str
      tree_level competion variable to use, options are:
      'bal', 'relht', or 'ballndbh'.
    stand_comp : str
      stand_level competition variable to use, options are:
      'ba', 'lnba', or 'ccf'
    num_cycles : int
      number of five-year steps (or cycles) of growth to simulate
    bark_b0, bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b0 + b1*(DBH**b2)
    obs_dbh : scalar or list-like,
        observed DBH measurements at 10 years since first measurement
    exist_5yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 5-yr radial increment
        recorded
    obs_5yr : scalar or list-like with shape (num_trees,)
        observed five-year radial increments
    exist_10yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 10-yr radial increment
        recorded
    obs_10yr : scalar or list-like with shape (num_trees,)
        observed ten-year radial increments
    """
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["bal", "ballndbh", "relht"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    (
        var_idx,
        loc_idx,
        plot_idx,
        site_index,
        slope,
        asp,
        elev,
        dbh,
        crown_ratio,
        bal,
        relht,
        bapa,
        ccf,
    ) = data

    num_trees = dbh.size
    variant = jnp.asarray(var_idx).reshape(-1)
    location = jnp.asarray(loc_idx).reshape(-1)
    plot = jnp.asarray(plot_idx).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    asp = jnp.asarray(asp).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)

    dbh = jnp.asarray(dbh).reshape(-1)
    dbh_next = jnp.asarray(dbh_next).reshape(-1)
    crown_ratio = jnp.asarray(crown_ratio)
    bal = jnp.asarray(bal)
    relht = jnp.asarray(relht)
    bapa = jnp.asarray(bapa)
    ccf = jnp.asarray(ccf)

    if tree_comp == "bal":
        X_tree = bal[:, 0]
    elif tree_comp == "relht":
        X_tree = relht[:, 0]
    else:  # tree_comp == 'ballndbh':
        X_tree = bal[:, 0] / jnp.log(dbh + 1.0)

    if stand_comp == "ba":
        X_stand = bapa[:, 0]
    elif stand_comp == "lnba":
        X_stand = jnp.log(bapa[:, 0])
    else:  # stand_comp == 'ccf':
        X_stand = ccf[:, 0]

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            slope**2,
            slope * jnp.sin(asp),
            slope * jnp.cos(asp),
            elev,
            elev**2,
            crown_ratio[:, 0],
            crown_ratio[:, 0] ** 2,
            X_tree,
            X_stand,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(2.5, 0.25))
        b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2
        b3z = numpyro.sample("b3z", dist.Normal(0.0, 1.0))  # ln(site_index)
        b4z = numpyro.sample("b4z", dist.Normal(0.0, 1.0))  # slope
        b5z = numpyro.sample("b5z", dist.Normal(0.0, 1.0))  # slope**2
        b6z = numpyro.sample("b6z", dist.Normal(0.0, 0.1))  # slsinasp
        b7z = numpyro.sample("b7z", dist.Normal(0.0, 0.1))  # slcosasp
        b8z = numpyro.sample("b8z", dist.Normal(0.0, 1.0))  # elev
        b9z = numpyro.sample("b9z", dist.Normal(0.0, 1.0))  # elev**2
        b10z = numpyro.sample("b10z", dist.Normal(0.0, 1.0))  # crown_ratio
        b11z = numpyro.sample("b11z", dist.Normal(0.0, 1.0))  # crown_ratio**2
        b12z = numpyro.sample("b12z", dist.Normal(0.0, 1.0))  # tree_comp
        b13z = numpyro.sample("b13z", dist.Normal(0.0, 1.0))  # stand_comp
        etree = numpyro.sample(
            "etree", dist.InverseGamma(10.0, 5.0)
        )  # periodic (5-yr error)

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(2.5, 0.25))
            b1z = numpyro.sample("b1z", dist.Normal(0.7, 0.25))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(-0.2, 0.25))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(0.25, 0.25))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(0.0, 0.25))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(0.0, 0.25))  # slope**2
            b6z = numpyro.sample("b6z", dist.Normal(0.0, 0.1))  # slsinasp
            b7z = numpyro.sample("b7z", dist.Normal(0.0, 0.1))  # slcosasp
            b8z = numpyro.sample("b8z", dist.Normal(-0.3, 0.25))  # elev
            b9z = numpyro.sample("b9z", dist.Normal(0.15, 0.25))  # elev**2
            b10z = numpyro.sample("b10z", dist.Normal(0.8, 0.25))  # crown_ratio
            b11z = numpyro.sample("b11z", dist.Normal(-0.4, 0.25))  # crown_ratio**2
            b12z = numpyro.sample("b12z", dist.Normal(0.0, 0.25))  # tree_comp
            b13z = numpyro.sample("b13z", dist.Normal(0.0, 0.25))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    else:  # pooling == 'partial'
        b0z_mu = numpyro.sample("b0z_mu", dist.Normal(2.5, 0.1))
        b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0.7, 0.1))  # ln(dbh)
        b2z_mu = numpyro.sample("b2z_mu", dist.Normal(-0.2, 0.1))  # dbh**2
        b3z_mu = numpyro.sample("b3z_mu", dist.Normal(0.25, 0.1))  # ln(site_index)
        b4z_mu = numpyro.sample("b4z_mu", dist.Normal(0.0, 0.1))  # slope
        b5z_mu = numpyro.sample("b5z_mu", dist.Normal(0.0, 0.11))  # slope**2
        b6z_mu = numpyro.sample("b6z_mu", dist.Normal(0.0, 0.05))  # slsinasp
        b7z_mu = numpyro.sample("b7z_mu", dist.Normal(0.0, 0.05))  # slcosasp
        b8z_mu = numpyro.sample("b8z_mu", dist.Normal(-0.3, 0.1))  # elev
        b9z_mu = numpyro.sample("b9z_mu", dist.Normal(0.15, 0.1))  # elev**2
        b10z_mu = numpyro.sample("b10z_mu", dist.Normal(0.8, 0.1))  # crown_ratio
        b11z_mu = numpyro.sample("b11z_mu", dist.Normal(-0.4, 0.1))  # crown_ratio**2
        b12z_mu = numpyro.sample("b12z_mu", dist.Normal(0.0, 0.25))  # tree_comp
        b13z_mu = numpyro.sample("b13z_mu", dist.Normal(0.0, 0.25))  # stand_comp
        b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(0.1))
        b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(0.1))
        b3z_sd = numpyro.sample("b3z_sd", dist.HalfNormal(0.1))
        b4z_sd = numpyro.sample("b4z_sd", dist.HalfNormal(0.1))
        b5z_sd = numpyro.sample("b5z_sd", dist.HalfNormal(0.1))
        b6z_sd = numpyro.sample("b6z_sd", dist.HalfNormal(0.1))
        b7z_sd = numpyro.sample("b7z_sd", dist.HalfNormal(0.1))
        b8z_sd = numpyro.sample("b8z_sd", dist.HalfNormal(0.1))
        b9z_sd = numpyro.sample("b9z_sd", dist.HalfNormal(0.1))
        b10z_sd = numpyro.sample("b10z_sd", dist.HalfNormal(0.1))
        b11z_sd = numpyro.sample("b11z_sd", dist.HalfNormal(0.1))
        b12z_sd = numpyro.sample("b12z_sd", dist.HalfNormal(0.1))
        b13z_sd = numpyro.sample("b13z_sd", dist.HalfNormal(0.1))

        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(b3z_mu, b3z_sd))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(b4z_mu, b4z_sd))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(b5z_mu, b5z_sd))  # slope**2
            b6z = numpyro.sample("b6z", dist.Normal(b6z_mu, b6z_sd))  # slsinasp
            b7z = numpyro.sample("b7z", dist.Normal(b7z_mu, b7z_sd))  # slcosasp
            b8z = numpyro.sample("b8z", dist.Normal(b8z_mu, b8z_sd))  # elev
            b9z = numpyro.sample("b9z", dist.Normal(b9z_mu, b9z_sd))  # elev**2
            b10z = numpyro.sample("b10z", dist.Normal(b10z_mu, b10z_sd))  # crown_ratio
            b11z = numpyro.sample(
                "b11z", dist.Normal(b11z_mu, b11z_sd)
            )  # crown_ratio**2
            b12z = numpyro.sample("b12z", dist.Normal(b12z_mu, b12z_sd))  # tree_comp
            b13z = numpyro.sample("b13z", dist.Normal(b13z_mu, b13z_sd))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    b5 = numpyro.deterministic("b5", b5z / X_sd[4])
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])
    b9 = numpyro.deterministic("b9", b9z / X_sd[8])
    b10 = numpyro.deterministic("b10", b10z / X_sd[9])
    b11 = numpyro.deterministic("b11", b11z / X_sd[10])
    b12 = numpyro.deterministic("b12", b12z / X_sd[11])
    b13 = numpyro.deterministic("b13", b13z / X_sd[12])

    adjust = (
        b1 * X_mu[0]
        + b2 * X_mu[1]
        + b3 * X_mu[2]
        + b4 * X_mu[3]
        + b5 * X_mu[4]
        + b6 * X_mu[5]
        + b7 * X_mu[6]
        + b8 * X_mu[7]
        + b9 * X_mu[8]
        + b10 * X_mu[9]
        + b11 * X_mu[10]
        + b12 * X_mu[11]
        + b13 * X_mu[12]
    )
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc_", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = -eloc_.sum()
        eloc = numpyro.deterministic(
            "eloc", jnp.concatenate([eloc_, eloc_last.reshape(-1)])
        )
    else:
        eloc = 0 * location

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot_", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = -eplot_.sum()
        eplot = numpyro.deterministic(
            "eplot", jnp.concatenate([eplot_, eplot_last.reshape(-1)])
        )
    else:
        eplot = 0 * plot

    def step(dbh0, t):
        cr_ = crown_ratio[:, t]
        if tree_comp == "bal":
            tree_var = bal[:, t]
        elif tree_comp == "ballndbh":
            tree_var = bal[:, t] / jnp.log(dbh0 + 1.0)
        else:  # tree_comp == 'relht':
            tree_var = relht[:, t]

        if stand_comp == "ba":
            stand_var = bapa[:, t]
        elif stand_comp == "lnba":
            stand_var = jnp.log(bapa[:, t])
        else:  # stand_comp == 'ccf':
            stand_var = ccf[:, t]

        if pooling == "pooled":
            size = b1 * jnp.log(dbh0) + b2 * dbh0**2
            site = (
                b3 * jnp.log(site_index)
                + b4 * slope
                + b5 * slope**2
                + b6 * slope * jnp.sin(asp)
                + b7 * slope * jnp.cos(asp)
                + b8 * elev
                + b9 * elev**2
            )
            comp = b10 * cr_ + b11 * cr_**2 + b12 * tree_var + b13 * stand_var

            ln_dds_ = b0 + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree))

        else:
            size = b1[variant] * jnp.log(dbh0) + b2[variant] * dbh0**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope
                + b5[variant] * slope**2
                + b6[variant] * slope * jnp.sin(asp)
                + b7[variant] * slope * jnp.cos(asp)
                + b8[variant] * elev
                + b9[variant] * elev**2
            )
            comp = (
                b10[variant] * cr_
                + b11[variant] * cr_**2
                + b12[variant] * tree_var
                + b13[variant] * stand_var
            )

            ln_dds_ = b0[variant] + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree[variant]))

        dds = jnp.exp(ln_dds)
        dib0 = bark_b0 + bark_b1 * dbh0**bark_b2
        dib1 = jnp.sqrt(dib0**2 + dds)
        radial_increment = (dib1 - dib0) / 2
        dbh1 = ((dib1 - bark_b0) / bark_b1) ** (1 / bark_b2)

        return dbh1, (radial_increment, dbh1)

    # incorporate measurement error into DBH records
    # assuming FIA meets measurement quality objectives
    # dbh_start = numpyro.sample('dbh_start', dist.Normal(dbh, dbh/20.*0.1/1.65))

    # best estimate of DBH meas error in PNW from Melson et al. (2002)
    # Melson estimate is about 3.75 times larger than FIA MQO
    # error is exponentially distributed, b[0]*dbh**b[1], b=[1.331459e-04, 2.178620e+00]
    dbh_start = numpyro.sample("dbh_start", dist.Normal(dbh, 0.01 * dbh))

    dbh_end, (radial_growth, dbh_series) = scan(step, dbh_start, jnp.arange(num_cycles))

    real_dbh = numpyro.deterministic(
        "real_dbh", jnp.hstack((dbh_start.reshape(-1, 1), dbh_series.T))
    )

    meas_dbh_next = numpyro.sample(
        "meas_dbh_next",
        # assuming FIA meets measurement quality objectives
        # dist.Normal(real_dbh[:, 2], real_dbh[:, 2]/20.*0.1/1.65),
        # best estimate of DBH meas error in PNW from Melson et al. (2002)
        # Melson estimate is about 3.75 times larger than FIA MQO
        dist.Normal(real_dbh[:, 2], 0.01 * real_dbh[:, 2]),
        obs=dbh_next
        # to convert into annualized timestep where observations may or
        # may not be available in each year, use something like this...
        # dist.Normal(
        #     dbh_series[year_obs, tree_obs],
        #     dbh_series[year_obs, tree_obs]*.01
        # ),
        # obs=dbh_obs[tree_obs, year_obs]
    )

    meas_5yr = numpyro.sample(
        "meas_5yr",
        dist.Normal(
            radial_growth[1, exist_5yr], radial_growth[1, exist_5yr] / 20.0 / 1.65
        ),
        obs=obs_5yr[exist_5yr],
    )

    meas_10yr = numpyro.sample(
        "meas_10yr",
        dist.Normal(
            radial_growth[:2, exist_10yr].sum(axis=0),
            radial_growth[:2, exist_10yr].sum(axis=0) / 20.0 / 1.65,
        ),
        obs=obs_10yr[exist_10yr],
    )


def simpler_wykoff_model(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_variants,
    num_locations,
    num_plots,
    num_cycles,
    data,
    dbh_next=None,
    exist_5yr=None,
    obs_5yr=None,
    exist_10yr=None,
    obs_10yr=None,
):
    """A recursive model that predicts diameter growth time series
    for individual trees following the general form of Wykoff (1990)
    utilizing a five-year timestep along with an optional mixed effects approach
    for localities and plots following Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The model can be fit using an arbitrary number
    of time steps, and with three alternatives to incorporate hierarchical model
    structure across ecoregions: fully pooled, fully unpooled, and partially
    pooled.

    Data likelihoods for the model can accomodate three different types of
    observations: outside bark diameter measurements (DBH), and radial
    growth increments measured from years 0-10 or years 5-10.

    Measurement error for both DBH and radial increment are incorporated,
    resulting in a Bayesian state space model form.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
      ... TO DO ...
    tree_comp : str
      tree_level competion variable to use, options are:
      'bal', 'relht', or 'ballndbh'.
    stand_comp : str
      stand_level competition variable to use, options are:
      'ba', 'lnba', or 'ccf'
    num_cycles : int
      number of five-year steps (or cycles) of growth to simulate
    bark_b0, bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b0 + b1*(DBH**b2)
    obs_dbh : scalar or list-like,
        observed DBH measurements at 10 years since first measurement
    exist_5yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 5-yr radial increment
        recorded
    obs_5yr : scalar or list-like with shape (num_trees,)
        observed five-year radial increments
    exist_10yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 10-yr radial increment
        recorded
    obs_10yr : scalar or list-like with shape (num_trees,)
        observed ten-year radial increments
    """
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["bal", "ballndbh", "relht"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    (
        var_idx,
        loc_idx,
        plot_idx,
        site_index,
        slope,
        asp,
        elev,
        dbh,
        crown_ratio,
        bal,
        relht,
        bapa,
        ccf,
    ) = data

    num_trees = dbh.size
    variant = jnp.asarray(var_idx).reshape(-1)
    location = jnp.asarray(loc_idx).reshape(-1)
    plot = jnp.asarray(plot_idx).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    asp = jnp.asarray(asp).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)

    dbh = jnp.asarray(dbh).reshape(-1)
    dbh_next = jnp.asarray(dbh_next).reshape(-1)
    crown_ratio = jnp.asarray(crown_ratio)
    bal = jnp.asarray(bal)
    relht = jnp.asarray(relht)
    bapa = jnp.asarray(bapa)
    ccf = jnp.asarray(ccf)

    if tree_comp == "bal":
        X_tree = bal[:, 0]
    elif tree_comp == "relht":
        X_tree = relht[:, 0]
    else:  # tree_comp == 'ballndbh':
        X_tree = bal[:, 0] / jnp.log(dbh + 1.0)

    if stand_comp == "ba":
        X_stand = bapa[:, 0]
    elif stand_comp == "lnba":
        X_stand = jnp.log(bapa[:, 0])
    else:  # stand_comp == 'ccf':
        X_stand = ccf[:, 0]

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            elev,
            crown_ratio[:, 0],
            X_tree,
            X_stand,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    if pooling == "pooled":
        b0z = numpyro.sample("b0z", dist.Normal(2.5, 0.25))
        b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
        b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2
        b3z = numpyro.sample("b3z", dist.Normal(0.0, 1.0))  # ln(site_index)
        b4z = numpyro.sample("b4z", dist.Normal(0.0, 1.0))  # slope
        b5z = numpyro.sample("b5z", dist.Normal(0.0, 1.0))  # elev
        b6z = numpyro.sample("b6z", dist.Normal(0.0, 1.0))  # crown_ratio
        b7z = numpyro.sample("b7z", dist.Normal(0.0, 1.0))  # tree_comp
        b8z = numpyro.sample("b8z", dist.Normal(0.0, 1.0))  # stand_comp
        etree = numpyro.sample(
            "etree", dist.InverseGamma(10.0, 5.0)
        )  # periodic (5-yr error)

    elif pooling == "unpooled":
        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(2.5, 0.25))
            b1z = numpyro.sample("b1z", dist.Normal(0.7, 0.25))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(-0.2, 0.25))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(0.25, 0.25))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(0.0, 0.25))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(-0.3, 0.25))  # elev
            b6z = numpyro.sample("b6z", dist.Normal(0.8, 0.25))  # crown_ratio
            b7z = numpyro.sample("b7z", dist.Normal(0.0, 0.25))  # tree_comp
            b8z = numpyro.sample("b8z", dist.Normal(0.0, 0.25))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    else:  # pooling == 'partial'
        b0z_mu = numpyro.sample("b0z_mu", dist.Normal(2.5, 0.1))
        b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0.7, 0.1))  # ln(dbh)
        b2z_mu = numpyro.sample("b2z_mu", dist.Normal(-0.2, 0.1))  # dbh**2
        b3z_mu = numpyro.sample("b3z_mu", dist.Normal(0.25, 0.1))  # ln(site_index)
        b4z_mu = numpyro.sample("b4z_mu", dist.Normal(0.0, 0.1))  # slope
        b5z_mu = numpyro.sample("b5z_mu", dist.Normal(-0.3, 0.1))  # elev
        b6z_mu = numpyro.sample("b6z_mu", dist.Normal(0.8, 0.1))  # crown_ratio
        b7z_mu = numpyro.sample("b7z_mu", dist.Normal(0.0, 0.25))  # tree_comp
        b8z_mu = numpyro.sample("b8z_mu", dist.Normal(0.0, 0.25))  # stand_comp
        b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(0.1))
        b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(0.1))
        b3z_sd = numpyro.sample("b3z_sd", dist.HalfNormal(0.1))
        b4z_sd = numpyro.sample("b4z_sd", dist.HalfNormal(0.1))
        b5z_sd = numpyro.sample("b5z_sd", dist.HalfNormal(0.1))
        b6z_sd = numpyro.sample("b6z_sd", dist.HalfNormal(0.1))
        b7z_sd = numpyro.sample("b7z_sd", dist.HalfNormal(0.1))
        b8z_sd = numpyro.sample("b8z_sd", dist.HalfNormal(0.1))

        with numpyro.plate("variants", num_variants):
            b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(b3z_mu, b3z_sd))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(b4z_mu, b4z_sd))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(b5z_mu, b5z_sd))  # elev
            b6z = numpyro.sample("b6z", dist.Normal(b6z_mu, b6z_sd))  # crown_ratio
            b7z = numpyro.sample("b7z", dist.Normal(b7z_mu, b7z_sd))  # tree_comp
            b8z = numpyro.sample("b8z", dist.Normal(b8z_mu, b8z_sd))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    b5 = numpyro.deterministic("b5", b5z / X_sd[4])
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])

    adjust = (
        b1 * X_mu[0]
        + b2 * X_mu[1]
        + b3 * X_mu[2]
        + b4 * X_mu[3]
        + b5 * X_mu[4]
        + b6 * X_mu[5]
        + b7 * X_mu[6]
        + b8 * X_mu[7]
    )
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with numpyro.plate("locations", num_locations - 1):
            eloc_ = numpyro.sample(
                "eloc_", dist.Normal(0, 0.1)
            )  # random location effect
        eloc_last = -eloc_.sum()
        eloc = numpyro.deterministic(
            "eloc", jnp.concatenate([eloc_, eloc_last.reshape(-1)])
        )
    else:
        eloc = 0 * location

    if plot_random:
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot_", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = -eplot_.sum()
        eplot = numpyro.deterministic(
            "eplot", jnp.concatenate([eplot_, eplot_last.reshape(-1)])
        )
    else:
        eplot = 0 * plot

    def step(dbh0, t):
        cr_ = crown_ratio[:, t]
        if tree_comp == "bal":
            tree_var = bal[:, t]
        elif tree_comp == "ballndbh":
            tree_var = bal[:, t] / jnp.log(dbh0 + 1.0)
        else:  # tree_comp == 'relht':
            tree_var = relht[:, t]

        if stand_comp == "ba":
            stand_var = bapa[:, t]
        elif stand_comp == "lnba":
            stand_var = jnp.log(bapa[:, t])
        else:  # stand_comp == 'ccf':
            stand_var = ccf[:, t]

        if pooling == "pooled":
            size = b1 * jnp.log(dbh0) + b2 * dbh0**2
            site = b3 * jnp.log(site_index) + b4 * slope + b5 * elev
            comp = b6 * cr_ + b7 * tree_var + b8 * stand_var

            ln_dds_ = b0 + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree))

        else:
            size = b1[variant] * jnp.log(dbh0) + b2[variant] * dbh0**2
            site = (
                b3[variant] * jnp.log(site_index)
                + b4[variant] * slope
                + b5[variant] * elev
            )
            comp = b6[variant] * cr_ + b7[variant] * tree_var + b8[variant] * stand_var

            ln_dds_ = b0[variant] + size + site + comp + eloc[location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree[variant]))

        dds = jnp.exp(ln_dds)
        dib0 = bark_b0 + bark_b1 * dbh0**bark_b2
        dib1 = jnp.sqrt(dib0**2 + dds)
        radial_increment = (dib1 - dib0) / 2
        dbh1 = ((dib1 - bark_b0) / bark_b1) ** (1 / bark_b2)

        return dbh1, (radial_increment, dbh1)

    # incorporate measurement error into DBH records
    # assuming FIA meets measurement quality objectives
    # dbh_start = numpyro.sample('dbh_start', dist.Normal(dbh, dbh/20.*0.1/1.65))

    # best estimate of DBH meas error in PNW from Melson et al. (2002)
    # Melson estimate is about 3.75 times larger than FIA MQO
    # error is exponentially distributed, b[0]*dbh**b[1], b=[1.331459e-04, 2.178620e+00]
    dbh_start = numpyro.sample("dbh_start", dist.Normal(dbh, 0.01 * dbh))

    dbh_end, (radial_growth, dbh_series) = scan(step, dbh_start, jnp.arange(num_cycles))

    real_dbh = numpyro.deterministic(
        "real_dbh", jnp.hstack((dbh_start.reshape(-1, 1), dbh_series.T))
    )

    meas_dbh_next = numpyro.sample(
        "meas_dbh_next",
        # assuming FIA meets measurement quality objectives
        # dist.Normal(real_dbh[:, 2], real_dbh[:, 2]/20.*0.1/1.65),
        # best estimate of DBH meas error in PNW from Melson et al. (2002)
        # Melson estimate is about 3.75 times larger than FIA MQO
        dist.Normal(real_dbh[:, 2], 0.01 * real_dbh[:, 2]),
        obs=dbh_next
        # to convert into annualized timestep where observations may or
        # may not be available in each year, use something like this...
        # dist.Normal(
        #     dbh_series[year_obs, tree_obs],
        #     dbh_series[year_obs, tree_obs]*.01
        # ),
        # obs=dbh_obs[tree_obs, year_obs]
    )

    meas_5yr = numpyro.sample(
        "meas_5yr",
        dist.Normal(
            radial_growth[1, exist_5yr], radial_growth[1, exist_5yr] / 20.0 / 1.65
        ),
        obs=obs_5yr[exist_5yr],
    )

    meas_10yr = numpyro.sample(
        "meas_10yr",
        dist.Normal(
            radial_growth[:2, exist_10yr].sum(axis=0),
            radial_growth[:2, exist_10yr].sum(axis=0) / 20.0 / 1.65,
        ),
        obs=obs_10yr[exist_10yr],
    )


def simpler_wykoff_multispecies_model(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_species,
    num_variants,
    num_locations,
    num_plots,
    num_cycles,
    data,
    dbh_next=None,
    exist_5yr=None,
    obs_5yr=None,
    exist_10yr=None,
    obs_10yr=None,
):
    """A recursive model that predicts diameter growth time series
    for individual trees following the general form of Wykoff (1990)
    utilizing a five-year timestep along with an optional mixed effects approach
    for localities and plots following Weiskittel et al., (2007).

    The Wykoff model employs a linear model to predict the log of the
    difference between squared inside-bark diameter from one timestep to the
    next. A mixed effects model form is used, with fixed effect for tree size,
    site variables, and competition variables, and random effects for each
    location, and for each plot. The model can be fit using an arbitrary number
    of time steps, and with three alternatives to incorporate hierarchical model
    structure across ecoregions: fully pooled, fully unpooled, and partially
    pooled.

    Data likelihoods for the model can accomodate three different types of
    observations: outside bark diameter measurements (DBH), and radial
    growth increments measured from years 0-10 or years 5-10.

    Measurement error for both DBH and radial increment are incorporated,
    resulting in a Bayesian state space model form.

    Parameters
    ----------
    data : list-like
      predictor variables, expected in the following order:
      ... TO DO ...
    tree_comp : str
      tree_level competion variable to use, options are:
      'bal', 'relht', or 'ballndbh'.
    stand_comp : str
      stand_level competition variable to use, options are:
      'ba', 'lnba', or 'ccf'
    num_cycles : int
      number of five-year steps (or cycles) of growth to simulate
    bark_b0, bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside
      bark, as DIB = b0 + b1*(DBH**b2)
    obs_dbh : scalar or list-like,
        observed DBH measurements at 10 years since first measurement
    exist_5yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 5-yr radial increment
        recorded
    obs_5yr : scalar or list-like with shape (num_trees,)
        observed five-year radial increments
    exist_10yr : bool or list-like of bools, shape (num_trees,)
        boolean value or array indicating trees that have 10-yr radial increment
        recorded
    obs_10yr : scalar or list-like with shape (num_trees,)
        observed ten-year radial increments
    """
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["bal", "ballndbh", "relht"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    (
        spp_idx,  # added this variable compared to single-species wykoff
        var_idx,
        loc_idx,
        plot_idx,
        site_index,
        slope,
        asp,
        elev,
        dbh,
        crown_ratio,
        bal,
        relht,
        bapa,
        ccf,
    ) = data

    bark_b0 = jnp.asarray(bark_b0).reshape(-1)
    bark_b1 = jnp.asarray(bark_b1).reshape(-1)
    bark_b2 = jnp.asarray(bark_b2).reshape(-1)
    num_trees = dbh.size
    spp = jnp.asarray(spp_idx).reshape(-1)
    variant = jnp.asarray(var_idx).reshape(-1)
    location = jnp.asarray(loc_idx).reshape(-1)
    plot = jnp.asarray(plot_idx).reshape(-1)
    site_index = jnp.asarray(site_index).reshape(-1)
    slope = jnp.asarray(slope).reshape(-1)
    asp = jnp.asarray(asp).reshape(-1)
    elev = jnp.asarray(elev).reshape(-1)

    dbh = jnp.asarray(dbh).reshape(-1)
    dbh_next = jnp.asarray(dbh_next).reshape(-1)
    crown_ratio = jnp.asarray(crown_ratio)
    bal = jnp.asarray(bal)
    relht = jnp.asarray(relht)
    bapa = jnp.asarray(bapa)
    ccf = jnp.asarray(ccf)

    if tree_comp == "bal":
        X_tree = bal[:, 0]
    elif tree_comp == "relht":
        X_tree = relht[:, 0]
    else:  # tree_comp == 'ballndbh':
        X_tree = bal[:, 0] / jnp.log(dbh + 1.0)

    if stand_comp == "ba":
        X_stand = bapa[:, 0]
    elif stand_comp == "lnba":
        X_stand = jnp.log(bapa[:, 0])
    else:  # stand_comp == 'ccf':
        X_stand = ccf[:, 0]

    X = jnp.array(
        [
            jnp.log(dbh),
            dbh**2,
            jnp.log(site_index),
            slope,
            elev,
            crown_ratio[:, 0],
            X_tree,
            X_stand,
        ]
    )
    X_mu = X.mean(axis=1)
    X_sd = X.std(axis=1)

    plate_spp = numpyro.plate("species", num_species, dim=-2)
    plate_var = numpyro.plate("variants", num_variants, dim=-1)

    if pooling == "pooled":
        with plate_spp:
            b0z = numpyro.sample("b0z", dist.Normal(2.5, 0.25))
            b1z = numpyro.sample("b1z", dist.Normal(0.0, 1.0))  # ln(dbh)
            b2z = numpyro.sample("b2z", dist.Normal(0.0, 1.0))  # dbh**2
            b3z = numpyro.sample("b3z", dist.Normal(0.0, 1.0))  # ln(site_index)
            b4z = numpyro.sample("b4z", dist.Normal(0.0, 1.0))  # slope
            b5z = numpyro.sample("b5z", dist.Normal(0.0, 1.0))  # elev
            b6z = numpyro.sample("b6z", dist.Normal(0.0, 1.0))  # crown_ratio
            b7z = numpyro.sample("b7z", dist.Normal(0.0, 1.0))  # tree_comp
            b8z = numpyro.sample("b8z", dist.Normal(0.0, 1.0))  # stand_comp
            etree = numpyro.sample(
                "etree", dist.InverseGamma(10.0, 5.0)
            )  # periodic (5-yr error)

    elif pooling == "unpooled":
        with plate_spp:
            with plate_var:
                b0z = numpyro.sample("b0z", dist.Normal(2.5, 0.25))
                b1z = numpyro.sample("b1z", dist.Normal(0.7, 0.25))  # ln(dbh)
                b2z = numpyro.sample("b2z", dist.Normal(-0.2, 0.25))  # dbh**2
                b3z = numpyro.sample("b3z", dist.Normal(0.25, 0.25))  # ln(site_index)
                b4z = numpyro.sample("b4z", dist.Normal(0.0, 0.25))  # slope
                b5z = numpyro.sample("b5z", dist.Normal(-0.3, 0.25))  # elev
                b6z = numpyro.sample("b6z", dist.Normal(0.8, 0.25))  # crown_ratio
                b7z = numpyro.sample("b7z", dist.Normal(0.0, 0.25))  # tree_comp
                b8z = numpyro.sample("b8z", dist.Normal(0.0, 0.25))  # stand_comp
                etree = numpyro.sample(
                    "etree", dist.InverseGamma(10.0, 5.0)
                )  # periodic (5-yr error)

    else:  # pooling == 'partial'
        with plate_spp:
            b0z_mu = numpyro.sample("b0z_mu", dist.Normal(2.5, 0.1))
            b1z_mu = numpyro.sample("b1z_mu", dist.Normal(0.7, 0.1))  # ln(dbh)
            b2z_mu = numpyro.sample("b2z_mu", dist.Normal(-0.2, 0.1))  # dbh**2
            b3z_mu = numpyro.sample("b3z_mu", dist.Normal(0.25, 0.1))  # ln(site_index)
            b4z_mu = numpyro.sample("b4z_mu", dist.Normal(0.0, 0.1))  # slope
            b5z_mu = numpyro.sample("b5z_mu", dist.Normal(-0.3, 0.1))  # elev
            b6z_mu = numpyro.sample("b6z_mu", dist.Normal(0.8, 0.1))  # crown_ratio
            b7z_mu = numpyro.sample("b7z_mu", dist.Normal(0.0, 0.25))  # tree_comp
            b8z_mu = numpyro.sample("b8z_mu", dist.Normal(0.0, 0.25))  # stand_comp
            b0z_sd = numpyro.sample("b0z_sd", dist.HalfNormal(1.0))
            b1z_sd = numpyro.sample("b1z_sd", dist.HalfNormal(0.1))
            b2z_sd = numpyro.sample("b2z_sd", dist.HalfNormal(0.1))
            b3z_sd = numpyro.sample("b3z_sd", dist.HalfNormal(0.1))
            b4z_sd = numpyro.sample("b4z_sd", dist.HalfNormal(0.1))
            b5z_sd = numpyro.sample("b5z_sd", dist.HalfNormal(0.1))
            b6z_sd = numpyro.sample("b6z_sd", dist.HalfNormal(0.1))
            b7z_sd = numpyro.sample("b7z_sd", dist.HalfNormal(0.1))
            b8z_sd = numpyro.sample("b8z_sd", dist.HalfNormal(0.1))

            with plate_var:
                b0z = numpyro.sample("b0z", dist.Normal(b0z_mu, b0z_sd))
                b1z = numpyro.sample("b1z", dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
                b2z = numpyro.sample("b2z", dist.Normal(b2z_mu, b2z_sd))  # dbh**2
                b3z = numpyro.sample(
                    "b3z", dist.Normal(b3z_mu, b3z_sd)
                )  # ln(site_index)
                b4z = numpyro.sample("b4z", dist.Normal(b4z_mu, b4z_sd))  # slope
                b5z = numpyro.sample("b5z", dist.Normal(b5z_mu, b5z_sd))  # elev
                b6z = numpyro.sample("b6z", dist.Normal(b6z_mu, b6z_sd))  # crown_ratio
                b7z = numpyro.sample("b7z", dist.Normal(b7z_mu, b7z_sd))  # tree_comp
                b8z = numpyro.sample("b8z", dist.Normal(b8z_mu, b8z_sd))  # stand_comp
                etree = numpyro.sample(
                    "etree", dist.InverseGamma(10.0, 5.0)
                )  # periodic (5-yr error)

    b1 = numpyro.deterministic("b1", b1z / X_sd[0])
    b2 = numpyro.deterministic("b2", b2z / X_sd[1])
    b3 = numpyro.deterministic("b3", b3z / X_sd[2])
    b4 = numpyro.deterministic("b4", b4z / X_sd[3])
    b5 = numpyro.deterministic("b5", b5z / X_sd[4])
    b6 = numpyro.deterministic("b6", b6z / X_sd[5])
    b7 = numpyro.deterministic("b7", b7z / X_sd[6])
    b8 = numpyro.deterministic("b8", b8z / X_sd[7])

    adjust = (
        b1 * X_mu[0]
        + b2 * X_mu[1]
        + b3 * X_mu[2]
        + b4 * X_mu[3]
        + b5 * X_mu[4]
        + b6 * X_mu[5]
        + b7 * X_mu[6]
        + b8 * X_mu[7]
    )
    b0 = numpyro.deterministic("b0", b0z - adjust)

    if loc_random:
        with plate_spp:
            with numpyro.plate("locations", num_locations - 1, dim=-1):
                eloc_ = numpyro.sample(
                    "eloc_", dist.Normal(0, 0.1)
                )  # random location effect
        eloc_last = -eloc_.sum(axis=1, keepdims=True)
        eloc = numpyro.deterministic("eloc", jnp.hstack([eloc_, eloc_last]))
    else:
        eloc = jnp.zeros((num_species, num_locations))

    if plot_random:  # not adding a separate plot effect for each species
        with numpyro.plate("plots", num_plots - 1):
            eplot_ = numpyro.sample(
                "eplot_", dist.Normal(0, 0.1)
            )  # random effect of plot
        eplot_last = -eplot_.sum()
        eplot = numpyro.deterministic(
            "eplot", jnp.concatenate([eplot_, eplot_last.reshape(-1)])
        )
    else:
        eplot = 0 * plot

    def step(dbh0, t):
        cr_ = crown_ratio[:, t]
        if tree_comp == "bal":
            tree_var = bal[:, t]
        elif tree_comp == "ballndbh":
            tree_var = bal[:, t] / jnp.log(dbh0 + 1.0)
        else:  # tree_comp == 'relht':
            tree_var = relht[:, t]

        if stand_comp == "ba":
            stand_var = bapa[:, t]
        elif stand_comp == "lnba":
            stand_var = jnp.log(bapa[:, t])
        else:  # stand_comp == 'ccf':
            stand_var = ccf[:, t]

        if pooling == "pooled":
            size = b1[spp] * jnp.log(dbh0) + b2[spp] * dbh0**2
            site = b3[spp] * jnp.log(site_index) + b4[spp] * slope + b5[spp] * elev
            comp = b6[spp] * cr_ + b7[spp] * tree_var + b8[spp] * stand_var

            ln_dds_ = b0[spp] + size + site + comp + eloc[spp, location] + eplot[plot]
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree[spp]))

        else:
            size = b1[spp, variant] * jnp.log(dbh0) + b2[spp, variant] * dbh0**2
            site = (
                b3[spp, variant] * jnp.log(site_index)
                + b4[spp, variant] * slope
                + b5[spp, variant] * elev
            )
            comp = (
                b6[spp, variant] * cr_
                + b7[spp, variant] * tree_var
                + b8[spp, variant] * stand_var
            )

            ln_dds_ = (
                b0[spp, variant]
                + size
                + site
                + comp
                + eloc[spp, location]
                + eplot[plot]
            )
            ln_dds = numpyro.sample("ln_dds", dist.Normal(ln_dds_, etree[spp, variant]))

        dds = jnp.exp(ln_dds)
        dib0 = bark_b0[spp] + bark_b1[spp] * dbh0 ** bark_b2[spp]
        dib1 = jnp.sqrt(dib0**2 + dds)
        radial_increment = (dib1 - dib0) / 2
        dbh1 = ((dib1 - bark_b0[spp]) / bark_b1[spp]) ** (1 / bark_b2[spp])

        return dbh1, (radial_increment, dbh1)

    # incorporate measurement error into DBH records
    # assuming FIA meets measurement quality objectives
    # dbh_start = numpyro.sample('dbh_start', dist.Normal(dbh, dbh/20.*0.1/1.65))

    # best estimate of DBH meas error in PNW from Melson et al. (2002)
    # Melson estimate is about 3.75 times larger than FIA MQO
    # error is exponentially distributed, b[0]*dbh**b[1], b=[1.331459e-04, 2.178620e+00]
    dbh_start = numpyro.sample("dbh_start", dist.Normal(dbh, 0.01 * dbh))

    dbh_end, (radial_growth, dbh_series) = scan(step, dbh_start, jnp.arange(num_cycles))

    real_dbh = numpyro.deterministic(
        "real_dbh", jnp.hstack((dbh_start.reshape(-1, 1), dbh_series.T))
    )
    meas_dbh_next = numpyro.sample(
        "meas_dbh_next",
        # assuming FIA meets measurement quality objectives
        # dist.Normal(real_dbh[:, 2], real_dbh[:, 2]/20.*0.1/1.65),
        # best estimate of DBH meas error in PNW from Melson et al. (2002)
        # Melson estimate is about 3.75 times larger than FIA MQO
        dist.Normal(real_dbh[:, 2], 0.01 * real_dbh[:, 2]),
        obs=dbh_next
        # to convert into annualized timestep where observations may or
        # may not be available in each year, use something like this...
        # dist.Normal(
        #     dbh_series[year_obs, tree_obs],
        #     dbh_series[year_obs, tree_obs]*.01
        # ),
        # obs=dbh_obs[tree_obs, year_obs]
    )

    meas_5yr = numpyro.sample(
        "meas_5yr",
        dist.Normal(
            radial_growth[1, exist_5yr], radial_growth[1, exist_5yr] / 20.0 / 1.65
        ),
        obs=obs_5yr[exist_5yr],
    )

    meas_10yr = numpyro.sample(
        "meas_10yr",
        dist.Normal(
            radial_growth[:2, exist_10yr].sum(axis=0),
            radial_growth[:2, exist_10yr].sum(axis=0) / 20.0 / 1.65,
        ),
        obs=obs_10yr[exist_10yr],
    )
