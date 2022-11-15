import numpyro
from jax import numpy as jnp
from jax.lax import scan

import numpyro.distributions as dist
from refit_fvs.models.distributions import NegativeGamma, NegativeHalfNormal


def wykoff_model(X, num_steps, bark_b1, bark_b2,
                 num_variants, num_locations, num_plots, 
                 y=None, pooling='unpooled'):
    """A recursive model that predicts basal area increment for individual trees
    following the general form of Wykoff (1990), with annualization inspired
    by Cao (2000, 2004), and mixed model approach illustrated by Weiskittel
    et al., (2007). 
    
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
    X : list-like
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
    num_steps : int
      number of steps (or cycles) of growth to simulate
    y : scalar or list-like
      observed outside-bark diameter growth
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside 
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'. 
     """
    (variant, location, plot, site_index, slope, elev, dbh, cr_start, cr_end, 
     comp_tree_start, comp_tree_end, comp_stand_start, comp_stand_end) = X
    
    dbh = jnp.asarray(dbh).reshape(-1,)
    y = jnp.asarray(y).reshape(-1,)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1,)
    location = jnp.asarray(location).reshape(-1,)    
    plot = jnp.asarray(plot).reshape(-1,)
    site_index = jnp.asarray(site_index).reshape(-1,)
    slope = jnp.asarray(slope).reshape(-1,)
    elev = jnp.asarray(elev).reshape(-1,)
    crown_ratio = jnp.linspace(cr_start, cr_end, num_steps)
    comp_tree = jnp.linspace(comp_tree_start, comp_tree_end, num_steps)
    comp_stand = jnp.linspace(comp_stand_start, comp_stand_end, num_steps)
    
    norm = jnp.array([
        jnp.log(dbh),
        dbh**2,
        jnp.log(site_index),
        slope,
        elev,
        cr_start,
        comp_tree_start,
        comp_stand_start
    ])
    norm_mu = norm.mean(axis=1)
    norm_sd = norm.std(axis=1)
    
    if pooling == 'pooled':
        b0z = numpyro.sample('b0z', dist.Normal(0.9, 2.))
        b1z = numpyro.sample('b1z', dist.Normal(0.7, 1.))  # ln(dbh)
        b2z = numpyro.sample('b2z', dist.Normal(-0.1, 1.))  # dbh**2
        b3z = numpyro.sample('b3z', dist.Normal(0.3, 1.))  # ln(site_index)
        b4z = numpyro.sample('b4z', dist.Normal(-0.04, 1.))  # slope
        b5z = numpyro.sample('b5z', dist.Normal(-0.1, 1.))  # elev
        b6z = numpyro.sample('b6z', dist.Normal(0.4, 1.))  # crown_ratio
        b7z = numpyro.sample('b7z', dist.Normal(-0.4, 1.))  # comp_tree  # BAL / ln(dbh+1)
        b8z = numpyro.sample('b8z', dist.Normal(0., 1.))  # comp_stand  # ln(BA)
    
    elif pooling == 'unpooled':
        with numpyro.plate('variants', num_variants):    
            b0z = numpyro.sample('b0z', dist.Normal(0.9, 2.))
            b1z = numpyro.sample('b1z', dist.Normal(0.7, 1.))  # ln(dbh)
            b2z = numpyro.sample('b2z', dist.Normal(-0.1, 1.))  # dbh**2
            b3z = numpyro.sample('b3z', dist.Normal(0.3, 1.))  # ln(site_index)
            b4z = numpyro.sample('b4z', dist.Normal(-0.04, 1.))  # slope
            b5z = numpyro.sample('b5z', dist.Normal(-0.1, 1.))  # elev
            b6z = numpyro.sample('b6z', dist.Normal(0.4, 1.))  # crown_ratio
            b7z = numpyro.sample('b7z', dist.Normal(-0.4, 1.))  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample('b8z', dist.Normal(0., 1.))  # comp_stand  # ln(BA)
    
    elif pooling == 'partial':
        b0z_mu = numpyro.sample('b0z_mu', dist.Normal(0.9, 0.5))
        b1z_mu = numpyro.sample('b1z_mu', dist.Normal(0.7, 0.3))
        b2z_mu = numpyro.sample('b2z_mu', dist.Normal(-0.1, 0.1))
        b3z_mu = numpyro.sample('b3z_mu', dist.Normal(0.3, 0.1))
        b4z_mu = numpyro.sample('b4z_mu', dist.Normal(-0.04, 0.1))
        b5z_mu = numpyro.sample('b5z_mu', dist.Normal(-0.1, 0.1))
        b6z_mu = numpyro.sample('b6z_mu', dist.Normal(0.4, 0.1))
        b7z_mu = numpyro.sample('b7z_mu', dist.Normal(-0.4, 0.1))
        b8z_mu = numpyro.sample('b8z_mu', dist.Normal(0., 0.1))
        b0z_sd = numpyro.sample('b0z_sd', dist.HalfNormal(1.0))
        b1z_sd = numpyro.sample('b1z_sd', dist.HalfNormal(0.05))
        b2z_sd = numpyro.sample('b2z_sd', dist.HalfNormal(0.05))
        b3z_sd = numpyro.sample('b3z_sd', dist.HalfNormal(0.05))
        b4z_sd = numpyro.sample('b4z_sd', dist.HalfNormal(0.05))
        b5z_sd = numpyro.sample('b5z_sd', dist.HalfNormal(0.05))
        b6z_sd = numpyro.sample('b6z_sd', dist.HalfNormal(0.05))
        b7z_sd = numpyro.sample('b7z_sd', dist.HalfNormal(0.05))
        b8z_sd = numpyro.sample('b8z_sd', dist.HalfNormal(0.05))
        
        with numpyro.plate('variants', num_variants):
            b0z = numpyro.sample('b0z', dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample('b1z', dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample('b2z', dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample('b3z', dist.Normal(b3z_mu, b3z_sd))  # ln(site_index)
            b4z = numpyro.sample('b4z', dist.Normal(b4z_mu, b4z_sd))  # slope
            b5z = numpyro.sample('b5z', dist.Normal(b5z_mu, b5z_sd))  # elev
            b6z = numpyro.sample('b6z', dist.Normal(b6z_mu, b6z_sd))  # crown_ratio
            b7z = numpyro.sample('b7z', dist.Normal(b7z_mu, b7z_sd))  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample('b8z', dist.Normal(b8z_mu, b8z_sd))  # comp_stand  # ln(BA)
    else:
        raise(ValueError("valid options for pooling are 'unpooled', 'pooled', or 'partial'"))
    
    b1 = numpyro.deterministic('b1', b1z/norm_sd[0])
    b2 = numpyro.deterministic('b2', b2z/norm_sd[1])
    b3 = numpyro.deterministic('b3', b3z/norm_sd[2])
    b4 = numpyro.deterministic('b4', b4z/norm_sd[3])
    b5 = numpyro.deterministic('b5', b5z/norm_sd[4])
    b6 = numpyro.deterministic('b6', b6z/norm_sd[5])
    b7 = numpyro.deterministic('b7', b7z/norm_sd[6])
    b8 = numpyro.deterministic('b8', b8z/norm_sd[7])

    adjust = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8)
    b0 = numpyro.deterministic('b0', b0z - adjust)
    
    eloc_mu = numpyro.sample('eloc_mu', dist.Normal(0., 0.5))
    eloc_sd = numpyro.sample('eloc_sd', dist.HalfNormal(1.0))
    with numpyro.plate('locations', num_locations):
        eloc = numpyro.sample('eloc', dist.Normal(eloc_mu, eloc_sd)) # random effect of location
    
    if y is not None:
        eplot_mu = numpyro.sample('eplot_mu', dist.Normal(0., 0.5))
        eplot_sd = numpyro.sample('eplot_sd', dist.HalfNormal(1.0))
        with numpyro.plate('plots', num_plots):
            eplot = numpyro.sample('eplot', dist.Normal(eplot_mu, eplot_sd)) # random effect of plot
    else:
        eplot = 0 * plot

    def step(dbh, step_covars):
        crown_ratio, comp_tree, comp_stand = step_covars
        if pooling == 'pooled':
            size = b1z * (jnp.log(dbh) - norm_mu[0])/norm_sd[0] + \
                   b2z * (dbh**2 - norm_mu[1])/norm_sd[1]
            site = b3z * (jnp.log(site_index) - norm_mu[2])/norm_sd[2] + \
                   b4z * (slope - norm_mu[3])/norm_sd[3] + \
                   b5z * (elev - norm_mu[4])/norm_sd[4]
            comp = b6z * (crown_ratio - norm_mu[5])/norm_sd[5] + \
                   b7z * (comp_tree - norm_mu[6])/norm_sd[6] + \
                   b8z * (comp_stand - norm_mu[7])/norm_sd[7]
                   
            ln_dds = b0z + size + site + comp + eloc[location] + eplot[plot]
        
        else:
            size = b1z[variant] * (jnp.log(dbh) - norm_mu[0])/norm_sd[0] + \
                   b2z[variant] * (dbh**2 - norm_mu[1])/norm_sd[1]
            site = b3z[variant] * (jnp.log(site_index) - norm_mu[2])/norm_sd[2] + \
                   b4z[variant] * (slope - norm_mu[3])/norm_sd[3] + \
                   b5z[variant] * (elev - norm_mu[4])/norm_sd[4]
            comp = b6z[variant] * (crown_ratio - norm_mu[5])/norm_sd[5] + \
                   b7z[variant] * (comp_tree - norm_mu[6])/norm_sd[6] + \
                   b8z[variant] * (comp_stand - norm_mu[7])/norm_sd[7]
            
            ln_dds = b0z[variant] + size + site + comp + eloc[location] + eplot[plot]
        
        dds = jnp.exp(ln_dds)
        dib_start = bark_b1*dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end/bark_b1)**(1/bark_b2)
        dg_ob = dbh_end - dbh
       
        return dbh_end, dg_ob
        
    step_covars = (crown_ratio, comp_tree, comp_stand)
    dbh_end, growth = scan(step, dbh, step_covars, length=num_steps)
    bai_pred = numpyro.deterministic('bai_pred', jnp.pi/4*(dbh_end**2 - dbh**2))
    
    etree = numpyro.sample('etree', dist.Gamma(4.0, 0.1))

    with numpyro.plate('plate_obs', size=num_trees):        
        obs = numpyro.sample('obs', dist.Cauchy(bai_pred, etree), obs=y)
    
    if y is not None:
        err = numpyro.sample('err', dist.Cauchy(0, etree))
        resid = numpyro.deterministic('resid', bai_pred + err - y)
        bias = numpyro.deterministic('bias', resid.mean())
        mae = numpyro.deterministic('mae', jnp.abs(resid).mean())
        rmse = numpyro.deterministic('rmse', jnp.sqrt((resid**2).sum() / len(resid)))
        
        
def potential_modified_model(X, num_steps, bark_b1, bark_b2,
                             num_variants, num_locations, num_plots, 
                             y=None, pooling='unpooled'):
    """A recursive model that predicts basal area increment for individual trees
    adapted from the general form of Wykoff (1990), with annualization inspired
    by Cao (2000, 2004), and mixed model approach illustrated by Weiskittel
    et al., (2007). 
    
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
    X : list-like
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
    num_steps : int
      number of steps (or cycles) of growth to simulate
    y : scalar or list-like
      observed outside-bark diameter growth
    bark_b1, bark_b2 : scalar
      coefficients used to convert diameter outside bark to diameter inside 
      bark, as DIB = b1*(DBH**b2)
    num_variants : int
      number of distinct variants or ecoregions across dataset
    num_locations : int
      number of distinct locations across dataset, modeled as a random effect
    num_plots : int
      number of distinct plots across dataset, modeled as a random effect
    pooling : str
      degree of pooling for model covariates across variants/ecoregions, valid
      are 'unpooled', 'pooled', or 'partial'. 
     """
    
    (variant, location, plot, site_index, slope, elev, dbh, cr_start, cr_end, 
     comp_tree_start, comp_tree_end, comp_stand_start, comp_stand_end) = X
    
    dbh = jnp.asarray(dbh).reshape(-1,)
    y = jnp.asarray(y).reshape(-1,)
    num_trees = dbh.size
    variant = jnp.asarray(variant).reshape(-1,)
    location = jnp.asarray(location).reshape(-1,)    
    plot = jnp.asarray(plot).reshape(-1,)
    site_index = 250/jnp.asarray(site_index).reshape(-1,)
    slope = jnp.asarray(slope).reshape(-1,)
    elev = jnp.asarray(elev).reshape(-1,)
    crown_ratio = jnp.linspace(1-cr_start, 1-cr_end, num_steps)
    comp_tree = jnp.linspace(comp_tree_start, comp_tree_end, num_steps)
    comp_stand = jnp.linspace(comp_stand_start, comp_stand_end, num_steps)
    
    norm = jnp.array([
        jnp.log(dbh),
        dbh**2,
        jnp.log(site_index),
        slope,
        elev,
        1-cr_start,
        comp_tree_start,
        comp_stand_start
    ])
    norm_mu = norm.mean(axis=1)
    norm_sd = norm.std(axis=1)
    
    if pooling == 'pooled':
        b0z = numpyro.sample('b0z', dist.Normal(0.7, 2.))
        b1z = numpyro.sample('b1z', dist.Normal(0.8, 1.))  # ln(dbh)
        b2z = numpyro.sample('b2z', dist.Normal(-0.1, 1.))  # dbh**2
        b3z = numpyro.sample('b3z', NegativeHalfNormal(2))  # ln(site_index)
        b4z = numpyro.sample('b4z', NegativeHalfNormal(2))  # slope
        b5z = numpyro.sample('b5z', NegativeHalfNormal(2))  # elev
        b6z = numpyro.sample('b6z', NegativeHalfNormal(2))   # crown_ratio
        b7z = numpyro.sample('b7z', NegativeHalfNormal(2))  # comp_tree  # BAL / ln(dbh+1)
        b8z = numpyro.sample('b8z', NegativeHalfNormal(2))  # comp_stand  # ln(BA)
    
    elif pooling == 'unpooled':
        with numpyro.plate('variants', num_variants):    
            b0z = numpyro.sample('b0z', dist.Normal(0.7, 2.))
            b1z = numpyro.sample('b1z', dist.Normal(0.8, 1.))  # ln(dbh)
            b2z = numpyro.sample('b2z', dist.Normal(-0.1, 1.))  # dbh**2
            b3z = numpyro.sample('b3z', NegativeHalfNormal(2))  # ln(site_index)
            b4z = numpyro.sample('b4z', NegativeHalfNormal(2))  # slope
            b5z = numpyro.sample('b5z', NegativeHalfNormal(2))  # elev
            b6z = numpyro.sample('b6z', NegativeHalfNormal(2))  # crown_ratio
            b7z = numpyro.sample('b7z', NegativeHalfNormal(2))  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample('b8z', NegativeHalfNormal(2))  # comp_stand  # ln(BA)
    
    elif pooling == 'partial':
        b0z_mu = numpyro.sample('b0z_mu', dist.Normal(0., 2.))
        b1z_mu = numpyro.sample('b1z_mu', dist.Normal(1., 2.))
        b2z_mu = numpyro.sample('b2z_mu', dist.Normal(0., 2.))
        b0z_sd = numpyro.sample('b0z_sd', dist.InverseGamma(1.2, 1.1))
        b1z_sd = numpyro.sample('b1z_sd', dist.InverseGamma(0.15, 4.))
        b2z_sd = numpyro.sample('b2z_sd', dist.InverseGamma(0.12, 3.5))
        
        b3z_conc = numpyro.sample('b3z_conc', NegativeHalfNormal(2))
        b3z_scale = numpyro.sample('b3z_scale', dist.HalfNormal(5))
        b4z_conc = numpyro.sample('b4z_conc', NegativeHalfNormal(2))
        b4z_scale = numpyro.sample('b4z_scale', dist.HalfNormal(5))
        b5z_conc = numpyro.sample('b5z_conc', NegativeHalfNormal(2))
        b5z_scale = numpyro.sample('b5z_scale', dist.HalfNormal(5))
        b6z_conc = numpyro.sample('b6z_conc', NegativeHalfNormal(2))
        b6z_scale = numpyro.sample('b6z_scale', dist.HalfNormal(5))
        b7z_conc = numpyro.sample('b7z_conc', NegativeHalfNormal(2))
        b7z_scale = numpyro.sample('b7z_scale', dist.HalfNormal(5))
        b8z_conc = numpyro.sample('b8z_conc', NegativeHalfNormal(2))
        b8z_scale = numpyro.sample('b8z_scale', dist.HalfNormal(5))
    
        with numpyro.plate('variants', num_variants):
            b0z = numpyro.sample('b0z', dist.Normal(b0z_mu, b0z_sd))
            b1z = numpyro.sample('b1z', dist.Normal(b1z_mu, b1z_sd))  # ln(dbh)
            b2z = numpyro.sample('b2z', dist.Normal(b2z_mu, b2z_sd))  # dbh**2
            b3z = numpyro.sample('b3z', NegativeGamma(b3z_conc, 1/b3z_scale))  # ln(site_index)
            b4z = numpyro.sample('b4z', NegativeGamma(b4z_conc, 1/b4z_scale))  # slope
            b5z = numpyro.sample('b5z', NegativeGamma(b5z_conc, 1/b5z_scale))  # elev
            b6z = numpyro.sample('b6z', NegativeGamma(b6z_conc, 1/b6z_scale))  # crown_ratio
            b7z = numpyro.sample('b7z', NegativeGamma(b7z_conc, 1/b7z_scale))  # comp_tree  # BAL / ln(dbh+1)
            b8z = numpyro.sample('b8z', NegativeGamma(b8z_conc, 1/b8z_scale))  # comp_stand  # ln(BA)
    else:
        raise(ValueError("valid options for pooling are 'unpooled', 'pooled', or 'partial'"))
    
    b1 = numpyro.deterministic('b1', b1z/norm_sd[0])
    b2 = numpyro.deterministic('b2', b2z/norm_sd[1])
    b3 = numpyro.deterministic('b3', b3z/norm_sd[2])
    b4 = numpyro.deterministic('b4', b4z/norm_sd[3])
    b5 = numpyro.deterministic('b5', b5z/norm_sd[4])
    b6 = numpyro.deterministic('b6', b6z/norm_sd[5])
    b7 = numpyro.deterministic('b7', b7z/norm_sd[6])
    b8 = numpyro.deterministic('b8', b8z/norm_sd[7])

    adjust = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8)
    b0 = numpyro.deterministic('b0', b0z - adjust)
    
    eloc_mu = numpyro.sample('eloc_mu', dist.Normal(0., 0.5))
    eloc_sd = numpyro.sample('eloc_sd', dist.HalfNormal(1.0))
    with numpyro.plate('locations', num_locations):
        eloc = numpyro.sample('eloc', dist.Normal(eloc_mu, eloc_sd)) # random effect of location
    
    if y is not None:
        eplot_mu = numpyro.sample('eplot_mu', dist.Normal(0., 0.5))
        eplot_sd = numpyro.sample('eplot_sd', dist.HalfNormal(1.0))
        with numpyro.plate('plots', num_plots):
            eplot = numpyro.sample('eplot', dist.Normal(eplot_mu, eplot_sd)) # random effect of plot
    else:
        eplot = 0 * plot

    def step(dbh, step_covars):
        crown_ratio, comp_tree, comp_stand = step_covars
        if pooling == 'pooled':
            size = b1z * (jnp.log(dbh) - norm_mu[0])/norm_sd[0] + \
                   b2z * (dbh**2 - norm_mu[1])/norm_sd[1]
            site = b3z * (jnp.log(site_index) - norm_mu[2])/norm_sd[2] + \
                   b4z * (slope - norm_mu[3])/norm_sd[3] + \
                   b5z * (elev - norm_mu[4])/norm_sd[4]
            comp = b6z * (crown_ratio - norm_mu[5])/norm_sd[5] + \
                   b7z * (comp_tree - norm_mu[6])/norm_sd[6] + \
                   b8z * (comp_stand - norm_mu[7])/norm_sd[7]
                   
            ln_dds = b0z + size + site + comp + eloc[location] + eplot[plot]
        
        else:
            size = b1z[variant] * (jnp.log(dbh) - norm_mu[0])/norm_sd[0] + \
                   b2z[variant] * (dbh**2 - norm_mu[1])/norm_sd[1]
            site = b3z[variant] * (jnp.log(site_index) - norm_mu[2])/norm_sd[2] + \
                   b4z[variant] * (slope - norm_mu[3])/norm_sd[3] + \
                   b5z[variant] * (elev - norm_mu[4])/norm_sd[4]
            comp = b6z[variant] * (crown_ratio - norm_mu[5])/norm_sd[5] + \
                   b7z[variant] * (comp_tree - norm_mu[6])/norm_sd[6] + \
                   b8z[variant] * (comp_stand - norm_mu[7])/norm_sd[7]
            
            ln_dds = b0z[variant] + size + site + comp + eloc[location] + eplot[plot]
        
        dds = jnp.exp(ln_dds)
        dib_start = bark_b1*dbh**bark_b2
        dib_end = jnp.sqrt(dib_start**2 + dds)
        dbh_end = (dib_end/bark_b1)**(1/bark_b2)
        dg_ob = dbh_end - dbh
       
        return dbh_end, dg_ob
        
    step_covars = (crown_ratio, comp_tree, comp_stand)
    dbh_end, growth = scan(step, dbh, step_covars, length=num_steps)
    bai_pred = numpyro.deterministic('bai_pred', jnp.pi/4*(dbh_end**2 - dbh**2))
    
    etree = numpyro.sample('etree', dist.Gamma(4.0, 0.1))

    with numpyro.plate('plate_obs', size=num_trees):        
        obs = numpyro.sample('obs', dist.Cauchy(bai_pred, etree), obs=y)
    
    if y is not None:
        err = numpyro.sample('err', dist.Cauchy(0, etree))
        resid = numpyro.deterministic('resid', bai_pred + err - y)
        bias = numpyro.deterministic('bias', resid.mean())
        mae = numpyro.deterministic('mae', jnp.abs(resid).mean())
        rmse = numpyro.deterministic('rmse', jnp.sqrt((resid**2).sum() / len(resid)))
        
        
        
class DiameterGrowth():
    def __init__(self, species, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, bark_ratio1, bark_ratio2):
        self.spp = species
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b6 = b6
        self.b7 = b7
        self.b8 = b8
        self.b9 = b9
        self.b10 = b10
        self.b11 = b11
        self.b12 = b12
        self.b13 = b13
        self.bark1 = bark_ratio1
        self.bark2 = bark_ratio2
    
    def grow(self, dbh, site_index, asp, slope, elev, 
             crown_ratio, comp_tree, comp_stand, num_steps,
             crown_ratio_end = None, comp_tree_end = None,
             comp_stand_end = None):
        
        dbh = jnp.asarray(dbh).reshape(-1,)
        num_trees = dbh.size
        site_index = jnp.asarray(site_index).reshape(-1,)
        asp = jnp.asarray(asp).reshape(-1,)
        slope = jnp.asarray(slope).reshape(-1,)
        elev = jnp.asarray(elev).reshape(-1,)
        site_index = jnp.full((num_steps, num_trees), site_index)
        asp = jnp.full((num_steps, num_trees), site_index) 
        slope = jnp.full((num_steps, num_trees), slope)
        elev = jnp.full((num_steps, num_trees), elev)
        
        if crown_ratio_end is not None:
            cr = jnp.linspace(crown_ratio, crown_ratio_end, num_steps)
        else:
            cr = jnp.full((num_steps, num_trees), crown_ratio)
        
        if comp_tree_end is not None:
            comp_tree = jnp.linspace(comp_tree, comp_tree_end, num_steps)
        else:
            comp_tree = jnp.full((num_steps, num_trees), comp_tree)
        
        if comp_stand_end is not None:
            comp_stand = jnp.linspace(comp_stand, comp_stand_end, num_steps)
        else:
            comp_stand = jnp.full((num_steps, num_trees), comp_stand)
            
        covars = site_index, asp, slope, elev, cr, comp_tree, comp_stand
        
        def step(dbh, covars):
            site_index, asp, slope, elev, crown_ratio, comp_tree, comp_stand = covars

            size = self.b1 * jnp.log(dbh) + self.b2 * (dbh**2)
            site = self.b3 * jnp.log(site_index) + \
                   self.b4 * (slope * jnp.cos(asp)) + self.b5 * (slope * jnp.sin(asp)) + \
                   self.b6 * slope + self.b7 * slope**2 + \
                   self.b8 * elev + self.b9 * elev ** 2
            comp = self.b10 * crown_ratio + self.b11 * crown_ratio**2 + \
                   self.b12 * comp_tree + self.b13 * comp_stand
            ln_dds = self.b0 + size + site + comp

            dds = jnp.exp(ln_dds)

            dib_start = self.bark1*(dbh**self.bark2)
            dg_ib = jnp.sqrt(dib_start**2 + dds) - dib_start
            dib_end = dib_start + dg_ib
            dbh_end = (dib_end / self.bark1)**(1/self.bark2)
            dg_ob = dbh_end - dbh

            return dbh_end, dg_ob
    
        # scan returns arrays of final dbh (shape: (n_trees,)) 
        # and of incremental growth (shape: (n_steps, n_trees))
        end_dbhs, growth = scan(step, dbh, covars, length=num_steps)       

        # return ending dbh for each tree, as well as incremental outside-bark diameter growth
        return end_dbhs, growth
    
DF_PN_COEFS = dict(  # 5-year coefs
    b0 = -0.739354,    # intercept
    b1 = 0.80,    # ln(dbh)
    b2 = -0.0000896,  # dbh**2
    b3 = 0.49,    # ln(site_index)
    b4 = 0.014165,    # slope*cos(asp)
    b5 = 0.003263,       # slope*sin(asp)
    b6 = -0.340401,    # slope
    b7 = 0,        # slope**2
    b8 = -0.009845,       # elev 
    b9 = 0,       # elev**2
    b10 = 1.936912,   # crown_ratio
    b11 = 0.,     # crown_ratio**2
    b12 = -0.001827,  # comp_tree (BAL/ln(dbh+1))
    b13 = -0.129474,   # comp_stand (lnBA)
    bark_ratio1 = 0.903563,
    bark_ratio2 = 0.989388,
)