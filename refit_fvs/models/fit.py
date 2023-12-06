from refit_fvs.models.diameter_growth import (
    wykoff_model,
    simpler_wykoff_model,
    simpler_wykoff_multispecies_model,
)
from refit_fvs.data.load import (
    prepare_data_for_modeling,
    prepare_data_for_modeling_multispecies,
)

import numpy as np
import glob

from jax import random

import pickle
import os

from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.reparam import LocScaleReparam
from numpyro.handlers import reparam


def fit_wykoff(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_cycles,
    data,
    model_name,
    checkpoint_dir,
    num_warmup=1000,
    num_samples=500,
    num_chains=1,
    chain_method="parallel",
    num_batches=1,
    seed=42,
    progress_bar=True,
    overwrite=False,
):
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["relht", "bal", "ballndbh"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    print(model_name)

    model_args, model_kwargs = prepare_data_for_modeling(
        bark_b0,
        bark_b1,
        bark_b2,
        tree_comp,
        stand_comp,
        pooling,
        loc_random,
        plot_random,
        num_cycles,
        data,
    )

    if pooling == "partial":
        config = {f"b{i}z": LocScaleReparam(0) for i in range(14)}
        model = reparam(wykoff_model, config=config)
        dense_mass = [
            ("b0z_decentered", "etree"),  # intercept and dds noise
            ("b1z_decentered", "b2z_decentered"),  # ln(dbh) and dbh**2
            ("b4z_decentered", "b5z_decentered"),  # slope and slope**2
            ("b8z_decentered", "b9z_decentered"),  # elev and elev**2
            ("b10z_decentered", "b11z_decentered"),  # crown ratio and crown ratio**2
            ("b12z_decentered", "b13z_decentered"),  # competition effects
        ]
    else:
        model = wykoff_model
        dense_mass = [
            ("b0z", "etree"),  # intercept and dds noise
            ("b1z", "b2z"),  # ln(dbh) and dbh**2
            ("b4z", "b5z"),  # slope and slope**2
            ("b8z", "b9z"),  # elev and elev**2
            ("b10z", "b11z"),  # crown ratio and crown ratio**2
            ("b12z", "b13z"),  # competition effects
        ]
    nuts_kernel = NUTS(
        model=model,
        dense_mass=dense_mass,
        init_strategy=init_to_median,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    already_done = np.sort(glob.glob(f"{checkpoint_dir}/{model_name}_*_mcmc.pkl"))

    if len(already_done) == 0 or overwrite:
        to_do = np.arange(num_batches)
    else:
        to_do = np.arange(num_batches)[len(already_done) :]

    if len(to_do) == 0:
        print("Already done.")
        print("MCMC saved at", already_done[-1])
        return

    for i in to_do:
        batch_name = model_name + f"_batch{i:02d}"
        print(batch_name)
        mcmc = MCMC(
            nuts_kernel,
            progress_bar=progress_bar,
            num_chains=num_chains,
            chain_method=chain_method,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )

        if i > 0:
            with open(already_done[-1], "rb") as g:
                mcmc.post_warmup_state = pickle.load(g)._last_state

        mcmc.run(
            random.PRNGKey(seed),
            *model_args,
            **model_kwargs,
        )

        samples = mcmc.get_samples()
        with open(f"{checkpoint_dir}/{batch_name}.pkl", "wb") as f:
            pickle.dump((samples), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{checkpoint_dir}/{batch_name}_mcmc.pkl", "wb") as f2:
            pickle.dump((mcmc), f2, protocol=pickle.HIGHEST_PROTOCOL)
        already_done = np.sort(glob.glob(f"{checkpoint_dir}/{model_name}_*_mcmc.pkl"))

    print("Done.")
    print("Samples saved at", f"{checkpoint_dir}/{batch_name}.pkl")
    print("MCMC saved at", f"{checkpoint_dir}/{batch_name}_mcmc.pkl")
    return


def fit_simpler_wykoff(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_cycles,
    data,
    model_name,
    checkpoint_dir,
    num_warmup=1000,
    num_samples=500,
    num_chains=1,
    chain_method="parallel",
    num_batches=1,
    seed=42,
    progress_bar=True,
    overwrite=False,
):
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["relht", "bal", "ballndbh"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    print(model_name)

    model_args, model_kwargs = prepare_data_for_modeling(
        bark_b0,
        bark_b1,
        bark_b2,
        tree_comp,
        stand_comp,
        pooling,
        loc_random,
        plot_random,
        num_cycles,
        data,
    )

    if pooling == "partial":
        config = {f"b{i}z": LocScaleReparam(0) for i in range(14)}
        model = reparam(simpler_wykoff_model, config=config)
        dense_mass = [
            ("b0z_decentered", "etree"),  # intercept and dds noise
            ("b1z_decentered", "b2z_decentered"),  # ln(dbh) and dbh**2
            ("b7z_decentered", "b8z_decentered"),  # competition effects
        ]
    else:
        model = simpler_wykoff_model
        dense_mass = [
            ("b0z", "etree"),  # intercept and dds noise
            ("b1z", "b2z"),  # ln(dbh) and dbh**2
            ("b7z", "b8z"),  # competition effects
        ]
    nuts_kernel = NUTS(
        model=model,
        dense_mass=dense_mass,
        init_strategy=init_to_median,
    )

    already_done = np.sort(glob.glob(f"{checkpoint_dir}/{model_name}_*_mcmc.pkl"))

    if len(already_done) == 0 or overwrite:
        to_do = np.arange(num_batches)
    else:
        to_do = np.arange(num_batches)[len(already_done) :]

    if len(to_do) == 0:
        print("Already done.")
        print("MCMC saved at", already_done[-1])
        return

    for i in to_do:
        batch_name = model_name + f"_batch{i:02d}"
        print(batch_name)
        mcmc = MCMC(
            nuts_kernel,
            progress_bar=progress_bar,
            num_chains=num_chains,
            chain_method=chain_method,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )

        if i > 0:
            with open(already_done[-1], "rb") as g:
                mcmc.post_warmup_state = pickle.load(g)._last_state

        mcmc.run(
            random.PRNGKey(seed),
            *model_args,
            **model_kwargs,
        )

        samples = mcmc.get_samples()
        with open(f"{checkpoint_dir}/{batch_name}.pkl", "wb") as f:
            pickle.dump((samples), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{checkpoint_dir}/{batch_name}_mcmc.pkl", "wb") as f2:
            pickle.dump((mcmc), f2, protocol=pickle.HIGHEST_PROTOCOL)
        already_done = np.sort(glob.glob(f"{checkpoint_dir}/{model_name}_*_mcmc.pkl"))

    print("Done.")
    print("Samples saved at", f"{checkpoint_dir}/{batch_name}.pkl")
    print("MCMC saved at", f"{checkpoint_dir}/{batch_name}_mcmc.pkl")
    return


def fit_simpler_wykoff_multispecies(
    bark_b0,
    bark_b1,
    bark_b2,
    tree_comp,
    stand_comp,
    pooling,
    loc_random,
    plot_random,
    num_cycles,
    data,
    model_name,
    checkpoint_dir,
    num_warmup=1000,
    num_samples=500,
    num_chains=1,
    chain_method="parallel",
    num_batches=1,
    seed=42,
    progress_bar=True,
    overwrite=False,
):
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["relht", "bal", "ballndbh"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    print(model_name)

    model_args, model_kwargs = prepare_data_for_modeling_multispecies(
        bark_b0,
        bark_b1,
        bark_b2,
        tree_comp,
        stand_comp,
        pooling,
        loc_random,
        plot_random,
        num_cycles,
        data,
    )

    if pooling == "partial":
        config = {f"b{i}z": LocScaleReparam(0) for i in range(14)}
        model = reparam(simpler_wykoff_multispecies_model, config=config)
        dense_mass = [
            ("b0z_decentered", "etree"),  # intercept and dds noise
            ("b1z_decentered", "b2z_decentered"),  # ln(dbh) and dbh**2
            ("b7z_decentered", "b8z_decentered"),  # competition effects
        ]
    else:
        model = simpler_wykoff_multispecies_model
        dense_mass = [
            ("b0z", "etree"),  # intercept and dds noise
            ("b1z", "b2z"),  # ln(dbh) and dbh**2
            ("b7z", "b8z"),  # competition effects
        ]
    nuts_kernel = NUTS(
        model=model,
        dense_mass=dense_mass,
        init_strategy=init_to_median,
    )

    already_done = np.sort(glob.glob(f"{checkpoint_dir}/{model_name}_*_mcmc.pkl"))

    if len(already_done) == 0 or overwrite:
        to_do = np.arange(num_batches)
    else:
        to_do = np.arange(num_batches)[len(already_done) :]

    if len(to_do) == 0:
        print("Already done.")
        print("MCMC saved at", already_done[-1])
        return

    for i in to_do:
        batch_name = model_name + f"_batch{i:02d}"
        print(batch_name)
        mcmc = MCMC(
            nuts_kernel,
            progress_bar=progress_bar,
            num_chains=num_chains,
            chain_method=chain_method,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )

        if i > 0:
            with open(already_done[-1], "rb") as g:
                mcmc.post_warmup_state = pickle.load(g)._last_state

        mcmc.run(
            random.PRNGKey(seed),
            *model_args,
            **model_kwargs,
        )

        samples = mcmc.get_samples()
        with open(f"{checkpoint_dir}/{batch_name}.pkl", "wb") as f:
            pickle.dump((samples), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{checkpoint_dir}/{batch_name}_mcmc.pkl", "wb") as f2:
            pickle.dump((mcmc), f2, protocol=pickle.HIGHEST_PROTOCOL)
        already_done = np.sort(glob.glob(f"{checkpoint_dir}/{model_name}_*_mcmc.pkl"))

    print("Done.")
    print("Samples saved at", f"{checkpoint_dir}/{batch_name}.pkl")
    print("MCMC saved at", f"{checkpoint_dir}/{batch_name}_mcmc.pkl")
    return
