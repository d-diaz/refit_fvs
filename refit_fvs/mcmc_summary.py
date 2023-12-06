import arviz as az
import glob
import numpy as np
import os
import pandas as pd
import pickle
import warnings
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from jax.random import PRNGKey
from refit_fvs.models.diameter_growth import wykoff_forward, simpler_wykoff_forward


from refit_fvs.data.load import (
    fia_for_diameter_growth_modeling,
    prepare_data_for_modeling,
)


def count_models(path, verbose=True):
    """
    Reports summary information about the number and type of models fit

    Parameters
    ----------
    path : str
      path to directory that contains pickled mcmc results

    Returns
    -------
    mcmc_path_dict : dict
      dictionary with paths to different types of pickled mcmcs found
    """
    mcmcs = np.sort(glob.glob(f"{path}/*mcmc.pkl"))
    mcmcs = [os.path.abspath(f) for f in mcmcs]
    unpooled_mcmcs = [m for m in mcmcs if "_unpooled" in m]
    pooled_mcmcs = [m for m in mcmcs if "_pooled" in m]
    partial_mcmcs = [m for m in mcmcs if "_partial" in m]
    full_mcmcs = [m for m in mcmcs if "simplerwykoff_" not in m]
    simpler_mcmcs = [m for m in mcmcs if "simplerwykoff_" in m]

    if verbose:
        print("total:", len(mcmcs))
        print("unpooled: ", len(unpooled_mcmcs))
        print("pooled:", len(pooled_mcmcs))
        print("partial:", len(partial_mcmcs))
        print("full:", len(full_mcmcs))
        print("simpler:", len(simpler_mcmcs))

    mcmc_path_dict = {
        "all": mcmcs,
        "unpooled": unpooled_mcmcs,
        "pooled": pooled_mcmcs,
        "partial": partial_mcmcs,
        "full": full_mcmcs,
        "simpler": simpler_mcmcs,
    }

    return mcmc_path_dict


def make_compare_dict(path_to_models, filter_models, verbose=True):
    """
    Generates a dictionary with Arviz-loaded models

    Parameters
    ----------
    path_to_models : str
      path to directory containing pickled MCMC models

    Returns
    --------
    compare_dict : dict
      dictionary containing Arviz-loaded models
    """
    to_glob = os.path.join(path_to_models, f"{filter_models}*_mcmc.pkl")
    paths_to_models = glob.glob(to_glob)
    if verbose:
        print(f"Found {len(paths_to_models):.0f} models")
    compare_dict = {}
    for path in paths_to_models:
        basename = os.path.basename(os.path.abspath(path))
        model = basename.split("_")[0]
        pooling = basename.split("_")[1]
        comp_vars = basename.split("_")[3]

        with open(path, "rb") as f:
            name = f"{model}_{pooling}_{comp_vars}"
            try:
                compare_dict[name] = az.from_numpyro(pickle.load(f))
                compare_dict[name].attrs["path"] = path
                spcd = os.path.basename(os.path.split(path)[0])
                compare_dict[name].attrs["spcd"] = spcd
                if verbose:
                    print(name)
            except Exception as e:
                print("Failed on", name)
                print(e)

    return compare_dict


def make_compare_df(
    fia_spcd,
    compare_dict,
    var_name="meas_dbh_next",
    bark_var="PN",
    num_cycles=2,
    loc_random=True,
    plot_random=True,
    verbose=True,
):
    """
    Makes a DataFrame comparing models using several performance measures.

    Parameters
    ----------
    fia_spcd : int
      the FIA tree species code
    compare_dict : dict
      dictionary of Arviz InferenceObjects to compare
    var_name : str
      name of the observed variable model performance should be based on
    bark_var : str
      abbrevation for FVS regional variant that should be used to specify
      bark ratio coefficients. Valid options are currently:
      "PN", "WC", "NC", or "CA".

    """
    assert bark_var in ["PN", "WC", "NC", "CA"]

    if verbose:
        print("Starting LOO-PSIS", end="... ")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        compare_df = az.compare(compare_dict, var_name=var_name)
    if verbose:
        print("done.")

    # add more performance metrics
    compare_df["MAE_MEAN"] = np.nan
    compare_df["MAE_SD"] = np.nan
    compare_df["RMSE_MEAN"] = np.nan
    compare_df["RMSE_SD"] = np.nan
    compare_df["BIAS_MEAN"] = np.nan
    compare_df["BIAS_SD"] = np.nan
    compare_df["ETREE_MEAN"] = np.nan
    compare_df["ETREE_SD"] = np.nan

    data, factors = fia_for_diameter_growth_modeling(
        path="../../data/interim/FIA_remeasured_trees_for_training.csv",
        filter_spp=[fia_spcd],
    )
    obs_variants, _, _ = factors

    bark = pd.read_csv("../../data/raw/fvs_barkratio_coefs.csv").set_index(
        ["FIA_SPCD", "FVS_VARIANT"]
    )
    bark_b0, bark_b1, bark_b2 = bark.loc[fia_spcd, bark_var][
        ["BARK_B0", "BARK_B1", "BARK_B2"]
    ]

    regional_comparisons = pd.DataFrame(
        index=pd.MultiIndex.from_product([compare_df.index, obs_variants]),
        columns=[
            "N",
            "MAE_MEAN",
            "MAE_SD",
            "MAE_LO",
            "MAE_HI",
            "RMSE_MEAN",
            "RMSE_SD",
            "RMSE_LO",
            "RMSE_HI",
            "BIAS_MEAN",
            "BIAS_SD",
            "BIAS_LO",
            "BIAS_HI",
        ],
    )

    if verbose:
        print("Starting to add additional performance metrics... ")
    for key in compare_dict:
        if verbose:
            print(key, end="... ")
        model = key.split("_")[0]
        pooling = key.split("_")[1]
        tree_comp = key.split("_")[2].split("-")[0]
        stand_comp = key.split("_")[2].split("-")[1]
        # spcd = compare_dict[key].attrs["spcd"]

        path = compare_dict[key].attrs["path"]
        with open(path, "rb") as f:
            mcmc = pickle.load(f)

        samples = mcmc.get_samples(group_by_chain=False)

        if model == "wykoff":
            use_vars = [f"b{i}" for i in range(14)] + ["etree", "eloc_", "eplot_"]
        elif model == "simplerwykoff":
            use_vars = [f"b{i}" for i in range(9)] + ["etree", "eloc_", "eplot_"]
        use_samples = {f"{v}": samples[v] for v in use_vars}

        if model == "wykoff":
            pred = Predictive(wykoff_forward, posterior_samples=use_samples)
        elif model == "simplerwykoff":
            pred = Predictive(simpler_wykoff_forward, posterior_samples=use_samples)

        args, kwargs = prepare_data_for_modeling(
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

        preds = pred(PRNGKey(0), *args, **kwargs)

        pred_dg = preds["real_dbh"][:, :, 2] - preds["real_dbh"][:, :, 0]
        obs_dg = data["DBH_NEXT"].values - data["DBH"].values
        resid = pred_dg - obs_dg
        compare_df.loc[key, "BIAS_MEAN"] = resid.mean(axis=1).mean()
        compare_df.loc[key, "BIAS_SD"] = resid.mean(axis=1).std()
        low, high = hpdi(resid.mean(axis=1), prob=0.9)
        compare_df.loc[key, "BIAS_LO"] = low
        compare_df.loc[key, "BIAS_HI"] = high

        mae = (abs(resid)).mean(axis=1)
        compare_df.loc[key, "MAE_MEAN"] = mae.mean().item()
        compare_df.loc[key, "MAE_SD"] = mae.std().item()
        low, high = hpdi(mae, prob=0.9)
        compare_df.loc[key, "MAE_LO"] = low
        compare_df.loc[key, "MAE_HI"] = high

        sq_err = resid**2
        mse = sq_err.mean(axis=1)
        rmse = np.sqrt(mse)
        compare_df.loc[key, "RMSE_MEAN"] = rmse.mean()
        compare_df.loc[key, "RMSE_SD"] = rmse.std()
        low, high = hpdi(rmse, prob=0.9)
        compare_df.loc[key, "RMSE_LO"] = low
        compare_df.loc[key, "RMSE_HI"] = high

        r2_mean, r2_sd = az.r2_score(obs_dg, pred_dg)
        compare_df.loc[key, "R2_MEAN"] = r2_mean
        compare_df.loc[key, "R2_SD"] = r2_sd

        if verbose:
            print("Regional calculations", end="... ")

        for reg in obs_variants:
            mask = (data.VARIANT == reg).values
            pred_dg = preds["real_dbh"][:, :, 2] - preds["real_dbh"][:, :, 0]
            obs_dg = data["DBH_NEXT"].values - data["DBH"].values
            resid = pred_dg[:, mask] - obs_dg[mask]
            regional_comparisons.loc[(key, reg), "BIAS_MEAN"] = resid.mean(
                axis=1
            ).mean()
            regional_comparisons.loc[(key, reg), "BIAS_SD"] = resid.mean(axis=1).std()
            mae = (abs(resid)).mean(axis=1)
            regional_comparisons.loc[(key, reg), "MAE_MEAN"] = mae.mean().item()
            regional_comparisons.loc[(key, reg), "MAE_SD"] = mae.std().item()
            sq_err = resid**2
            mse = sq_err.mean(axis=1)
            rmse = np.sqrt(mse)
            regional_comparisons.loc[(key, reg), "RMSE_MEAN"] = rmse.mean()
            regional_comparisons.loc[(key, reg), "RMSE_SD"] = rmse.std()
            r2_mean, r2_sd = az.r2_score(obs_dg[mask], pred_dg[:, mask])
            regional_comparisons.loc[(key, reg), "R2_MEAN"] = r2_mean
            regional_comparisons.loc[(key, reg), "R2_SD"] = r2_sd
            regional_comparisons.loc[(key, reg), "N"] = mask.sum()
            if verbose:
                print(reg, end="... ")
        if verbose:
            print("done.")

    return compare_df, regional_comparisons


def benchmark_regions(
    path_to_data,
    path_to_bark_coefs,
    fia_spcd,
    bark_var,
    benchmark_model,
    compare_dict,
    num_cycles=2,
    loc_random=True,
    plot_random=True,
):
    """
    Makes a DataFrame comparing change in performance of a benchmark model against
    other models.

    Parameters
    ----------
    path_to_data : str
      path to CSV file with data used to score model performance
    path_to_bark_coefs : str
      path to CSV file containing bark ratio coefficients
    fia_spcd : int
      the FIA tree species code
    bark_var : str
      variant to get bark ratios from, one of ["PN", "WC", "NC", "CA"]
    benchmark_model : str
      the model to use as the benchmark that all others will be compared against
    compare_dict : dict
      dictionary of Arviz InferenceObjects to compare
    """
    assert bark_var in ["PN", "WC", "NC", "CA"]
    assert benchmark_model in compare_dict

    data, factors = fia_for_diameter_growth_modeling(
        path=path_to_data, filter_spp=[fia_spcd]
    )
    obs_variants, _, _ = factors
    dg_obs = (data["DBH_NEXT"] - data["DBH"]).values

    bark = pd.read_csv(path_to_bark_coefs).set_index(["FIA_SPCD", "FVS_VARIANT"])
    bark_b0, bark_b1, bark_b2 = bark.loc[fia_spcd, bark_var][
        ["BARK_B0", "BARK_B1", "BARK_B2"]
    ]

    bench_path = compare_dict[benchmark_model].attrs["path"]
    bench_model_type = benchmark_model.split("_")[0]
    bench_pooling = benchmark_model.split("_")[1]
    bench_tree_comp = benchmark_model.split("_")[2].split("-")[0]
    bench_stand_comp = benchmark_model.split("_")[2].split("-")[1]

    with open(bench_path, "rb") as f:
        bench_mcmc = pickle.load(f)
        bench_samples = bench_mcmc.get_samples(group_by_chain=False)

    if bench_model_type == "simplerwykoff":
        bench_vars = [f"b{i}" for i in range(9)] + ["etree", "eloc_", "eplot_"]
        bench_model = simpler_wykoff_forward
    elif bench_model_type == "wykoff":
        bench_vars = [f"b{i}" for i in range(14)] + ["etree", "eloc_", "eplot_"]
        bench_model = wykoff_forward

    bench_use = {f"{v}": bench_samples[v] for v in bench_vars}

    bench_pred = Predictive(bench_model, posterior_samples=bench_use)

    bench_args, bench_kwargs = prepare_data_for_modeling(
        bark_b0,
        bark_b1,
        bark_b2,
        bench_tree_comp,
        bench_stand_comp,
        bench_pooling,
        loc_random,
        plot_random,
        num_cycles,
        data,
    )

    bench_kwargs.update(
        dict(
            dbh_next=None,
            exist_5yr=np.full(len(data), False),
            exist_10yr=np.full(len(data), False),
        )
    )

    bench_preds = bench_pred(PRNGKey(0), *bench_args, **bench_kwargs)

    dg_bench = bench_preds["real_dbh"][:, :, 2] - bench_preds["real_dbh"][:, :, 0]

    compare_models = [x for x in compare_dict if x != benchmark_model]

    hdi_rope = pd.DataFrame(
        index=["WORSE", "EQUIVALENT", "BETTER", "INCONCLUSIVE", "TOTAL"],
        columns=[obs_variants],
    ).fillna(0)

    detail = {m: {v: None for v in obs_variants} for m in compare_dict.keys()}
    ropes = {v: None for v in obs_variants}

    for name in compare_models:
        path = compare_dict[name].attrs["path"]
        model_type = name.split("_")[0]
        pooling = name.split("_")[1]
        tree_comp = name.split("_")[2].split("-")[0]
        stand_comp = name.split("_")[2].split("-")[1]

        with open(path, "rb") as g:
            mcmc = pickle.load(g)

        samples = mcmc.get_samples(group_by_chain=False)
        if model_type == "simplerwykoff":
            mod_vars = [f"b{i}" for i in range(9)] + ["etree", "eloc_", "eplot_"]
            model = simpler_wykoff_forward
        elif model_type == "wykoff":
            mod_vars = [f"b{i}" for i in range(14)] + ["etree", "eloc_", "eplot_"]
            model = wykoff_forward

        use_vars = {f"{v}": samples[v] for v in mod_vars}
        pred = Predictive(
            model,
            posterior_samples=use_vars,
        )

        args, kwargs = prepare_data_for_modeling(
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
        kwargs.update(
            dict(
                dbh_next=None,
                exist_5yr=np.full(len(data), False),
                exist_10yr=np.full(len(data), False),
            )
        )

        preds = pred(PRNGKey(0), *args, **kwargs)

        dg_pred = preds["real_dbh"][:, :, 2] - preds["real_dbh"][:, :, 0]

        for variant in obs_variants:
            mask = (data.VARIANT == variant).values
            rope = dg_obs[mask].mean() * 0.01
            ropes[variant] = rope
            resid = dg_pred[:, mask] - dg_obs[mask]
            mae = (abs(resid)).mean(axis=1)
            resid_b = dg_bench[:, mask] - dg_obs[mask]
            mae_b = (abs(resid_b)).mean(axis=1)
            diff_mae = mae_b - mae
            low, high = hpdi(diff_mae, prob=0.9)
            detail[name][variant] = mae
            detail[benchmark_model][variant] = mae_b

            if low > rope:
                hdi_rope.loc["BETTER", variant] += 1
            elif high < -rope:
                hdi_rope.loc["WORSE", variant] += 1
            elif (high < rope) and (low > -rope):
                hdi_rope.loc["EQUIVALENT", variant] += 1
            else:
                hdi_rope.loc["INCONCLUSIVE", variant] += 1

            hdi_rope.loc["TOTAL", variant] += 1

    return hdi_rope, detail, ropes


def regional_mae(
    path_to_data,
    path_to_bark_coefs,
    fia_spcd,
    bark_var,
    path_to_model,
    num_cycles=2,
    loc_random=True,
    plot_random=True,
):
    """
    ... TO DO ...

    Parameters
    ----------
    path_to_data : str
      path to CSV file with data used to score model performance
    path_to_bark_coefs : str
      path to CSV file containing bark ratio coefficients
    fia_spcd : int
      the FIA tree species code
    bark_var : str
      variant to get bark ratios from, one of ["PN", "WC", "NC", "CA"]
    benchmark_model : str
      the model to use as the benchmark that all others will be compared against
    compare_dict : dict
      dictionary of Arviz InferenceObjects to compare
    """
    assert bark_var in ["PN", "WC", "NC", "CA"]

    data, factors = fia_for_diameter_growth_modeling(
        path=path_to_data, filter_spp=[fia_spcd]
    )
    obs_variants, _, _ = factors
    dg_obs = (data["DBH_NEXT"] - data["DBH"]).values

    bark = pd.read_csv(path_to_bark_coefs).set_index(["FIA_SPCD", "FVS_VARIANT"])
    bark_b0, bark_b1, bark_b2 = bark.loc[fia_spcd, bark_var][
        ["BARK_B0", "BARK_B1", "BARK_B2"]
    ]

    benchmark_model = os.path.basename(path_to_model)
    bench_model_type = benchmark_model.split("_")[0]
    bench_pooling = benchmark_model.split("_")[1]
    bench_tree_comp = benchmark_model.split("_")[3].split("-")[0]
    bench_stand_comp = benchmark_model.split("_")[3].split("-")[1]

    with open(path_to_model, "rb") as f:
        bench_mcmc = pickle.load(f)
        bench_samples = bench_mcmc.get_samples(group_by_chain=False)

    if bench_model_type == "simplerwykoff":
        bench_vars = [f"b{i}" for i in range(9)] + ["etree", "eloc_", "eplot_"]
        bench_model = simpler_wykoff_forward
    elif bench_model_type == "wykoff":
        bench_vars = [f"b{i}" for i in range(14)] + ["etree", "eloc_", "eplot_"]
        bench_model = wykoff_forward

    bench_use = {f"{v}": bench_samples[v] for v in bench_vars}

    bench_pred = Predictive(bench_model, posterior_samples=bench_use)

    bench_args, bench_kwargs = prepare_data_for_modeling(
        bark_b0,
        bark_b1,
        bark_b2,
        bench_tree_comp,
        bench_stand_comp,
        bench_pooling,
        loc_random,
        plot_random,
        num_cycles,
        data,
    )

    bench_kwargs.update(
        dict(
            dbh_next=None,
            exist_5yr=np.full(len(data), False),
            exist_10yr=np.full(len(data), False),
        )
    )

    bench_preds = bench_pred(PRNGKey(0), *bench_args, **bench_kwargs)

    dg_bench = bench_preds["real_dbh"][:, :, 2] - bench_preds["real_dbh"][:, :, 0]
    ropes = {v: None for v in obs_variants}
    maes = {v: None for v in obs_variants}

    for variant in obs_variants:
        mask = (data.VARIANT == variant).values
        rope = dg_obs[mask].mean() * 0.01
        ropes[variant] = rope
        resid = dg_bench[:, mask] - dg_obs[mask]
        mae = (abs(resid)).mean(axis=1)
        maes[variant] = mae

    return maes, ropes
