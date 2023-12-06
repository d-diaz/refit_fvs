import numpy as np
import pandas as pd


def fia_for_diameter_growth_modeling(path, filter_spp=None, filter_vars=None):
    """Fetches a pre-processed list of trees from the FIA dataset
    and loads them in a format suitable for diameter growth modeling

    Parameters
    ----------
    path : str
      path to the pre-processed FIA treelist as a CSV file
    filter_spp : list-like
      list of FIA species code that the returned dataset will be filtered for
    filter_vars : list-like
      list of FVS variants that the returned dataset will be filtered for

    Returns
    -------
    data : DataFrame
      a Pandas DataFrame with the relevant data for modeling
    factors : tuple
      a three-tuple containing uniques returned by pd.factorize for
      FVS variants, FVS locations, and FIA plots which can be used
      to cross-walk `VAR_IDX`, `LOC_IDX`, and `PLOT_IDX` from `data`
      to their original values.
    """
    fia = pd.read_csv(path)
    RAW_COVARS = [
        "VARIANT",
        "LOCATION",
        "PLOT_ID",
        "FIA_SPCD",
        "MAICF",
        "SLOPE",
        "ASPECT",
        "ELEV",
        "DBH",
        "CR",
        "CR_NEXT",
        "BAPALARGER",
        "BAPALARGER_NEXT",
        "RELHT",
        "RELHT_NEXT",
        "PTBAPA",
        "PTBAPA_NEXT",
        "PTCCF",
        "PTCCF_NEXT",
    ]
    RAW_OBS = ["DBH_NEXT", "INC5YR", "INC10YR"]
    DATE_COLS = ["MEASYEAR", "MEASMON", "MEASYEAR_NEXT", "MEASMON_NEXT"]

    data = (
        fia.loc[
            (fia.STATUSCD == 1)
            & (fia.STATUSCD_NEXT == 1)
            & (fia.DBH >= 5.0)
            & ((fia.MEASYEAR_NEXT - fia.MEASYEAR) == 10)
        ]
        .dropna(subset=RAW_COVARS)[RAW_COVARS + RAW_OBS + DATE_COLS]
        .copy()
    )
    if filter_spp is not None:
        data = data.loc[data.FIA_SPCD.isin(filter_spp)].copy()

    if filter_vars is not None:
        data = data.loc[data.VARIANT.isin(filter_vars)].copy()

    data["MEAS_INTERVAL"] = data["MEASYEAR_NEXT"] - data["MEASYEAR"]
    data[["CR", "CR_NEXT"]] = data[["CR", "CR_NEXT"]] / 100.0
    data["SLOPE"] = data["SLOPE"] / 100.0
    data["ASPECT"] = np.deg2rad(data["ASPECT"])
    data["ELEV"] = data["ELEV"] / 100.0
    data["INC5YR"] = data["INC5YR"] / 20.0
    data["INC10YR"] = data["INC10YR"] / 20.0
    data["HAS_INC5YR"] = ~data["INC5YR"].isna()
    data["HAS_INC10YR"] = ~data["INC10YR"].isna()

    # remove some trees based on large outlier measurement errors
    # trees where DBH has shrunk more than allowable Measurement Quality Objective error
    data = data.loc[(data.DBH_NEXT - data.DBH) > -0.1 * data.DBH / 20.0]
    # trees with increment msmt that is more than two inches different than DBH change
    data = data.loc[~(abs((data.DBH_NEXT - data.DBH) - data.INC10YR * 2) > 2.0)]

    data["VAR_IDX"], obs_variants = pd.factorize(data["VARIANT"])
    data["LOC_IDX"], obs_locations = pd.factorize(data["LOCATION"])
    data["PLOT_IDX"], obs_plots = pd.factorize(data["PLOT_ID"])
    factors = (obs_variants, obs_locations, obs_plots)

    return data, factors


def interpolate_values(start, stop, num_cycles, start_cycle=0, stop_cycle=-1):
    """Interpolates rows of an array between user-provided values.

    Parameters
    ----------
    start : array
      observed starting values
    stop : array
      observed ending values
    num_cycles : int
      number of steps (columns) for resulting array
    start_cycle : int or array, optional
      cycle in which starting values were observed
    stop_cycle : int or array, optional
      cycle in which stopping values were observed

    Returns
    -------
    arr : array
      array with values for each row of the array interpolated between
      `start` and `stop`
    """
    arr = np.full((len(start), num_cycles), fill_value=np.nan)
    arr[np.arange(len(start)), start_cycle] = start
    arr[np.arange(len(start)), stop_cycle] = stop
    for i in range(len(start)):
        arr[i] = np.interp(
            np.arange(num_cycles),
            np.arange(num_cycles)[~np.isnan(arr[i])],
            arr[i][~np.isnan(arr[i])],
        )

    return arr


def prepare_data_for_modeling(
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
    start_cycle=0,
    stop_cycle=-1,
):
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["relht", "bal", "ballndbh"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    train_data = data.reset_index(drop=True).copy()
    cr = interpolate_values(
        train_data.CR.values,
        train_data.CR_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    bal = interpolate_values(
        train_data.BAPALARGER.values,
        train_data.BAPALARGER_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    relht = interpolate_values(
        train_data.RELHT.values,
        train_data.RELHT_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    bapa = interpolate_values(
        train_data.PTBAPA.values,
        train_data.PTBAPA_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    ccf = interpolate_values(
        train_data.PTCCF.values,
        train_data.PTCCF_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    num_variants = train_data.VAR_IDX.max() + 1
    num_locations = train_data.LOC_IDX.max() + 1
    num_plots = train_data.PLOT_IDX.max() + 1

    data_ = [
        train_data.VAR_IDX.values,
        train_data.LOC_IDX.values,
        train_data.PLOT_IDX.values,
        train_data.MAICF.values,
        train_data.SLOPE.values,
        train_data.ASPECT.values,
        train_data.ELEV.values,
        train_data.DBH.values,
        cr,
        bal,
        relht,
        bapa,
        ccf,
    ]

    model_args = [
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
        data_,
    ]

    model_kwargs = dict(
        dbh_next=train_data.DBH_NEXT.values,
        exist_5yr=train_data.HAS_INC5YR.values,
        obs_5yr=train_data.INC5YR.values,
        exist_10yr=train_data.HAS_INC10YR.values,
        obs_10yr=train_data.INC10YR.values,
    )

    return model_args, model_kwargs


def prepare_data_for_modeling_multispecies(
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
    start_cycle=0,
    stop_cycle=-1,
):
    assert pooling in ["pooled", "unpooled", "partial"]
    assert tree_comp in ["relht", "bal", "ballndbh"]
    assert stand_comp in ["ba", "lnba", "ccf"]

    train_data = data.reset_index(drop=True).copy()
    cr = interpolate_values(
        train_data.CR.values,
        train_data.CR_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    bal = interpolate_values(
        train_data.BAPALARGER.values,
        train_data.BAPALARGER_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    relht = interpolate_values(
        train_data.RELHT.values,
        train_data.RELHT_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    bapa = interpolate_values(
        train_data.PTBAPA.values,
        train_data.PTBAPA_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    ccf = interpolate_values(
        train_data.PTCCF.values,
        train_data.PTCCF_NEXT.values,
        num_cycles,
        start_cycle,
        stop_cycle,
    )
    num_variants = train_data.VAR_IDX.max() + 1
    num_locations = train_data.LOC_IDX.max() + 1
    num_plots = train_data.PLOT_IDX.max() + 1
    num_spp = train_data.SPP_IDX.max() + 1

    data_ = [
        train_data.SPP_IDX.values,
        train_data.VAR_IDX.values,
        train_data.LOC_IDX.values,
        train_data.PLOT_IDX.values,
        train_data.MAICF.values,
        train_data.SLOPE.values,
        train_data.ASPECT.values,
        train_data.ELEV.values,
        train_data.DBH.values,
        cr,
        bal,
        relht,
        bapa,
        ccf,
    ]

    model_args = [
        bark_b0,
        bark_b1,
        bark_b2,
        tree_comp,
        stand_comp,
        pooling,
        loc_random,
        plot_random,
        num_spp,
        num_variants,
        num_locations,
        num_plots,
        num_cycles,
        data_,
    ]

    model_kwargs = dict(
        dbh_next=train_data.DBH_NEXT.values,
        exist_5yr=train_data.HAS_INC5YR.values,
        obs_5yr=train_data.INC5YR.values,
        exist_10yr=train_data.HAS_INC10YR.values,
        obs_10yr=train_data.INC10YR.values,
    )

    return model_args, model_kwargs


def get_folds(path, spcd, n, k, filter_vars=None):
    """
    Returns variant-stratified subsamples of a dataset.

    Parameters
    ----------
    data : DataFrame
      data to be subsampled
    n : int
      sample size, minimum number of observations in each variant
    k : int
      number of non-overlapping folds of the dataset to return
    filter_vars : list-like, optional
      list-like of VARIANT strings the dataset should be limited to

    Returns
    -------
    folds : list
      list of k DataFrames
    """
    data, _ = fia_for_diameter_growth_modeling(path, spcd, filter_vars=filter_vars)
    min_sample = k * n
    var_counts = data.groupby(by="VARIANT")["DBH"].count()
    keep_vars = var_counts.loc[var_counts >= min_sample].index
    keep_data = data.loc[data.VARIANT.isin(keep_vars)].copy()

    keep_data["VAR_IDX"], _ = pd.factorize(keep_data["VARIANT"])

    folds = []
    for _ in range(k):
        fold = keep_data.groupby(by="VARIANT", group_keys=False).apply(
            lambda x: x.sample(n) if x.shape[0] >= n else x
        )
        fold["LOC_IDX"], _ = pd.factorize(fold["LOCATION"])
        fold["PLOT_IDX"], _ = pd.factorize(fold["PLOT_ID"])
        folds.append(fold)
        keep_data = keep_data.drop(fold.index)

    return folds


def get_resamples(path, spcd, n, k, filter_vars=None):
    """
    Returns variant- and location-stratified subsamples of a dataset.

    Parameters
    ----------
    data : DataFrame
      data to be subsampled
    n : int
      sample size, minimum number of observations in each variant
    k : int
      number of groups of samples of the dataset to return
    filter_vars : list-like, optional
      list-like of VARIANT strings the dataset should be limited to

    Returns
    -------
    folds : list
      list of k DataFrames
    """
    data, _ = fia_for_diameter_growth_modeling(
        path, filter_spp=[spcd], filter_vars=filter_vars
    )

    # total number of examples we expect from each variant
    max_samples = data.groupby(by=["VARIANT"])["DBH"].count().apply(lambda x: min(n, x))
    variants = max_samples.index.values
    num_locations = data.groupby(by=["VARIANT"])["LOCATION"].nunique()
    # if all locations had the same number of samples, we could sample each equally
    min_samples = (max_samples / num_locations).astype(int)

    # the number of samples per location in each variant
    loc_samples = data.groupby(by=["VARIANT", "LOCATION"])["DBH"].count()
    # location with most samples in each variant, used to prune extra samples if needed
    max_loc = (
        loc_samples.loc[loc_samples.groupby(level=[0]).idxmax().values]
        .reset_index(level=1)
        .drop("DBH", axis=1)
    )

    # because locations have uneven sampling, we need to
    # determine the minimum sample size per location
    for var in variants:
        min_n = min_samples.loc[var].copy()
        fold_len = (
            data.loc[data.VARIANT == var]
            .groupby("LOCATION")["DBH"]
            .count()
            .apply(lambda x: min(min_n, x))
            .sum()
        )
        while fold_len < max_samples.loc[var]:
            min_n += 1
            fold_len = (
                data.loc[data.VARIANT == var]
                .groupby("LOCATION")["DBH"]
                .count()
                .apply(lambda x: min(min_n, x))
                .sum()
            )
        # update the series indicating minimum samples per location for each variant
        min_samples.loc[var] = min_n

    # build the requested number of folds of with subsampled data
    # we collect a location-stratified sample from each variant
    # then concatenate all the variants into a single fold
    folds = []
    for _ in range(k):
        fold = []
        for var in variants:
            min_n = min_samples.loc[var]
            fold_ = (
                data.loc[data.VARIANT == var]
                .groupby(by="LOCATION", group_keys=False)
                .apply(lambda x: x.sample(min_n) if x.shape[0] >= min_n else x)
            )
            # if we got more samples than we need
            # drop the samples from the most abundant location
            if len(fold_) > max_samples.loc[var]:
                drop_samples = len(fold_) - max_samples.loc[var]
                mask = (fold_.VARIANT == var) & (
                    fold_.LOCATION == max_loc.loc[var, "LOCATION"]
                )
                to_drop = fold_.loc[mask].sample(drop_samples).index
                fold_ = fold_.drop(to_drop)
            fold.append(fold_)

        # concatenate the variants into a single fold
        fold = pd.concat(fold, axis=0)

        # update the location and plot indices to only consider this fold
        fold["LOC_IDX"], _ = pd.factorize(fold["LOCATION"])
        fold["PLOT_IDX"], _ = pd.factorize(fold["PLOT_ID"])
        folds.append(fold)

    return folds
