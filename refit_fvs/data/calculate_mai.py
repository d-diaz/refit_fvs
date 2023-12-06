import numpy as np


def eq1(si):
    """
    Douglas-fir and grand fir in western Oregon (after 1984) except for Jackson and
    Josephine Counties, western Washington except in silver fir zone, California except
    in mixed-conifer stands (McArdle and others 1961).
    """
    return np.piecewise(
        si,
        [si < 75, si >= 75, si >= 130],
        [
            lambda x: -60 + 1.71 * x,
            lambda x: -81.3 + 2.02 * x,
            lambda x: 22.9 + 1.21 * x,
        ],
    )


def eq2(si):
    """
    Douglas-fir in Jackson and Josephine Counties, Oregon (McArdle and others 1961).
    """
    return 1.8 * si - 57.12


def eq3(si):
    """
    Grand fir and white fir in Jackson and Josephine Counties, Oregon (Cochran 1979a)
    """
    return 1.9407 * si - 34


def eq4(si):
    """
    Western hemlock and Sitka spruce in western Oregon, eastern Oregon, western
    Washington, eastern Washington, and California (Barnes 1962).
    """
    return 2.628 * si - 49.8


def eq5(si):
    """
    Redwood in western Oregon and California (Lindquist and Palley 1963).
    """
    return np.exp(0.2995 * np.sqrt(si) + 2.404)


def eq6(si):
    """
    Noble fir, Shasta red fir in Oregon, Pacific silver fir, subalpine fir, mountain
    hemlock in western Oregon, eastern Oregon, western Washington, eastern Washington,
    and California (Barnes 1962).
    """
    return 1.6 * si - 50


def eq7(si):
    """
    Ponderosa pine, Jeffrey pine, Coulter pine, and Bishop pine, in western Oregon,
    eastern Oregon, western Washington, eastern Washington, and California (Meyer 1961).
    """
    return np.exp(0.702695 * si**0.42 - 0.51367)


def eq8(si):
    """
    Douglas-fir in eastern Oregon and eastern Washington (Cochran 1979a)
    """
    return 0.00473 * si**2.04


def eq9(si):
    """
    White fir and grand fir in eastern Oregon and eastern Washington (Cochran 1979a).
    """
    return np.exp(8.24227 - 23.53735 * si ** (-0.4))


def eq10(si):
    """
    Lodgepole pine and western white pine in eastern Oregon, western Washington,
    and California (Dahms 1964).
    """
    return 0.8594 * si - 22.32


def eq11(si):
    """
    Lodgepole pine in eastern Washington (Brickell 1970).
    """
    return 0.0122 * si**2 - 0.2026 * si + 7.4


def eq12(si):
    """
    Western larch in eastern Oregon (Cochran 1985).
    """
    return np.exp(0.05 - 72.1299 / (63.8 - 0.066 * si) + 1.4 * np.log(si - 20))


def eq13(si):
    """
    Western larch in western Washington and eastern Washington (Brickell 1970).
    """
    return -126.05 + 2.7974081 * si + 1919.3157 / si


def eq14(si):
    """
    Engelmann spruce in eastern Oregon, western Washington, and eastern Washington.
    """
    return 1.92 * si - 18.4


def eq15(si):
    """
    Douglas-fir in silver fir zone of western Washington (McArdle and others 1961).
    """
    return 1.166 * si - 50


def eq16(si):
    """
    Western redcedar in western Oregon, western Washington, and California
    (Barnes 1962).
    """
    return 2.628 * si - 49.8


def eq17(si):
    """
    Western white pine in eastern Washington (Brickell 1970).
    """
    return 14.849891 + 1.7311563 * si


def eq18(si):
    """
    Mixed conifer in California (Dunning and Reineke 1933)
    """
    return np.exp(0.578265 * si**0.4 + 1.8108)


def eq19(si):
    """
    Red fir, Shasta red fir in California, white fir in California (Schumacher 1928).
    """
    return 48.278 + 0.23638 * si**1.6


def eq20(si):
    """
    All hardwoods (Worthington and others 1960).
    """
    return 1.7102 * si - 53.1279


def calc_mai(si, sisp, sibase, vol_loc_grp, stockability):
    """
    Calculates potential mean annual increment in cubic feet per year
    following

    Hanson, E.J., Azuma, D.L., Hiserote, B.A., 2002. Site index equations and mean
    annual increment equations for Pacific Northwest Research Station Forest Inventory
    and Analysis Inventories, 1985-2001. PNW-RN-533. USDA Forest Service, PNW Research
    Station, Portland, OR. 26pp. https://www.fs.fed.us/pnw/pubs/pnw_rn533.pdf

    Parameters
    ----------
    si : array-like
      site index (feet)
    sisp : array-like
      site species (FIA species codes)
    sibase : array-like
      base age of site index equation
    vol_loc_grp : array-like
      volume location group defined by FIA, must be one of
          'S26LCA': 'California other than mixed conifer type',
          'S26LCAMIX': 'California mixed conifer type',
          'S26LEOR': 'Eastern Oregon',
          'S26LEWA': 'Eastern Washington',
          'S26LORJJ': 'Oregon, Jackson and Josephine Counties',
          'S26LWACF': 'Washington Silver Fir Zone',
          'S26LWOR': 'Western Oregon',
          'S26LWWA': 'Western Washington'
    stockability : array-like
      plant stockability factor, a value from 0 to 1.0.

    Returns
    -------
    mai : array
      potential mean annual increment. Unrecognized species, regions,
      and mai estimates <= 0 are return as NaNs.
    """

    mai = np.zeros_like(si)
    stk = np.where(np.isnan(stockability), 1.0, stockability)

    mask = np.isin(sisp, [202, 17]) & (vol_loc_grp == "S26LWOR") & (sibase == 50)
    mai[mask] = eq1(si[mask]) * stk[mask]

    mask = np.isin(sisp, [202]) & (vol_loc_grp == "S26LORJJ") & (sibase == 50)
    mai[mask] = eq2(si[mask]) * stk[mask]

    mask = np.isin(sisp, [15, 17]) & (vol_loc_grp == "S26LORJJ") & (sibase == 50)
    mai[mask] = eq3(si[mask]) * stk[mask]

    mask = np.isin(sisp, [263, 98]) & (sibase == 50)
    mai[mask] = eq4(si[mask]) * stk[mask]

    mask = np.isin(sisp, [211]) & (sibase == 50)
    mai[mask] = eq5(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [22, 21])
        & (np.isin(vol_loc_grp, ["S26LEOR", "S26LORJJ", "S26LWOR"]))
        & (sibase == 100)
    )
    mai[mask] = eq6(si[mask]) * stk[mask]

    mask = np.isin(sisp, [11, 19, 264]) & (sibase == 100)
    mai[mask] = eq6(si[mask]) * stk[mask]

    mask = np.isin(sisp, [122, 116, 109, 120]) & (sibase == 100)
    mai[mask] = eq7(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [202])
        & (np.isin(vol_loc_grp, ["S26LEOR", "S26LEWA"]))
        & (sibase == 100)
    )
    mai[mask] = eq8(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [15, 17])
        & (np.isin(vol_loc_grp, ["S26LEOR", "S26LEWA"]))
        & (sibase == 100)
    )
    mai[mask] = eq9(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [108, 119])
        & (
            np.isin(
                vol_loc_grp, ["S26LCA", "S26LCAMIX", "S26LEOR", "S26LWACF", "S26LWWA"]
            )
        )
        & (sibase == 100)
    )
    mai[mask] = eq10(si[mask]) * stk[mask]

    mask = np.isin(sisp, [108]) & (np.isin(vol_loc_grp, ["S26LEWA"])) & (sibase == 100)
    mai[mask] = eq11(si[mask]) * stk[mask]

    mask = np.isin(sisp, [73]) & (np.isin(vol_loc_grp, ["S26LEOR"])) & (sibase == 50)
    mai[mask] = eq12(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [73])
        & (np.isin(vol_loc_grp, ["S26LWWA", "S26LEWA"]))
        & (sibase == 50)
    )
    mai[mask] = eq13(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [93])
        & (np.isin(vol_loc_grp, ["S26LEOR", "S26LEWA", "S26LWWA"]))
        & (sibase == 50)
    )
    mai[mask] = eq14(si[mask]) * stk[mask]

    mask = np.isin(sisp, [202]) & (np.isin(vol_loc_grp, ["S26LWACF"])) & (sibase == 100)
    mai[mask] = eq15(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [242])
        & (
            np.isin(
                vol_loc_grp, ["S26LWWA", "S26LWACF", "S26LWOR", "S26LCA", "S26LCAMIX"]
            )
        )
        & (sibase == 50)
    )
    mai[mask] = eq16(si[mask]) * stk[mask]

    mask = np.isin(sisp, [119]) & (np.isin(vol_loc_grp, ["S26LEWA"])) & (sibase == 100)
    mai[mask] = eq17(si[mask]) * stk[mask]

    mask = (
        np.isin(sisp, [20, 21, 15])
        & (np.isin(vol_loc_grp, ["S26LCA", "S26LCAMIX"]))
        & (sibase == 50)
    )
    mai[mask] = eq19(si[mask]) * stk[mask]

    mask = np.isin(sisp, [542, 747, 351]) & (sibase == 50)
    mai[mask] = eq20(si[mask]) * stk[mask]

    return np.where(mai <= 0, np.nan, mai)
