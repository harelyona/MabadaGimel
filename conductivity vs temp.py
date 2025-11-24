import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as scip
import csv
import re
from decimal import Decimal
import matplotlib.ticker as mticker


def read_three_column_csv(path, delimiter=',', encoding='utf-8', debug=False):
    """
    Robustly read first three numeric columns from a CSV (header + data rows).
    Handles:
      - Standard scientific notation: 1.17E-05, 2.7e-3
      - '×10^' / 'x10^' multiplicative forms
      - Unicode minus signs
      - Decimal commas when no dot is present
    Returns: (headers, col1, col2, col3) as numpy arrays.
    """
    import csv, re
    import numpy as np

    sci_pat = re.compile(
        r'^[\s"]*([+-]?\d+(?:\.\d+)?)([eE][+-]?\d+)[\s"]*$'
    )
    mult_pat = re.compile(
        r'([+-]?\d+(?:[.,]\d+)?)[xX×]\s*10\^?\s*([+-]?\d+)'
    )

    def parse_num(cell: str):
        s = cell.strip().replace('\u2212', '-')  # unicode minus
        if not s:
            raise ValueError("empty")
        # Replace multiplier forms like 3×10^-4, 3x10^-4 -> 3e-4
        s = mult_pat.sub(r'\1e\2', s)

        # Normalize spaced exponents: "1.2E -05" -> "1.2E-05"
        s = re.sub(r'([Ee])\s*([+-]?\d+)', r'\1\2', s)

        # Replace decimal comma only if no dot present (locale formats)
        if ',' in s and '.' not in s:
            s = s.replace(',', '.')

        # Quick path: strict scientific notation matches
        m = sci_pat.match(s)
        if m:
            base, exp = m.groups()
            try:
                return float(base + exp)
            except Exception:
                pass  # fallback below

        # General float attempt
        try:
            return float(s)
        except Exception:
            pass

        # Try Decimal (then cast)
        try:
            return float(Decimal(s))
        except Exception:
            pass

        # Try numpy fromstring
        try:
            v = np.fromstring(s.replace(',', ' '), sep=' ')
            if v.size == 1 and np.isfinite(v[0]):
                return float(v[0])
        except Exception:
            pass

        raise ValueError(f"Could not parse numeric value: {cell!r}")

    # Try pandas first; if user wants pure parser they can disable by removing this block
    try:
        import pandas as pd
        df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)
        df = df.iloc[:, :3].copy()
        for c in df.columns[:3]:
            df[c] = df[c].astype(str).map(parse_num)
        arr = df.to_numpy(dtype=float)
        mask = np.all(np.isfinite(arr), axis=1)
        arr = arr[mask]
        headers = [str(h).strip() for h in df.columns[:3]]
        if debug:
            # Show a few smallest (potential sci-notation) values to verify parsing
            smallest = np.sort(arr[:,1])[:5]
            print("Debug(read_three_column_csv): smallest 5 in col2 =", smallest)
        return headers, arr[:, 0], arr[:, 1], arr[:, 2]
    except Exception:
        if debug:
            print("Debug: pandas failed, falling back to manual CSV parsing")

    headers = ['', '', '']
    col1, col2, col3 = [], [], []
    with open(path, 'r', encoding=encoding, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if not row or all(c.strip() == '' for c in row):
                continue
            if i == 0:
                headers = [c.strip() for c in row[:3]]
                while len(headers) < 3:
                    headers.append('')
                continue
            if len(row) < 3:
                continue
            try:
                a = parse_num(row[0])
                b = parse_num(row[1])
                c = parse_num(row[2])
            except Exception:
                continue
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
                continue
            col1.append(a); col2.append(b); col3.append(c)

    if debug and col2:
        smallest = np.sort(np.array(col2))[:5]
        print("Debug(read_three_column_csv manual): smallest 5 in col2 =", smallest)

    return headers, np.asarray(col1, dtype=float), np.asarray(col2, dtype=float), np.asarray(col3, dtype=float)


path1="v_vs_temp_3_11.csv"
path2="v_vs_temp_clean (4).csv"
path3="Non-Shifted Results Clean.csv"
path4="temp_17_11 (1).csv"

def run1():
    (_, t, v, temp) = read_three_column_csv(path1)
    temp=temp+273.15
    mask = (t > 3000) & (t < 8000) & ~((t > 4400) & (t < 5000)) & ~((t > 6300) & (t < 7100))
    t=t[mask]
    temp=temp[mask]
    v=v[mask]

    # print("max temp", max(temp))
    # plt.plot(t, v * 100)
    # plt.plot(t, temp)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    # ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    # ax.tick_params(axis='x', which='major', length=6)
    # ax.tick_params(axis='x', which='minor', length=3)
    # plt.title("run1")
    # plt.show()

    # plt.plot(t,temp)
    # plt.plot(t,v*200)
    # plt.show()

    plt.loglog(temp,v/fac1,label="run1")
    plt.xlabel("T")
    plt.ylabel("V")

def _model_three_param(T, A, B, C):
    # model: y = 1 / ( 1/(A + B T^{3/2}) + C )
    x = T ** 1.5
    denom = (A + B * x)
    # avoid division by zero in numerator of inner fraction
    denom = np.where(np.isclose(denom, 0.0), np.sign(denom) * 1e-12, denom)
    inner = 1.0 / denom
    outer_denom = inner + C
    # avoid division by zero
    outer_denom = np.where(np.isclose(outer_denom, 0.0), np.sign(outer_denom) * 1e-12, outer_denom)
    return 1.0 / outer_denom

def fit_three_param(T, y):
    """
    Fit y(T) = 1 / ( 1/(A + B T^{3/2}) + C ) with A>0, B>0, C>0.
    Returns: (popt, perr, used_curve_fit)
      - popt: (A,B,C)
      - perr: (sigma_A, sigma_B, sigma_C) when available (sigma_C may be nan if fallback used)
      - used_curve_fit: True if scipy.curve_fit was used, False if fallback grid-search used
    """
    x = T ** 1.5
    # initial linear fit a + b x for initial A,B
    X_lin = np.vstack([np.ones_like(x), x]).T
    coef, residuals, rank, s = np.linalg.lstsq(X_lin, y, rcond=None)
    # force positive initial guesses
    A0 = max(abs(float(coef[0])), 1e-8)
    B0 = max(abs(float(coef[1])), 1e-8)
    C0 = 1e-8

    # try scipy curve_fit first with positive bounds
    try:
        from scipy.optimize import curve_fit
        lower = [1e-12, 1e-12, 1e-12]
        upper = [np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(_model_three_param, T, y, p0=[A0, B0, C0],
                               bounds=(lower, upper), maxfev=20000)
        perr = tuple(np.sqrt(np.diag(pcov))) if pcov is not None else (np.nan, np.nan, np.nan)
        # ensure numeric positivity (tiny numerical rounding)
        popt = (max(popt[0], 1e-12), max(popt[1], 1e-12), max(popt[2], 1e-12))
        return popt, perr, True
    except Exception:
        # fallback: grid-search on strictly positive C and linear solve for positive A,B
        Cmin, Cmax = 1e-8, 0.1
        Cgrid = np.linspace(Cmin, Cmax, 401)
        best = None
        best_ssr = np.inf
        for C in Cgrid:
            denom = 1.0 / y - C
            if np.any(np.isclose(denom, 0.0)) or np.any(~np.isfinite(denom)):
                continue
            z = 1.0 / denom
            try:
                coef2, resid2, rank2, s2 = np.linalg.lstsq(X_lin, z, rcond=None)
            except Exception:
                continue
            A_cand, B_cand = float(coef2[0]), float(coef2[1])
            # require positivity
            if not (A_cand > 0 and B_cand > 0):
                continue
            if resid2.size > 0:
                ssr = float(resid2[0])
            else:
                ssr = float(np.sum((z - (A_cand + B_cand * x)) ** 2))
            if ssr < best_ssr:
                best_ssr = ssr
                best = (A_cand, B_cand, float(C), ssr)

        if best is None:
            # no positive solution found in grid; return small positive defaults
            print("Warning: no strictly positive (A,B,C) found in fallback; returning small positive defaults.")
            return (A0, B0, C0), (np.nan, np.nan, np.nan), False

        A_fit, B_fit, C_fit, _ = best
        # estimate uncertainties for A,B from residuals of z fit
        denom = 1.0 / y - C_fit
        z = 1.0 / denom
        coef_final, resid_final, rank_f, s_f = np.linalg.lstsq(X_lin, z, rcond=None)
        n = x.size
        p = 2
        if resid_final.size > 0 and n > p:
            sigma2 = float(resid_final[0]) / (n - p)
            try:
                covAB = sigma2 * np.linalg.inv(X_lin.T @ X_lin)
                sa, sb = float(np.sqrt(abs(covAB[0, 0]))), float(np.sqrt(abs(covAB[1, 1])))
            except Exception:
                sa, sb = float('nan'), float('nan')
        else:
            sa, sb = float('nan'), float('nan')
        # C uncertainty not estimated in grid fallback
        return (A_fit, B_fit, C_fit), (sa, sb, float('nan')), False

def fit_a_plus_b_T32(T, y):
    """
    Fit y(T) = A + B * T^(3/2) via linear least squares.
    Returns (A, B), (sigma_A, sigma_B)
    Uncertainties are NaN if they cannot be estimated.
    """
    x = T ** 1.5
    X = np.vstack([np.ones_like(x), x]).T
    coef, resid, rank, s = np.linalg.lstsq(X, y, rcond=None)
    A, B = float(coef[0]), float(coef[1])

    n = x.size
    p = 2
    if resid.size > 0 and n > p:
        sigma2 = float(resid[0]) / (n - p)
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            sigmaA = float(np.sqrt(np.abs(cov[0, 0])))
            sigmaB = float(np.sqrt(np.abs(cov[1, 1])))
        except Exception:
            sigmaA, sigmaB = float('nan'), float('nan')
    else:
        sigmaA, sigmaB = float('nan'), float('nan')

    return (A, B), (sigmaA, sigmaB)

def run2():
    (_, t, v, temp) = read_three_column_csv(path2)
    temp=temp+273.15

    mask = (t > 4000) & (t<7100)
    t = t[mask]
    temp = temp[mask]
    v = v[mask]
    log_v = np.log(v)
    log_T=np.log(temp)
    plt.loglog(temp,v/fac2,label="run2")

    # plt.plot(t,v*20000)
    # plt.plot(t,temp)

    # m = (log_v[-1] - log_v[0]) / (log_T[-1] - log_T[0])
    # x_line = np.linspace(log_T[0], log_T[-1], 100)
    # y_line = m * (x_line - log_T[0])
    # plt.loglog(np.exp(x_line), np.exp(y_line), '--', color='red', label=f'avg slope={m:.3f}')
    # print(m)

    # # --- fit A + B T^1.5 and overplot ---
    # T = np.asarray(temp, dtype=float)
    # y = np.asarray(v, dtype=float)
    # valid = np.isfinite(T) & np.isfinite(y)
    # T = T[valid]
    # y = y[valid]
    #
    # if T.size >= 2:
    #     (A_fit, B_fit), (sa, sb) = fit_a_plus_b_T32(T, y)
    #     print(f"Fit y = A + B T^1.5: A={A_fit:.6g} ± {sa:.6g}, B={B_fit:.6g} ± {sb:.6g}")
    #
    #     Tplot = np.linspace(np.min(T), np.max(T), 400)
    #     yfit = A_fit + B_fit * Tplot**1.5
    #     plt.loglog(Tplot, yfit, '-', color='green', label='A + B T^1.5 fit')
    #     plt.legend()
    # else:
    #     print("Not enough valid data points to fit run2")
    #
    # plt.title("run2")
    # plt.show()

def run3():
    (_, t, v, temp) = read_three_column_csv(path3)
    temp=temp+273.15

    mask = (t > 2000) & (temp > 100) & (temp<2.1e2)
    t=t[mask]
    temp=temp[mask]
    v=v[mask]

    #remove jump
    remove_idx = np.arange(246, 266)
    valid_remove = remove_idx[(remove_idx >= 0) & (remove_idx < t.size)]
    if valid_remove.size > 0:
        t = np.delete(t, valid_remove)
        temp = np.delete(temp, valid_remove)
        v = np.delete(v, valid_remove)

    log_v=np.log(v)
    log_T=np.log(temp)
    plt.loglog(temp,v/fac3,label="run3")

    m = (log_v[-1] - log_v[0]) / (log_T[-1] - log_T[0])
    x_line = np.linspace(log_T[0], log_T[-1], 100)
    y_line = m * (x_line - log_T[0])
    # plt.loglog(np.exp(x_line), np.exp(y_line)/fac3, '--', color='red', label=f'avg slope={m:.3f}')

    T = np.asarray(temp, dtype=float)
    y = np.asarray(v, dtype=float)
    valid = np.isfinite(T) & np.isfinite(y)
    T = T[valid]
    y = y[valid]


    # (A_fit, B_fit), (sa, sb) = fit_a_plus_b_T32(T, y)
    # # print(f"Fit y = A + B T^1.5: A={A_fit:.6g} ± {sa:.6g}, B={B_fit:.6g} ± {sb:.6g}")
    #
    # Tplot = np.linspace(np.min(T), np.max(T), 400)
    # yfit = A_fit + B_fit * Tplot**1.5
    # # plt.loglog(Tplot, yfit/fac3, '-', color='green', label='A + B T^1.5 fit')
    # plt.legend()


    plt.title("run3")
    plt.legend()
    plt.show()




def run4():
    (_, t, v, temp) = read_three_column_csv(path4)
    temp=temp+273.15



    mask = (temp>0) & (t>1700) & (t<8300) #(temp>0) & (t>3500)#
    t=t[mask]
    temp=temp[mask]
    v=v[mask]-v_offset4
    remove_idx = []
    # iterate to second-last index so i+1 is valid
    good_data_idx = 0
    for i in range(len(t) - 1):
        if abs(v[i + 1] - v[good_data_idx]) > 0.1:
            remove_idx.append(i + 1)
        else:
            good_data_idx=i+1

    good_data_idx = 0
    for i in range(len(t) - 1):
        if abs(temp[i + 1] - temp[good_data_idx]) > 30:
            remove_idx.append(i + 1)
        else:
            good_data_idx = i + 1

    if remove_idx:
        remove_idx = np.unique(remove_idx).astype(int)
        valid_remove = remove_idx[(remove_idx >= 0) & (remove_idx < t.size)]
        if valid_remove.size > 0:
            t = np.delete(t, valid_remove)
            temp = np.delete(temp, valid_remove)
            v = np.delete(v, valid_remove)

    #remove jump
    # valid_remove = remove_idx[(remove_idx >= 0) & (remove_idx < t.size)]
    # if valid_remove.size > 0:
    #     t = np.delete(t, valid_remove)
    #     temp = np.delete(temp, valid_remove)
    #     v = np.delete(v, valid_remove)


    #v = v / v[0]
    # m = (log_v[-1] - log_v[0]) / (log_T[-1] - log_T[0])
    # x_line = np.linspace(log_T[0], log_T[-1], 100)
    # y_line = m * (x_line - log_T[0])
    #plt.loglog(np.exp(x_line), np.exp(y_line), '--', color='red', label=f'avg slope={m:.3f}')


    T = np.asarray(temp, dtype=float)
    y = np.asarray(v, dtype=float)
    valid = np.isfinite(T) & np.isfinite(y)
    T = T[valid]
    y = y[valid]

    # (A_fit, B_fit), (sa, sb) = fit_a_plus_b_T32(T, y)
    # print(f"Fit y = A + B T^1.5: A={A_fit:.6g} ± {sa:.6g}, B={B_fit:.6g} ± {sb:.6g}")



    # plt.plot(t,temp/15,marker='.',linestyle='None')
    # plt.plot(t,v,marker='.',linestyle='None')
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=16))  # increase major tick count
    # ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))  # add minor ticks between majors
    # ax.tick_params(axis='x', which='major', length=6)
    # ax.tick_params(axis='x', which='minor', length=3)
    # plt.show()

    plt.loglog(temp,v/fac4,label="run4")
    plt.xlabel("T")
    plt.ylabel("V")

#    plt.show()

    # Tplot = np.linspace(np.min(T), np.max(T), 400)
    # yfit = A_fit + B_fit * Tplot**1.5
    # plt.loglog(Tplot, yfit, '-', color='green', label='A + B T^1.5 fit')
    # plt.legend()
    # plt.title("run4")
    # plt.show()

def show_data(path):
    (_, t, v, temp) = read_three_column_csv(path)
    temp = temp + 273.15
    matplotlib.use('TkAgg')

    # plt.plot(t,temp)
    # plt.show()
    plt.plot(t,v)
    plt.show()


#set these to the current in microamps, in each run
fac1=500
fac2=0.65
fac3=200#170
fac4=1000


v_offset4=0#3.374

# show_data(path1) #no jumps
# show_data(path2) #really low voltages for some reason
# show_data(path3) #no jumps
# show_data(path4) #jumps by 3.374V
matplotlib.use('module://backend_interagg')
run1()
#run2()
run4()
run3()
