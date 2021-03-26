from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spm1d
import analysis


# set serif fonts for matplotlib
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family'] = 'serif'

try:
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif'] = cmfont.get_name()
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['axes.unicode_minus'] = False  # so the minus sign '-' displays correctly in plots
except FileNotFoundError as error:
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of titles

# START COLORS
base_color = "#000000"  # black
pot_color = "#dd502d"  # orange
atroph_color = "#3997bf"  # blue
base_alpha = 0.20
active_alpha = 0.75

tline_color = "#000000"  # black
tpotfill_color = "#7e3728"  # light orange
tatrophfill_color = "#244d90"  # light blue
# END COLORS

# to accomodate different colors in "potentiated" and "atrophied" (injured) modes
BASE_POT = "BASE_POT"
BASE_ATRO = "BASE_INJ"


def plot_test_results(t, ti, baseline_data, active_data, figure_output_path, time_offset=0, mode=BASE_POT, mode_name="Potentiated", show_plot=True, save_figures=True):
    """
    Plots the baseline and active data's mean and standard deviation clouds on axis one
    Plots the SPM t-test between the baseline and active data on axis two

    :param t: An SPM T object TODO what this is
    :param ti: An SPM TI inference object
    :param active_data: 2D numpy array containing "active" measurement data---either potentiated or atrophied
    :param baseline_data: 2D numpy array containing baseline measurement data
    :param figure_output_path: path at which to save output plot
    :param time_offset: integer [milliseconds] to correct for potentially changing SPM start time
    :param mode_name: either BASE_POT or BASE_ATRO (to accomodate different colors in "potentiated" and "atrophied" (injured) modes)
    :param mode_name: either "Potentiated" or "Atrophied"
    :param show_plot: whether or not to show matplotlib plot. Generally turned off for automated processes.
    :param save_figures: whether or not to automatically save figure before plotting.
    """
    num_points = np.shape(active_data)[0]  # could also use base_data
    time = np.linspace(0, num_points - 1, num_points) + time_offset  # include time offset

    active_mean = np.mean(active_data, axis=1)
    base_mean = np.mean(baseline_data, axis=1)
    pot_sd = np.std(active_data, ddof=1, axis=1)  # note SD is lessened somewhat for reasonable plot scale
    base_sd = np.std(baseline_data, ddof=1, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # plot TMG measurement
    ax = axes[0]
    remove_spines(ax)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Position [mm]")

    ax.plot(time, base_mean, color=base_color, linewidth=2.5, label="Baseline", zorder=4)  # plot in front
    ax.plot(time, active_mean, color=get_active_color(mode), linewidth=2.5, label=mode_name, zorder=3)
    ax.fill_between(time, active_mean - pot_sd, active_mean + pot_sd, color=get_active_color(mode), alpha=active_alpha, zorder=2)  # standard deviation clouds
    ax.fill_between(time, base_mean - base_sd, base_mean + base_sd, color=base_color, alpha=base_alpha, zorder=1)  # standard deviation clouds

    ax.axhline(y=0, color='k', linestyle=':')  # dashed line at y = 0
    ax.legend()

    # plot SPM results:
    ax = axes[1]
    remove_spines(ax)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("SPM $t$ Statistic", labelpad=-0.1)

    ax.plot(time, t.z, color=tline_color)  # plot t-curve
    ax.axhline(y=0, color='#000000', linestyle=':')  # dashed line at y = 0
    ax.axhline(y=ti.zstar, color='#000000', linestyle='--')  # dashed line at t threshold
    ax.text(73, ti.zstar + 0.4, "$\\alpha = {:.2f}$\n$t^* = {:.2f}$".format(ti.alpha, ti.zstar),
            va='bottom', ha='left', bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.3'))

    ax.fill_between(time, t.z, ti.zstar, where=t.z >= ti.zstar, interpolate=True, color=get_tfill_color(mode))  # shade between curve and threshold
    plt.tight_layout()

    if save_figures: plt.savefig(figure_output_path, dpi=150)

    if show_plot:  # either show plot...
        plt.show()
    else:  # or clear plot during automated batch tasks to clear memory
        plt.close(fig)


def remove_spines(ax):
    """ Simple auxiliary function to remove upper and right spines from the passed axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def get_active_color(mode):
    """
    Returns a color dynamically to match either Potentiated or Atrophied mode
    Used in the graph of ``active'' measurement data
    """
    if mode == BASE_POT:
        return pot_color
    elif mode == BASE_ATRO:
        return atroph_color
    else:  # should never happen
        print("Error: Unidentified mode: {}".format(mode))
        return pot_color


def get_tfill_color(mode):
    """
    Returns a color dynamically to match either Potentiated or Atrophied mode
    Used in when shading the region between t statistic and the threshold level
    """
    if mode == BASE_POT:
        return tpotfill_color
    elif mode == BASE_ATRO:
        return tatrophfill_color
    else:  # should never happen
        print("Error: Unidentified mode: {}".format(mode))
        return tpotfill_color

