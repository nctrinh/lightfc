import json
import os
import pickle

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import tikzplotlib
import torch

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation.environment import env_settings


def get_plot_draw_styles():
    plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style


def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved


def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        match = [t['name'] == base_tracker['name'] and t['param'] == base_tracker['param'] for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()

    return eval_data


def get_tracker_display_name(tracker):
    if tracker['disp_name'] is None:
        if tracker['run_id'] is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{:03d}'.format(tracker['name'], tracker['param'],
                                              tracker['run_id'])
    else:
        disp_name = tracker['disp_name']

    return disp_name


def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    # Plot settings
    font_size = plot_opts.get('font_size', 20)
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    ylabel = "%s" % (ylabel.replace('%', '\%'))
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = r"$\bf{%s}$" % (plot_opts['title'])

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    try:
        # add bold to our method
        for i in range(1, 2):
            legend_text[-i] = r'\textbf{%s}' % (legend_text[-i])

        ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
                  fontsize=font_size_legend, framealpha=1.0)
    except:
        pass

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    # tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()


def check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False, env_num=None, **kwargs):
    # Load data
    settings = env_settings(env_num=env_num)

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path) and not force_evaluation:
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
    else:
        # print('Pre-computed evaluation data not found. Computing results!')
        eval_data = extract_results(trackers, dataset, report_name, env_num=env_num, **kwargs)

    if not check_eval_data_is_valid(eval_data, trackers, dataset):
        # print('Pre-computed evaluation data invalid. Re-computing results!')
        eval_data = extract_results(trackers, dataset, report_name, env_num=env_num, **kwargs)
        # pass
    else:
        # Update display names
        tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                         for t in trackers]
        eval_data['trackers'] = tracker_names
    with open(eval_data_path, 'wb') as fh:
        pickle.dump(eval_data, fh)
    return eval_data


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]

    return prec_curve, prec_score


def plot_results(trackers, dataset, report_name, merge_results=False,
                 plot_types=('success'), force_evaluation=False, env_num=None, **kwargs):
    """
    Plot results for the given trackers

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success',
                    'prec' (precision), and 'norm_prec' (normalized precision)
    """
    # Load data
    settings = env_settings(env_num)

    plot_draw_styles = get_plot_draw_styles()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']

    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print(
        '\nPlotting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    print('\nGenerating plots for: {}'.format(report_name))

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 88), 'title': 'Success'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path,
                       success_plot_opts)

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       precision_plot_opts)

    # ********************************  Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 85), 'title': 'Normalized Precision'}
        plot_draw_save(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles,
                       result_plot_path,
                       norm_precision_plot_opts)

    plt.show()


def generate_formatted_report(row_labels, scores, table_name=''):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = '{prev}{tracker: <{width}} |'.format(prev=report_text, tracker=d_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value[trk_id].item()),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text


def print_results(trackers, dataset, report_name, merge_results=False, plot_types=('success'), env_num=None, **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, env_num=env_num, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(),
                                                              valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)


def plot_got_success(trackers, report_name, env_num):
    """ Plot success plot for GOT-10k dataset using the json reports.
    Save the json reports from http://got-10k.aitestunion.com/leaderboard in the directory set to
    env_settings.got_reports_path

    The tracker name in the experiment file should be set to the name of the report file for that tracker,
    e.g. DiMP50_report_2019_09_02_15_44_25 if the report is name DiMP50_report_2019_09_02_15_44_25.json

    args:
        trackers - List of trackers to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
    """
    # Load data
    settings = env_settings(env_num)
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    auc_curve = torch.zeros((len(trackers), 101))
    scores = torch.zeros(len(trackers))

    # Load results
    tracker_names = []
    for trk_id, trk in enumerate(trackers):
        json_path = '{}/{}.json'.format(settings.got_reports_path, trk.name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                eval_data = json.load(f)
        else:
            raise Exception('Report not found {}'.format(json_path))

        if len(eval_data.keys()) > 1:
            raise Exception

        # First field is the tracker name. Index it out
        eval_data = eval_data[list(eval_data.keys())[0]]
        if 'succ_curve' in eval_data.keys():
            curve = eval_data['succ_curve']
            ao = eval_data['ao']
        elif 'overall' in eval_data.keys() and 'succ_curve' in eval_data['overall'].keys():
            curve = eval_data['overall']['succ_curve']
            ao = eval_data['overall']['ao']
        else:
            raise Exception('Invalid JSON file {}'.format(json_path))

        auc_curve[trk_id, :] = torch.tensor(curve) * 100.0
        scores[trk_id] = ao * 100.0

        tracker_names.append({'name': trk.name, 'param': trk.parameter_name, 'run_id': trk.run_id,
                              'disp_name': trk.display_name})

    threshold_set_overlap = torch.arange(0.0, 1.01, 0.01, dtype=torch.float64)

    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
    plot_draw_save(auc_curve, threshold_set_overlap, scores, tracker_names, plot_draw_styles, result_plot_path,
                   success_plot_opts)
    plt.show()


def print_per_sequence_results(trackers, dataset, report_name, merge_results=False,
                               filter_criteria=None, env_num=None, **kwargs):
    """ Print per-sequence results for the given trackers. Additionally, the sequences to list can be filtered using
    the filter criteria.

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        filter_criteria - Filter sequence results which are reported. Following modes are supported
                        None: No filtering. Display results for all sequences in dataset
                        'ao_min': Only display sequences for which the minimum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences where at least one tracker performs poorly.
                        'ao_max': Only display sequences for which the maximum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences all tracker performs poorly.
                        'delta_ao': Only display sequences for which the performance of different trackers vary by at
                                    least filter_criteria['threshold'] in average overlap (AO) score. This mode can
                                    be used to select sequences where the behaviour of the trackers greatly differ
                                    between each other.
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, env_num=env_num, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    sequence_names = eval_data['sequences']
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

    # Filter sequences
    if filter_criteria is not None:
        if filter_criteria['mode'] == 'ao_min':
            min_ao = avg_overlap_all.min(dim=1)[0]
            valid_sequence = valid_sequence & (min_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'ao_max':
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & (max_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'delta_ao':
            min_ao = avg_overlap_all.min(dim=1)[0]
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & ((max_ao - min_ao) > filter_criteria['threshold'])
        else:
            raise Exception

    avg_overlap_all = avg_overlap_all[valid_sequence, :]
    sequence_names = [s + ' (ID={})'.format(i) for i, (s, v) in enumerate(zip(sequence_names, valid_sequence.tolist()))
                      if v]

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]

    scores_per_tracker = {k: avg_overlap_all[:, i] for i, k in enumerate(tracker_disp_names)}
    report_text = generate_formatted_report(sequence_names, scores_per_tracker)

    print(report_text)


def print_results_per_video(trackers, dataset, report_name, merge_results=False,
                            plot_types=('success'), per_video=False, env_num=None, **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, env_num=env_num, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    seq_lens = len(eval_data['sequences'])
    eval_datas = [{} for _ in range(seq_lens)]
    if per_video:
        for key, value in eval_data.items():
            if len(value) == seq_lens:
                for i in range(seq_lens):
                    eval_datas[i][key] = [value[i]]
            else:
                for i in range(seq_lens):
                    eval_datas[i][key] = value

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(),
                                                              valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)

    if per_video:
        for i in range(seq_lens):
            eval_data = eval_datas[i]

            print('\n{} sequences'.format(eval_data['sequences'][0]))

            scores = {}
            valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

            # ********************************  Success Plot **************************************
            if 'success' in plot_types:
                threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
                ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

                # Index out valid sequences
                auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
                scores['AUC'] = auc
                scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
                scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

            # ********************************  Precision Plot **************************************
            if 'prec' in plot_types:
                ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

                # Index out valid sequences
                prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
                scores['Precision'] = prec_score

            # ********************************  Norm Precision Plot *********************************
            if 'norm_prec' in plot_types:
                ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

                # Index out valid sequences
                norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
                scores['Norm Precision'] = norm_prec_score

            # Print
            tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
            report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
            print(report_text)


def _read_pred_bboxes(bbox_file):
    """Read a tracker's per-sequence result file (x y w h per line, tab separated)."""
    if not os.path.isfile(bbox_file):
        return None
    pred_bb = np.loadtxt(bbox_file, delimiter='\t', dtype=np.float64)
    if pred_bb.ndim == 1:
        pred_bb = pred_bb[None, :]
    return pred_bb


def _draw_box(image, box, color, thickness=2, label=None):
    """Draw a single [x, y, w, h] box (and optional label) on an image (in-place)."""
    if box is None or np.any(np.isnan(box)):
        return image
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    if label is not None:
        cv2.putText(image, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def save_tracking_visualizations(trackers, dataset, report_name, save_dir=None, env_num=None,
                                  gt_color=(0, 255, 0), pred_color=(0, 0, 255),
                                  sequence_names=None, max_frames=None, save_video=False, fps=25):
    """Save frames overlaid with ground-truth and predicted bounding boxes.

    For every sequence in `dataset` and every tracker in `trackers`, this reads the
    tracker's saved result file (produced by tracking/test.py) together with the
    sequence's ground-truth annotations, draws both boxes on each frame, and saves
    the resulting images (and optionally a video) to disk.

    args:
        trackers - List of Tracker objects (as returned by trackerlist())
        dataset - List of Sequence objects (as returned by get_dataset())
        report_name - Name used for the output sub-folder, e.g. dataset name
        save_dir - Root directory to save visualizations. Defaults to
                   <env.result_plot_path>/<report_name>/visualizations
        env_num - Environment number (same as used elsewhere in this repo)
        gt_color - BGR color for the ground-truth box (default green)
        pred_color - BGR color for the predicted box (default red)
        sequence_names - Optional list of sequence names to restrict visualization to.
                          If None, all sequences in `dataset` are processed.
        max_frames - Optional cap on number of frames drawn per sequence (useful for a
                     quick preview). If None, all frames are processed.
        save_video - If True, also saves an .mp4 video per (tracker, sequence) built
                     from the annotated frames, in addition to the individual images.
        fps - Frames per second used when save_video is True.
    """
    settings = env_settings(env_num)

    if save_dir is None:
        save_dir = os.path.join(settings.result_plot_path, report_name, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    for seq in dataset:
        if sequence_names is not None and seq.name not in sequence_names:
            continue

        gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)

        for trk in trackers:
            bbox_file = os.path.join(trk.results_dir, seq.dataset, '{}.txt'.format(seq.name))
            pred_bb = _read_pred_bboxes(bbox_file)
            if pred_bb is None:
                # print('Skipping {} / {}: result file not found ({})'.format(
                #     get_tracker_display_name({'name': trk.name, 'param': trk.parameter_name,
                #                               'run_id': trk.run_id, 'disp_name': trk.display_name}),
                #     seq.name, bbox_file))
                continue

            disp_name = get_tracker_display_name({'name': trk.name, 'param': trk.parameter_name,
                                                  'run_id': trk.run_id, 'disp_name': trk.display_name})
            print(disp_name, seq.name, 'Saving annotated frames to disk...')
            seq_save_dir = os.path.join(save_dir, disp_name, seq.name)
            os.makedirs(seq_save_dir, exist_ok=True)

            num_frames = len(seq.frames)
            if max_frames is not None:
                num_frames = min(num_frames, max_frames)

            video_writer = None
            for frame_idx in range(num_frames):
                image = cv2.imread(seq.frames[frame_idx])
                if image is None:
                    continue

                # Ground-truth box for this frame (if available)
                if frame_idx < gt.shape[0]:
                    _draw_box(image, gt[frame_idx], gt_color, label='GT')

                # Predicted box for this frame (if available)
                if frame_idx < pred_bb.shape[0]:
                    _draw_box(image, pred_bb[frame_idx], pred_color, label=disp_name)

                out_path = os.path.join(seq_save_dir, '{:06d}.jpg'.format(frame_idx))
                cv2.imwrite(out_path, image)

                if save_video:
                    if video_writer is None:
                        h, w = image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_path = os.path.join(save_dir, disp_name, '{}.mp4'.format(seq.name))
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                    video_writer.write(image)

            if video_writer is not None:
                video_writer.release()

            print('Saved {} annotated frames for {} / {} to {}'.format(
                num_frames, disp_name, seq.name, seq_save_dir))


def print_results_and_save_visualizations(trackers, dataset, report_name, merge_results=False,
                                          plot_types=('success'), env_num=None,
                                          save_dir=None, sequence_names=None, max_frames=None,
                                          save_video=False, **kwargs):
    """Convenience wrapper: print the metric table (like print_results) AND save
    per-frame images with ground-truth / predicted bounding boxes drawn on them
    (like save_tracking_visualizations), in a single call.

    args:
        trackers, dataset, report_name, merge_results, plot_types, env_num, **kwargs -
            same as print_results()
        save_dir, sequence_names, max_frames, save_video -
            same as save_tracking_visualizations()
    """
    # 1) Metrics table (existing behaviour, unchanged)
    print_results(trackers, dataset, report_name, merge_results=merge_results,
                 plot_types=plot_types, env_num=env_num, **kwargs)

    # 2) Save annotated frames (bbox + ground-truth) for every sequence/tracker
    save_tracking_visualizations(trackers, dataset, report_name, save_dir=save_dir, env_num=env_num,
                                 sequence_names=sequence_names, max_frames=max_frames,
                                 save_video=save_video)