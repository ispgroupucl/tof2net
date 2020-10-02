import matplotlib.pyplot as plt
from glob import glob
from os import path
import os
import argparse
import pandas
from collections import OrderedDict
import re
import plotly.offline as ply
import plotly.graph_objs as go
from plotly import tools
import shutil
import time


def get_metrics(args):
    metrics = {}
    files = sorted(glob(path.join(args.dir, "**/metrics.csv"), recursive=True))
    if args.regex is not None:
        files = [x for x in files if re.search(args.regex, x)]
    for metrics_file in files:
        name = metrics_file.split("/")[-2]
        metric = pandas.read_csv(metrics_file)
        metrics[name] = metric
    
    return metrics


def list_metrics(args):
    metrics = OrderedDict()
    for metric in get_metrics(args).values():
        metrics.update({x: True for x in metric})
    return metrics

def compare_dirs(args):
    print(args)
    # fig, ax = plt.subplots(1,1)
    data = []
    args.metrics = get_final_metrics(args)
    for name, metric in get_metrics(args).items():
        if args.subplots:
            data.append([])
        metric = metric.add_prefix("{name}_".format(locals()))
        prefix = name + "_"
        for mname in args.metrics:
            mname = "{prefix}{mname}".format(locals())
            sc = go.Scatter(
                        x=metric["{prefix}epoch".format(locals())],
                        y=metric[mname],
                        name=mname)
            data[-1].append(sc)

        # metric.iplot(f'{name}_epoch', [f'{name}_{x}' for x in args.metrics], ax=ax)
    if args.subplots:
        fig = tools.make_subplots(rows=len(data), cols=1)
        for i, d in enumerate(data):
            for dd in d:
                fig.append_trace(dd, row=i+1, col=1)
        fig['layout'].update(height=540*len(data))
        url = ply.plot(fig, filename="test1")
    else:
        url = ply.plot(data, filename="test1")
    time.sleep(2)
    os.remove(url)

def get_final_metrics(args):
    metrics = list_metrics(args)
    final_metrics = OrderedDict()
    for remetric in args.metrics:
        r = re.compile(remetric)
        for metric in metrics:
            if r.search(metric):
                final_metrics[metric] = True
    return final_metrics


def main(args):
    if args.list:
        print("available metrics: ")
        print(', '.join(list_metrics(args)))
        return
    compare_dirs(args)

def create_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Compare multiple runs between each other")
    parser.add_argument("dir")
    parser.add_argument("--list", "-l", action="store_true")
    parser.add_argument("--metrics", "-m", nargs="+", default=["loss", "val_loss"])
    parser.add_argument("--regex", "-r")
    parser.add_argument("--subplots", "-s", action="store_true")
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)