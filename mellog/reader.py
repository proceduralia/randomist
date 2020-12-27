import os
import warnings
import json
import pandas as pd
import numpy
from typing import List, Union, Dict


class Reader:
    r"""Class for reading and aggregating the results of multiple runs.

    Arguments:
        logdir (str): name of the directory containing the runs of the experiments of interest
        exp_name (str or list(str)): name of the experiments. If None, all the experiments are automatically detected
        enforce_parameter_consistency (bool): whether to throw an exception in case of params inconsistency across runs
    """
    def __init__(self, logdir: str, exp_names: Union[str, List[str]] = None,
                 enforce_parameter_consistency: bool = False):
        experiments = os.listdir(logdir)
        if exp_names is None:
            # Autodetect names of experiments
            experiment_names = set(map(lambda x: x[:x.find("__ts")], experiments))
        elif isinstance(exp_names, str):
            experiment_names = [exp_names]
        else:
            experiment_names = exp_names
        self.experiment_names = experiment_names
        self.experiments_dataframes = {}
        self.ns_runs = {}

        for exp_name in experiment_names:
            # Filter required experiment
            experiment_dirs = list(filter(lambda x: exp_name+"__ts" in x, experiments))
            # Check whether the parameters are the same
            parameters_dicts = []
            for dir_name in experiment_dirs:
                with open(os.path.join(logdir, dir_name, "parameters.json")) as f:
                    parameters_dicts.append(json.load(f))
            # Check whether the parameters are the same for all the runs
            if parameters_dicts.count(parameters_dicts[0]) != len(parameters_dicts):
                if enforce_parameter_consistency:
                    raise Exception("Parameters are not the same in all the runs of the experiment")
                else:
                    warnings.warn("Parameters are not the same in all the runs of the experiment")
            # Read all the metric dataframes
            dataframe_list = []
            for dir_name in experiment_dirs:
                filename = os.path.join(logdir, dir_name, "metrics.csv")
                try:
                    dataframe_list.append(pd.read_csv(filename))
                except FileNotFoundError:
                    warnings.warn("No metrics were found in the run directory {}".format(dir_name))
            # If all the directories were empty, skip current experiment and notify
            if dataframe_list == []:
                warnings.warn("No metrics were found for experiment {}".format(exp_name))
                continue

            # Sanity check that all the metrics are consistent in name
            keys_list = [list(df.keys()) for df in dataframe_list]
            self.ns_runs[exp_name] = len(keys_list)
            # Average the runs. If length are not the same, partial average is done
            mean_dataframe = pd.concat(dataframe_list).groupby(level=0).mean().add_suffix("_mean")
            std_dataframe = pd.concat(dataframe_list).groupby(level=0).std().add_suffix("_std")
            max_dataframe = None #pd.concat(dataframe_list).groupby(level=0).max().add_suffix("_max")
            min_dataframe = None #pd.concat(dataframe_list).groupby(level=0).min().add_suffix("_min")
            count_dataframe = None #pd.concat(dataframe_list).groupby(level=0).count().add_suffix("_count")

            final_dataframe = pd.concat([mean_dataframe, std_dataframe, max_dataframe,
                                        min_dataframe, count_dataframe], axis=1)
            self.experiments_dataframes[exp_name] = final_dataframe

    def get_metric_mean(self, metric_name: str,
                        exp_names: Union[str, List[str]] = None) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
        key = metric_name + "_mean"
        if exp_names is None:
            # Compute metric for all the available experiments
            experiment_names = self.experiment_names
        elif isinstance(exp_names, str):
            return self.experiments_dataframes[exp_names][key].values
        else:
            experiment_names = exp_names
        return {exp_name: self.experiments_dataframes[exp_name][key].values for exp_name in experiment_names}

    def get_metric_std(self, metric_name: str,
                       exp_names: Union[str, List[str]] = None) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
        key = metric_name + "_std"
        if exp_names is None:
            # Compute metric for all the available experiments
            experiment_names = self.experiment_names
        elif isinstance(exp_names, str):
            return self.experiments_dataframes[exp_names][key].values
        else:
            experiment_names = exp_names
        return {exp_name: self.experiments_dataframes[exp_name][key].values for exp_name in experiment_names}
