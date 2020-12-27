import json
import pandas as pd
import git
import os
import sys
from datetime import datetime
from collections import defaultdict
import atexit
from typing import Union, Dict, Any
import warnings
import shutil


class Mellogger:
    r"""Logger for a single run.

    Arguments:
        log_dir (str): name of a directory for logging
        exp_name (str): name of the experiment. Should be the same for all the runs of an experiment
        args (str or dict): if string, path of a JSON file with the arguments, else, dict of parameters. If None, no params are registered.
        dump_frequency (int): cumulative number of logged value before a dump on disk is done
        external_logging (str): name of the external tool to be used, if needed
        external_project (str): name of the project on the external logging tool
        external_account (str): name of the account on the external logging tool
        test_mode (bool): for temporary experiments, remove all the directory
    """
    def __init__(self, log_dir: str, exp_name: str, args: Union[str, Dict[str, Any]] = None,
                 dump_frequency: int = int(1e6), external_logging: str = None,
                 external_project: str = None, external_account: str = None,
                 test_mode: bool = False):
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.dump_frequency = dump_frequency
        self.external_logging = external_logging
        self.external_account = external_account
        self.external_project = external_project
        self.test_mode = test_mode
        self.log_counter = 0
        # Obtain "system" information
        repo = git.Repo(search_parent_directories=True)
        # Hash of the current commit
        commit = repo.head.object.hexsha
        # Name of the current branch
        branch = str(repo.active_branch)
        # Date and time of the launch of the run
        start_timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        # Command used for launching main script
        command = " ".join([os.path.basename(sys.executable)] + sys.argv)

        system_info = dict(commit=commit, branch=branch, start_timestamp=start_timestamp,
                           command=command)

        self.base_dir = os.path.join(log_dir, exp_name+"__ts"+start_timestamp.replace(" ", ""))
        os.makedirs(self.base_dir, exist_ok=True)

        with open(os.path.join(self.base_dir, "system_info.json"), 'w') as f:
            json.dump(system_info, f, indent=4, sort_keys=True)
        if isinstance(args, str):
            with open(args, 'r') as f:
                self.parameters = json.load(f)
        elif isinstance(args, dict):
            self.parameters = args
        elif args is None:
            self.parameters = {}
        # Log parameters
        with open(os.path.join(self.base_dir, "parameters.json"), 'w') as f:
            json.dump(self.parameters, f, indent=4, sort_keys=True)

        # Initialize data structure for metrics
        self.metric_dict = defaultdict(list)

        # Setup external logging if required
        if self.external_logging and not self.external_account:
            raise ValueError("An external account should be provided")
        if self.external_logging == "neptune":
            raise NotImplementedError
        # Setup final dump
        atexit.register(self.conclude_experiment)

        # If test mode is enabled, do cleanup even after a crash
        if self.test_mode:
            def clean_directory(type, value, tb):
                shutil.rmtree(self.base_dir, ignore_errors=True)
                # Call default function for displaying the traceback
                sys.__excepthook__(type, value, tb)
            sys.excepthook = clean_directory

    def dump_metrics(self) -> None:
        # Dump metrics if log3 has been used
        if self.log_counter >= 1:
            df = pd.DataFrame.from_dict(self.metric_dict, orient='index').T
            try:
                df.to_csv(os.path.join(self.base_dir, "metrics.csv"), index=False)
            except FileNotFoundError:
                # Someone deleted the directory before the program was finished
                warnings.warn("Experiment directory was deleted before the experiment finished")

    def log(self, key: str, value: float) -> None:
        self.log_counter += 1
        self.metric_dict[key].append(value)
        # If dump frequency is reached, rewrite log3
        if self.log_counter % self.dump_frequency == 0:
            self.dump_metrics()

    def conclude_experiment(self) -> None:
        self.dump_metrics()
        if self.test_mode:
            shutil.rmtree(self.base_dir, ignore_errors=True)
