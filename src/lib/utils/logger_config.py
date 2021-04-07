#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Log information regarding the execution environment.
This is helpful if you want to recreate an experiment at a later time, or if
you want to understand the environment in which you execute the training.
"""

import logging
import logging.config
import os
import time


def config_pylogger(log_cfg_file, experiment_name, output_dir='logs', verbose=False):
    """Configure the Python logger.
    For each execution of the application, we'd like to create a unique log directory.
    By default this directory is named using the date and time of day, so that directories
    can be sorted by recency.  You can also name your experiments and prefix the log
    directory with this name.  This can be useful when accessing experiment data from
    TensorBoard, for example.
    """
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr if experiment_name is None else experiment_name
    logdir = os.path.join(output_dir, exp_full_name, timestr)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, timestr + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={
                                  'logfilename': log_filename})
    else:
        print("Could not find the logger configuration file {} - using default logger configuration".format(log_cfg_file))
        apply_default_logger_cfg(log_filename)
    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    if verbose:
        msglogger.setLevel(logging.DEBUG)
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

    # Create a symbollic link to the last log file created (for easier access)
    # try:
    #     os.unlink("latest_log_file")
    # except FileNotFoundError:
    #     pass
    # try:
    #     os.unlink("latest_log_dir")
    # except FileNotFoundError:
    #     pass
    # try:
    #     os.symlink(logdir, "latest_log_dir")
    #     os.symlink(log_filename, "latest_log_file")
    # except OSError:
    #     msglogger.debug("Failed to create symlinks to latest logs")
    return msglogger


def apply_default_logger_cfg(log_filename):
    d = {
        'version': 1,
        'formatters': {
            'simple': {
                'class': 'logging.Formatter',
                'format': '%(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_filename,
                'mode': 'w',
                'formatter': 'simple',
            },
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app_cfg': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
        }
    }

    logging.config.dictConfig(d)