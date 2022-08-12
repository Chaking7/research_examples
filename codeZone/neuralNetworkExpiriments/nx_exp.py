import os
import numpy as np
import pandas as pd
import logging

from Experiment import Experiment
from datetime import datetime
from experiment_utils import hyper_search, ValSplitter
from peplearn import write_to_csv, create_range


class Datasize(Experiment):
    def __init__(self):
        super().__init__()

        self.splitter = ValSplitter(x=self.x, y=self.y)
        self.params_to_try = hyper_search(path=self.args.params_file)

    def run(self):
        x_train, y_train, x_val, y_val = self.splitter(split_no=0)
        logging.info('')
        logging.info('Performing dataset size walk')

        # creates dataset list up to the amount of datapoints available
        size = create_range(length=len(x_train),num_points=10)
        i = 0  # for model number
        for point_count in size:
            logging.info('Trying with %f data points.', point_count)
            logging.info(point_count)
            params = {**vars(self.args),
                      'output_dir': os.path.join(self.args.log_dir,
                      'datasize' + str(point_count)), 'epoch_size': 1000}
            self.model = self.set_model(params=params)

            if not os.path.exists(params['output_dir']):
                os.makedirs(params['output_dir'])
            # reset model

            try:
                # This goes through the dataset and randomly selects
                # point_count datapooints
                chosen_idx = np.random.choice(len(x_train), size=point_count)
                # choose N indices for this reduced training set
                self.model = self.model.train(x=x_train[chosen_idx],
                                            y=y_train[chosen_idx],
                                            x_val=x_val, y_val=y_val,
                                            params=params)

                # Model is trained now:
                y_predicted = self.model(x_val)
                # Report performance metrics
                self.model.save(params['output_dir'])
                metrics = self.report_result(y_predicted, y_val,
                                             directory=params['output_dir'])

                if self.model.val_loss[1]:
                    val_loss = min(self.model.val_loss[1])
                else:
                    val_loss = 'NA'
                i = i + 1  # this is for model number
                write_to_csv({'date': str(datetime.now()),
                              'Pearson': metrics['pearson'],
                              'model_no': i,
                              'datapoint_size': point_count,
                              'val_loss': val_loss},
                             os.path.join(self.args.log_dir, 'summary.csv'))
            except Exception as e:
                logging.error(e)

    def set_parser(self):
        # Set default parser args
        parser = super().set_parser()

        parser.add_argument('--params_file', type=str,
                            default='hyper_tuning.csv',
                    help='CSV containing hyperparam values & spread.')
        parser.add_argument('--num_trials', type=int, default=10,
                    help='Number of hyperparameter randomizations to execute.')
        parser.add_argument('--max_samples', type=int, default=1e5,
                    help='Number of samples to train each model.')
        parser.add_argument('--stop_samples', type=int, default=5e4,
                    help='Samples with no progress to trigger early stop.')
        parser.add_argument('--cut_samples', type=int, default=1.5e4,
                help='No progress epochs to trigger learning rate reduction.')
        return parser

if __name__ == "__main__":
    exp = Datasize()
    exp.run()
