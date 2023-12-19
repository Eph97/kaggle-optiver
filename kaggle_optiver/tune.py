import importlib
import logging
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from optuna import create_study
from optuna.samplers import TPESampler
from typing import Any, Dict, Tuple


from sklearn.base import clone
from sklearn.model_selection import cross_val_predict


class TuningSession:
    """A class to carry out an optuna search of hyperparameters to be used in
    the models selected by user. It outputs two csv's per estimator. The
    first contains the set of hyperparameters used in the optuna trials. The
    second contains the backtest series generated in each individual trial.

    Args:
        start_date (datetime): The start date of the in-sample period used in
        cross-validation

        end_date (datetime): The end date of the in-sample period used in
        cross-validation

        hyperparam_dists_path (str): Path to the yaml file with
        hyperparameters distributions
    """

    def __init__(
        self,
        hyperparam_dists_path: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        with open(hyperparam_dists_path, "rb") as file:
            self.classifiers_config = yaml.load(file, Loader=yaml.FullLoader)

    def _get_imported_class(self, import_path: str) -> Any:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        class_ = getattr(module, class_name)
        return class_

    def _config_parser(self) -> Dict[str, Tuple[Any, Dict[str, Any], int]]:
        classifier_distributions_dict = {}
        for (
            classifier_path,
            config_for_classifier,
        ) in self.classifiers_config.items():
            _, classifier_name = classifier_path.rsplit(".", 1)
            classifier = self._get_imported_class(
                import_path=classifier_path,
            )()
            n_trials = config_for_classifier["n_trials"]

            distribution = {}
            key_values = config_for_classifier["hyperparameters"].items()
            for hyperparameter_name, hyperparameter_dist_info in key_values:
                distribution[hyperparameter_name] = self._get_imported_class(
                    import_path=hyperparameter_dist_info["distribution_type"]
                )(**hyperparameter_dist_info["distribution_kwargs"])

            classifier_distributions_dict[classifier_name] = (
                classifier,
                distribution,
                n_trials,
            )
        return classifier_distributions_dict

    def _get_study(
        self,
        classifier,
        distributions,
        transformed_data: pd.DataFrame,
        y: np.ndarray,
    ):
        # self._backtests_dict = {}

        def objective(trial):
            estimator = clone(classifier).set_params(
                **{
                    name: trial._suggest(name, distribution)
                    for name, distribution in distributions.items()
                }
            )

            y_pred = pd.Series(
                data=cross_val_predict(
                    estimator=estimator,
                    X=transformed_data,
                    y=y,
                    cv=None,  # None my default gives 5-fold cross-validation
                    method="predict",
                ),
            )

            score = np.mean(np.abs(y - y_pred))

            # bookkepping
            # self._backtests_dict[trial.number] = y_pred

            return score

        return (
            create_study(sampler=TPESampler(seed=42), direction=None),
            objective,
        )

    def run(
        self,
        data: None,
        labels: None,
    ) -> None:
        logging.basicConfig(
            format="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
        logging.info("Preparing and Transforming data")
        transformed_data = data
        y = labels
        key_values = self._config_parser().items()
        self.backtests_dict = {}

        for classifier_name, config_for_classifier in key_values:
            logging.info(f"Starting {classifier_name}'s study")
            classifier, distributions, n_trials = config_for_classifier

            study, objective = self._get_study(
                classifier=classifier,
                distributions=distributions,
                transformed_data=transformed_data,
                y=y,
            )
            study.optimize(func=objective, n_trials=n_trials)

            self.backtests_dict[classifier_name] = study.trials_dataframe()

            filepath = f"trials_{classifier_name}.csv"
            study.trials_dataframe().to_csv(
                filepath,
                index=False,
            )

            logging.info(f"Finished {classifier_name}'s study")

        # for classifier_name, backtests in backtests_master_dict.items():
        #     # upload a single csv with all trials' backtests for this
        #     # classifier
        #     backtests_df = pd.concat(backtests, axis=1)
        #     filepath = f"trials_{classifier_name}.csv"
        #     backtests_df.to_csv(
        #         filepath,
        #         index=True,
        #     )

        logging.info("Finished")
