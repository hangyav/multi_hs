from sklearn.metrics import precision_recall_fscore_support

import datasets


_DESCRIPTION = """
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class F1Report(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("int32"),
                "references": datasets.Value("int32"),
            }),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support"],
        )

    def _compute(self, predictions, references, label_names=None):
        res = dict()
        scores = precision_recall_fscore_support(
            references,
            predictions,
            average=None,
        )

        for metric, metric_values in zip(['P', 'R', 'F1', 'Support'], scores):
            for idx, value in enumerate(metric_values):
                label_name = f'Label_{idx}' if label_names is None else label_names[idx]
                res[f'{label_name}_{metric}'] = value

        for avg_method in ['macro', 'micro', 'weighted']:
            scores = precision_recall_fscore_support(
                references,
                predictions,
                average=avg_method,
            )
            for metric, value in zip(['P', 'R', 'F1'], scores):
                res[f'{avg_method}_averaged_{metric}'] = value

        return res
