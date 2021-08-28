class ResultsStorage:
    def __init__(self):
        self._results = dict()
        self._label = "default"

    def set_label(self, label):
        self._label = label

    def append(self, result, label=None):
        if label is None:
            label = self._label

        current_results = self._results.get(label, [])
        current_results.append(result)
        self._results[label] = current_results

    def __getitem__(self, item):
        return self._results[item]

    @property
    def results(self):
        return self._results
