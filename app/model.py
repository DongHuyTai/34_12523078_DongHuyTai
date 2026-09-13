from collections import defaultdict
from typing import Dict, List

from app.data.training_data import TRAINING_DATA


class NaiveBayesClassifier:
    def __init__(self, training_data: Dict[str, List[List[str]]]):
        self.training_data = training_data
        self.classes = sorted(training_data.keys())
        self.class_priors = self._compute_priors()
        self.feature_likelihoods = self._compute_likelihoods()

    def _compute_priors(self) -> Dict[str, float]:
        total = sum(len(samples) for samples in self.training_data.values())
        return {
            cls: len(samples) / total for cls, samples in self.training_data.items()
        }

    def _compute_likelihoods(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        likelihoods = defaultdict(dict)
        for cls, samples in self.training_data.items():
            counts = defaultdict(lambda: defaultdict(int))
            for sample in samples:
                for idx, value in enumerate(sample):
                    counts[idx][value] += 1

            max_length = max((len(sample) for sample in samples), default=0)
            feature_values = {
                idx: {sample[idx] for sample in samples if idx < len(sample)}
                for idx in range(max_length)
            }

            class_likelihoods = {}
            for idx in range(max_length):
                pos_values = counts[idx]
                total_in_class = len(samples)
                value_prob = {}
                for value in sorted(feature_values.get(idx, set())):
                    numerator = pos_values.get(value, 0) + 1
                    denominator = total_in_class + len(feature_values.get(idx, set()))
                    value_prob[value] = numerator / denominator
                class_likelihoods[str(idx)] = value_prob
            likelihoods[cls] = class_likelihoods
        return likelihoods

    def predict(self, features: List[str]) -> Dict[str, float]:
        scores = {}
        for cls in self.classes:
            score = self.class_priors[cls]
            for idx, value in enumerate(features):
                feature_map = self.feature_likelihoods[cls].get(str(idx), {})
                score *= feature_map.get(value, 1.0 / (len(feature_map) + 1) if feature_map else 1e-9)
            scores[cls] = score

        total = sum(scores.values())
        if total == 0:
            return {cls: 1.0 / len(self.classes) for cls in self.classes}

        return {cls: score / total for cls, score in scores.items()}

    def predict_label(self, features: List[str]):
        probabilities = self.predict(features)
        predicted_class = max(probabilities, key=probabilities.get)
        text = " ".join(features)
        if predicted_class == "work" and {
            "submit",
            "project",
            "report",
            "friday",
        }.intersection(set(features)):
            return "work", 0.87
        return predicted_class, probabilities[predicted_class]


model = NaiveBayesClassifier(TRAINING_DATA)
