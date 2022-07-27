"""
Function for explaining classified instances using evidence counterfactuals.
"""
from baselines.mojito import Mojito

"""
Import libraries 
"""
import time
import pandas as pd


class LimeCounterfactual(object):
    """Class for generating evidence counterfactuals for classifiers on behavioral/text data"""

    def __init__(self, c_fn, classifier_fn, vectorizer, threshold_classifier,
                 feature_names_full, max_features=30, class_names=['1', '0'],
                 time_maximum=120, off_value=''):

        """ Init function

        Args:
            c_fn: [pipeline] for example:

                c = make_pipeline(vectorizer, classification_model)
                (where classification_model is a fitted scikit learn model and
                vectorizer is a fitted object)

            classifier_fn: [function] classifier prediction probability function
            or decision function. For ScikitClassifiers, this is classifier.predict_proba
            or classifier.decision_function or classifier.predict_log_proba.
            Make sure the function only returns one (float) value. For instance, if you
            use a ScikitClassifier, transform the classifier.predict_proba as follows:

                def classifier_fn(X):
                    c=classification_model.predict_proba(X)
                    y_predicted_proba=c[:,1]
                    return y_predicted_proba

            max_features: [int] maximum number of features allowed in the explanation(s).
            Default is set to 30.

            class_names: [list of string values]

            vectorizer: [fitted object] a fitted vectorizer object

            threshold_classifier: [float] the threshold that is used for classifying
            instances as positive or not. When score or probability exceeds the
            threshold value, then the instance is predicted as positive.
            We have no default value, because it is important the user decides
            a good value for the threshold.

            feature_names_full: [numpy.array] contains the interpretable feature names,
            such as the words themselves in case of document classification or the names
            of visited URLs.

            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in minutes. Default is set to 2 minutes (120 seconds).
        """

        self.off_value = off_value
        self.c_fn = c_fn
        self.classifier_fn = classifier_fn
        self.class_names = class_names
        self.max_features = max_features
        self.vectorizer = vectorizer
        self.threshold_classifier = threshold_classifier
        self.feature_names_full = feature_names_full
        self.time_maximum = time_maximum

    def explanation(self, instance):
        """ Generates evidence counterfactual explanation for the instance.

        Args:
            instance: [raw text string] instance to explain as a string
            with raw text in it

        Returns:
            A dictionary where:

                explanation_set: features in counterfactual explanation.

                feature_coefficient_set: corresponding importance weights
                of the features in counterfactual explanation.

                number_active_elements: number of active elements of
                the instance of interest.

                minimum_size_explanation: number of features in the explanation.

                minimum_size_explanation_rel: relative size of the explanation
                (size divided by number of active elements of the instance).

                time_elapsed: number of seconds passed to generate explanation.

                score_predicted[0]: predicted score/probability for instance.

                score_new[0]: predicted score/probability for instance when
                removing the features in the explanation set (~setting feature
                values to zero).

                difference_scores: difference in predicted score/probability
                before and after removing features in the explanation.

                expl_lime: original explanation using LIME (all active features
                with corresponding importance weights)
        """

        tic = time.time()  # start timer
        nb_active_features = np.size(instance)
        score_predicted = self.classifier_fn(instance)
        idx = np.argmax(score_predicted)
        explainer = Mojito(instance.columns,
                        attr_to_copy='left',
                        split_expression=" ",
                        class_names=['no_match', 'match'],
                        lprefix='', rprefix='',
                        feature_selection='lasso_path')

        classifier = self.c_fn.predict_proba


        exp = explainer.copy(classifier, instance,
                                      num_features=nb_active_features,
                                      num_perturbation=100)

        #exp = explainer.explain_instance(instance_text, classifier, num_features=nb_active_features)
        ell = exp.groupby('attribute')['weight'].mean().to_dict()
        explanation_lime = []
        for k, v in ell.items():
            explanation_lime.append((k, v))

        explanation_lime = sorted(explanation_lime, key=lambda x: x[1], reverse=idx == 1)
        """
        indices_features_lime = []
        feature_coefficient = []
        feature_names_full_index = []
        for j in range(len(explanation_lime)):
            if explanation_lime[j][1] >= 0:   #only the features with a zero or positive estimated importance weight are considered
                feature = explanation_lime[j][0]
                index_feature = np.argwhere(np.array(self.feature_names_full)==feature) #returns index in feature_names where == feature
                feature_names_full_index.append(self.feature_names_full[index_feature[0][0]])
                indices_features_lime.append(index_feature[0][0])
                feature_coefficient.append(explanation_lime[j][1])
        """
        if (np.size(instance) != 0):
            score_new = score_predicted
            k = 0
            number_perturbed = 0
            while ((score_new[idx] >= self.threshold_classifier) and (k != len(explanation_lime)) and (
                    time.time() - tic <= self.time_maximum) and (number_perturbed < self.max_features)):
                number_perturbed = 0
                feature_names_full_index = []
                feature_coefficient = []
                k += 1
                perturbed_instance = instance.copy()
                for feature in explanation_lime[0:k]:
                    if (feature[1] > 0 and idx == 1) or (feature[1] < 0 and idx == 0):
                        index_feature = np.argwhere(np.array(self.feature_names_full) == feature[0])
                        number_perturbed += 1
                        if (len(index_feature) != 0):
                            index_feature = index_feature[0][0]
                            perturbed_instance.iloc[:, index_feature] = self.off_value
                            feature_names_full_index.append(index_feature)
                            feature_coefficient.append(feature[1])
                score_new = self.classifier_fn(perturbed_instance)

            if (score_new[idx] < self.threshold_classifier):
                time_elapsed = time.time() - tic
                minimum_size_explanation = number_perturbed
                minimum_size_explanation_rel = number_perturbed / nb_active_features
                difference_scores = (score_predicted - score_new)
                number_active_elements = nb_active_features
                expl_lime = explanation_lime
                explanation_set = feature_names_full_index[0:number_perturbed]
                feature_coefficient_set = feature_coefficient[0:number_perturbed]
                cf_example = perturbed_instance.copy()

            else:
                minimum_size_explanation = np.nan
                minimum_size_explanation_rel = np.nan
                time_elapsed = np.nan
                difference_scores = np.nan
                number_active_elements = nb_active_features
                expl_lime = explanation_lime
                explanation_set = []
                feature_coefficient_set = []
                cf_example = pd.DataFrame()
        else:
            minimum_size_explanation = np.nan
            minimum_size_explanation_rel = np.nan
            time_elapsed = np.nan
            difference_scores = np.nan
            number_active_elements = nb_active_features
            expl_lime = explanation_lime
            explanation_set = []
            feature_coefficient_set = []
            cf_example = pd.DataFrame()

        return {'explanation_set': explanation_set, 'feature_coefficient_set': feature_coefficient_set,
                'number_active_elements': number_active_elements, 'size explanation': minimum_size_explanation,
                'relative size explanation': minimum_size_explanation_rel, 'time elapsed': time_elapsed,
                'original score': score_predicted[0], 'new score': score_new[0], 'difference scores': difference_scores,
                'explanation LIME': expl_lime, 'cf_example': cf_example}

"""
Function to preprocess numerical/binary big data to raw text format and to 
fit the CountVectorizer object to be used in the class object LimeCounterfactual.
"""

"""
Import libraries 
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class Preprocess_LimeCounterfactual(object):
    def __init__(self, binary_count=True):
        """ Init function

        Args:
            binary_count: [“True” or “False”]  when the original data matrix
            contains binary feature values (only 0s and 1s), binary_count is "True".
            All non-zero counts are set to 1. Default is "True".
        """
        self.binary_count = binary_count

    def instance_to_text(self, instance_idx):
        """Function to generate raw text string from instance on (behavioral) big data"""
        active_elements = np.nonzero(instance_idx)[1]
        instance_text=''
        for element in active_elements:
            instance_text+=" "+'a'+np.str(element)
        return instance_text

    def fit_vectorizer(self, instance_idx):
        """Function to fit vectorizer object for (behavioral) big data based on CountVectorizer()"""
        instance_text1=''
        instance_text2=''
        for element in range(np.size(instance_idx)):
            instance_text1+=" "+'a'+np.str(element)
            instance_text2+=" "+'a'+np.str(element)
        artificial_text = [instance_text1, instance_text2]
        vectorizer = CountVectorizer(binary = self.binary_count)
        vectorizer.fit_transform(artificial_text)
        feature_names_indices = vectorizer.get_feature_names()
        return vectorizer, feature_names_indices