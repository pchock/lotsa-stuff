import numpy as np
import random


class ThompsonSampler:
    def __init__(self, variants):
        self.variants = variants
        self.cum_successes = [0] * self.variants
        self.cum_fails = [0] * self.variants
        self.beta_probs = [random.betavariate(1, 1) for i in range(self.variants)]
        self.feedback_api = 'feedback_api'
        self.regret_vector = []
        self.cum_regret = []

    def run_trials(self, trials=10000):
        for trial in range(trials):
            # get variant with the highest probability to test in this trial
            variant = max(range(len(self.beta_probs)), key=self.beta_probs.__getitem__)
            feedback = self.get_feedback(variant)
            self.sampler(variant, feedback)

    def sampler(self, variant, feedback):
        # if success, add to successes for that variant, otherwise added to fail
        self.cum_successes[variant] += feedback
        self.cum_fails[variant] += 1 - feedback

        # recalculate beta_probs based on cum_successes and cum_fails for each variant over all the trials so far
        self.beta_probs = [np.random.beta(self.cum_successes[i] + 1, self.cum_fails[i] + 1) 
                           for i in range(self.variants)]

        # calculate regret as the maximum beta_prob of all variants minus the beta_prob of the variant that was chosen
        regret = max(self.beta_probs) - self.beta_probs[variant]
        self.regret_vector.append(regret)
        self.cum_regret.append(self.cum_regret[-1] + regret)
        
    def get_feedback(self, variant):
        return 'got it!'

