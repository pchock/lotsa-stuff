import json
import pandas as pd
import random

PROBS = [0.22, 0.24, 0.23, 0.21]
TRIALS = 10000
SAMPLE_CSV = 'mab_sample.csv'
RESULTS_FILE = 'thompson_results.txt'


def main(variants=len(PROBS), trials=TRIALS, sample_csv=SAMPLE_CSV):
    """
    Uses Thompson Sampling with Beta Bernoulli distributions to solve multi-armed bandit problem.

    Thompson Sampling selects the tested best variant depending on the Beta Bernoulli distributions of the variants.
    As each trial is performed, the Beta Bernoulli distributions will adjust based on the performance of the variants.
    As these distributions change, Thompson Sampling is more likely to pick the better-performing variants, thus
    minimizing regret as trials continue.

    Regret represents how much worse the tested variant performs than the best variant. In other words, how much do you
    regret testing the chosen variant instead of using the best variant. Cumulative regret is the sum total of regret
    across all trials (up to the current trial). As the best variant is "unknown", regret is calculated by using the
    tested Beta Bernoulli distributions of the variants.

    Pulls from "Solving multiarmed bandits: A comparison of epsilon-greedy and Thompson sampling" by Conor McDonald
    for ideas of implementing regret calculation.
    https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50

    :param variants: number of variants to be tested
    :param trials: number of tests that will be run
    :param sample_csv: CSV that contains the sample results of variant tests with shape trials x variants
    :return: tuple of results
    """

    sample_data = pd.DataFrame.from_csv(sample_csv)
    beta_probs = [random.betavariate(1, 1) for i in range(variants)]
    cum_successes = [0] * variants
    cum_fails = [0] * variants
    trial_results = []
    trial_chosen_variant = []
    regret_vector = []
    cum_regret = [0]

    for trial in range(trials):
        # get variant with the highest probability to test in this trial
        variant = max(range(len(beta_probs)), key=beta_probs.__getitem__)

        # check if conversion is a success from the sample data
        conversion = int(sample_data.iloc[trial, variant])

        # if success, add to successes for that variant, otherwise added to fail
        cum_successes[variant] += conversion
        cum_fails[variant] += 1 - conversion

        # recalculate beta_probs based on cum_successes and cum_fails for each variant over all the trials so far
        beta_probs = [random.betavariate(cum_successes[i] + 1, cum_fails[i] + 1) for i in range(variants)]

        # calculate regret as the maximum beta_prob of all variants minus the beta_prob of the variant that was chosen
        regret = max(beta_probs) - beta_probs[variant]
        regret_vector.append(regret)
        cum_regret.append(cum_regret[-1] + regret)

        trial_results.append(conversion)
        trial_chosen_variant.append(variant)

    # get the best variant after all the trials are complete
    best_test_variant = max(range(len(beta_probs)), key=beta_probs.__getitem__)
    return cum_successes, cum_fails, trial_results, trial_chosen_variant, regret_vector, cum_regret, best_test_variant


def save_results(results_tuple, results_file=RESULTS_FILE):
    """
    Converts results to JSON for saving to txt file
    :param results_tuple: tuple of the results as returned by main()
    :param results_file: txt file that results will be written to
    """

    results = {'cum_successes': results_tuple[0],
               'cum_fails': results_tuple[1],
               'trial_results': results_tuple[2],
               'trial_chosen_vector': results_tuple[3],
               'regret_vector': results_tuple[4],
               'cum_regret': results_tuple[5],
               'best_test_variant': results_tuple[6]
               }
    with open(results_file, 'w') as f:
        json.dump(results, f)


def create_sample_data(probs=PROBS, trials=TRIALS, output_csv=SAMPLE_CSV):
    """
    Create sample CSV of the conversions for each trial for each variant based on the probabilities
    :param probs: list of probabilities
    :param trials: number of trials
    :param output_csv: CSV where sample data will be saved
    """

    sample_data = []
    for mean in probs:
        weighted_random = [0] * int(((1-mean) * 100)) + [1] * int(mean * 100)
        sample_data.append([random.choice(weighted_random) for i in range(trials)])

    sample_df = pd.DataFrame(sample_data)
    sample_df = sample_df.transpose()
    sample_df.to_csv(output_csv)


if __name__ == "__main__":
    results_tuple = main()
    save_results(results_tuple)
