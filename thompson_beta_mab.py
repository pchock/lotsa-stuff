import random

VARIANTS = 5
TRIALS = 10000


def main(variants=VARIANTS, trials=TRIALS):
    """
    Uses Thompson Sampling with Beta Bernoulli distribution to solve multi-armed bandit problem
    :param variants: number of variants to be tested
    :param trials: number of tests that will be run
    :return: tuple of (success vector, failure vector, best variant)
    """
    means = [random.random() for i in range(variants)]
    beta_probs = [0.0] * variants
    successes = [0] * variants
    fails = [0] * variants

    print('True means: ')
    print(means)
    for trial in range(trials):
        for i in range(variants):
            # update beta probability for each variant as tests are run
            beta_probs[i] = random.betavariate(successes[i] + 1, fails[i] + 1)

        # get variant with the highest probability to test in this trial
        variant = max(range(len(beta_probs)), key=beta_probs.__getitem__)

        # test if conversion is a success based on true probability of the variant
        conversion = random.choices([0, 1], weights=[1 - means[variant], means[variant]])[0]

        # if success, add to successes for that variant, otherwise added to fails
        successes[variant] += conversion
        fails[variant] += 1 - conversion

    # get the best variant after all the trials are complete
    best_variant = max(range(len(beta_probs)), key=beta_probs.__getitem__)
    return successes, fails, best_variant


if __name__ == "__main__":
    wins, losses, best = main()
    print("Wins: {}".format(wins))
    print("Win Sum: {}".format(sum(wins)))
    print("Losses: {}".format(losses))
    print("Loss Sum: {}".format(sum(losses)))
    print("Best Variant: {}".format(best))
