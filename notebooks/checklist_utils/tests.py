from typing import List
import pandas as pd
from checklist.test_types import INV, MFT
import re
from checklist.perturb import Perturb
from checklist.editor import Editor
from checklist.expect import Expect


COIN_NAMES = [
    "bitcoin",
    "ethereum",
    "ripple",
    "tether",
    "cardano",
    "stellar",
    "dogecoin",
]

COIN_CODES = ["btc", "xbt", "eth", "xrp", "usdt", "ada", "xlm", "doge"]


def change_coin(x) -> List[str]:
    ret: List[str] = []

    for coin in COIN_NAMES:
        if re.search(r"\b%s\b" % coin, x):
            ret.extend(
                [
                    re.sub(r"\b%s\b" % coin, another_coin, x)
                    for another_coin in COIN_NAMES
                    if coin != another_coin
                ]
            )

    return ret


def get_coin_invariance_test(data: pd.DataFrame, num_samples: int = 200) -> INV:

    filter_coins = r"\b(" + "|".join(COIN_NAMES) + r")\b"
    filter_codes = r"\b(" + "|".join(COIN_CODES) + r")\b"

    # Select sampels only with coin names
    samples_with_coin = data[data.str.contains(filter_coins, case=False)]

    # Remove samples with coin codes to avoid conflicts
    # when coin name was changed and coin code wasn't
    samples_with_coin_wout_currency_code = samples_with_coin[
        ~samples_with_coin.str.contains(filter_codes, case=False)
    ]

    samples = samples_with_coin_wout_currency_code.sample(
        n=num_samples, random_state=42
    )

    ret = Perturb.perturb(
        samples.str.lower().values,
        change_coin,
        keep_original=True,
    )

    return INV(
        ret.data,
        name="Switch coin name.",
        capability="NER",
        description="Switching coin name schould not change the prediction.",
    )


def get_not_negative_test() -> MFT:

    verb = ["drop", "fall", "dip", "plunge"]

    editor = Editor()

    t = editor.template(
        "{coin} {benot} {verb} below 39000$.",
        coin=COIN_NAMES,
        benot=["does not", "doesn't", "will not", "won't"],
        verb=verb,
        save=True,
        remove_duplicates=True,
    )

    is_not_0 = lambda x, pred, *args: pred != 0

    return MFT(
        t.data,
        expect=Expect.single(is_not_0),
        templates=t.templates,
        name="Simple negation: not negative.",
        capability="Negation",
        description="Negation of negative statement schould be positive or neutral.",
    )


def get_simple_negation_test() -> MFT:

    adj = ["legal", "legitimate", "stable", "accessible", "regulated", "secure", "safe"]

    editor = Editor()

    t = editor.template(
        "{coin} {benot} {adj}.",
        coin=COIN_NAMES,
        benot=["is not", "isn't"],
        adj=adj,
        save=True,
        remove_duplicates=True,
    )

    return MFT(
        t.data,
        labels=0,
        templates=t.templates,
        name="Simple negation: negative samples.",
        capability="Negation",
        description="Simple negations of positive statements.",
    )


def get_punctuation_test(data: List[str], n_samples: int = 500) -> INV:
    t = Perturb.perturb(data, Perturb.punctuation, nsamples=n_samples)

    return INV(
        t.data,
        name="Punctuation.",
        capability="Robustness",
        description="Add or removes punctuation.",
    )


def get_typos_test(data: List[str], n_samples: int = 500) -> INV:
    t = Perturb.perturb(data, Perturb.add_typos, nsamples=n_samples, typos=1)

    return INV(
        t.data,
        name="Typos.",
        capability="Robustness",
        description="Add one typo to input by swapping two adjacent characters.",
    )


def get_contractions_test(data: List[str], n_samples: int = 500) -> INV:
    t = Perturb.perturb(data, Perturb.contractions, nsamples=n_samples)

    return INV(
        t.data,
        name="Contractions.",
        capability="Robustness",
        description="Contract or expand contractions.",
    )


def get_change_names_test(data: List[str], n_samples: int = 500) -> INV:
    t = Perturb.perturb(data, Perturb.change_names, nsamples=n_samples)

    return INV(
        t.data,
        name="Change names.",
        capability="NER",
        description="Replace names with other common names.",
    )


def get_change_locations_test(data: List[str], n_samples: int = 500) -> INV:
    t = Perturb.perturb(data, Perturb.change_location, nsamples=n_samples)

    return INV(
        t.data,
        name="Change locations.",
        capability="NER",
        description="Replace city or country names with other cities or countries.",
    )
