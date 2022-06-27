import pandas as pd
from typing import Dict
from sklearn.utils import shuffle


def read_assessment_data(data_path: str) -> pd.DataFrame:

    annotators: Dict[str, str] = {
        "A3KE73FIKEXFG9": "Yury",
        "A37V9Y0507BD97": "Lina",
        "A3SNZUC6ZCTWFF": "Victor",
        "ARZUW61D1HIHW": "Senya",
        "AUJFPBJZCAO6C": "Nikita",
        "AK8BJE69LD69K": "Unknown",
    }

    df = pd.concat(
        [
            pd.read_csv(
                f"{data_path}/20220417_Batch_353286_trial1_results.csv",
                index_col="HITId",
            ),
            pd.read_csv(
                f"{data_path}/20220417_Batch_353289_trial2_results.csv",
                index_col="HITId",
            ),
        ]
    )

    df["WorkerId"] = df["WorkerId"].map(annotators)

    df = df[df["WorkerId"] != "Unknown"]

    result_df = pd.crosstab(
        index=df["Input.content"],
        columns=df["WorkerId"],
        values=df["Answer.sentiment"],
        aggfunc=lambda x: x,  # weird, needs this aggfunc to be not None
    )

    annotators_names = [v for v in annotators.values() if v != "Unknown"]

    result_df["num_votes"] = result_df[annotators_names].notnull().sum(axis=1)
    result_df["unique_labels"] = (
        result_df[annotators_names]
        .apply(lambda x: sorted([el for el in set(x) if not pd.isnull(el)]), axis=1)
        .values
    )

    result_df["majority"] = result_df[annotators_names].mode(axis=1)[0]

    result_df.loc[
        (result_df["num_votes"] == 3) & (result_df["unique_labels"].apply(len) == 3),
        "majority",
    ] = "Controversial"

    result_df = result_df.loc[result_df["majority"] != "Controversial"]

    result_df = result_df.reset_index().rename(
        columns={"Input.content": "title", "majority": "sentiment"}
    )

    return result_df[["title", "sentiment"]].rename_axis(None, axis=1)


def read_data(data_path: str) -> pd.DataFrame:
    base_data = pd.read_csv(f"{data_path}/20190110_train_4500.csv")
    assessment_data = read_assessment_data(data_path)

    df = pd.concat([base_data, assessment_data])
    df = df[~df["title"].str.contains("?", regex=False)]

    df = shuffle(df.reset_index(drop=True))

    label_mapping = {"Negative": 0, "Positive": 2, "Neutral": 1}

    df["label"] = df["sentiment"].map(label_mapping)

    return df
