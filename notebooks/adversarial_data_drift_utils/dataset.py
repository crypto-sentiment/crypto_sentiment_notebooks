import pandas as pd


def prepare_data_for_adv_validation(
    path_to_old_data: str,
    path_to_new_data: str,
    text_col_name: str = "title",
    label_col_name: str = "is_new",
) -> pd.DataFrame:
    """
    Prepares data for adversarial validation. Text column `text_col_name` stays the same.
    The label column `label_col_name` is a binary one – whether a records comes from
    the new dataset `path_to_new_data` (1) or the old dataset `path_to_old_data` (0).

    :param path_to_old_data: path to a CSV file with old data, e.g. to the train data
    :param path_to_old_data: path to a CSV file with old data, e.g. to the train data
    """

    old_df = pd.read_csv(path_to_old_data)
    new_df = pd.read_csv(path_to_new_data)

    old_df[label_col_name] = 0
    new_df[label_col_name] = 1

    df = pd.concat(
        [
            old_df[[text_col_name, label_col_name]],
            new_df[[text_col_name, label_col_name]],
        ]
    ).reset_index(drop=True)

    return df
