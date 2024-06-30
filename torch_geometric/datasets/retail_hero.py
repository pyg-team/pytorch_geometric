import os
import zipfile
from typing import Callable, Optional

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class RetailHero(InMemoryDataset):
    """The retailhero dataset
    https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    from the uplift modeling competition.
    A bipartite graph where the edges indicate user who buys a product.
    The causal information for the users includes treatments and outcome.
    The causal outcome for each node is defined by the change in average amount
    of money spent before and after treatment time.
    Edges are separated in before treatment and after (T=1 or T=0).
    The node features include demographics and consuming habbits.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = ("https://storage.yandexcloud.net/datasouls-ods/"
           "/materials/9c6913e5/retailhero-uplift.zip")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # os.makedirs(extract_to, exist_ok=True)
        local_filename = self.url.split("/")[-1]

        download_url(f"{self.url}", self.raw_dir)
        # Unzip the file
        with zipfile.ZipFile(f"{self.raw_dir}/{local_filename}",
                             "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        # Remove the downloaded zip file
        os.remove(f"{self.raw_dir}/{local_filename}")

    def process(
        self,
        train_ind_file: str = "data/uplift_train.csv",
        feature_file: str = "data/clients.csv",
        purchases_file: str = "data/purchases.csv",
        features_file: str = "retailhero_features.csv",
        edge_index_file: str = "retailhero_graph.csv",
        age_filter: int = 16,
    ):

        encoder = OneHotEncoder()
        train = pd.read_csv(f"{self.raw_dir}/{train_ind_file}").set_index(
            "client_id")

        df_features = pd.read_csv(f"{self.raw_dir}/{feature_file}")

        df_features["first_redeem_date"] = pd.to_datetime(
            df_features["first_redeem_date"])
        df_features["first_issue_abs_time"] = (
            pd.to_datetime(df_features["first_issue_date"]) -
            pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        df_features["first_redeem_abs_time"] = (
            pd.to_datetime(df_features["first_redeem_date"]) -
            pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        df_features["redeem_delay"] = (df_features["first_redeem_abs_time"] -
                                       df_features["first_issue_abs_time"])

        df_features = df_features[df_features["age"] > age_filter]
        df_features = df_features[df_features["redeem_delay"] > 0]
        df_features = df_features.reset_index(drop=True)

        one_hot_encoded = encoder.fit_transform(df_features[["gender"]])
        one_hot_encoded_array = one_hot_encoded.toarray()
        encoded_categories = encoder.categories_

        df_encoded = pd.DataFrame(one_hot_encoded_array,
                                  columns=encoded_categories[0])
        df_features = df_features.drop("gender", axis=1)

        columns = list(df_features.columns) + list(encoded_categories[0])
        df_features = pd.concat([df_features, df_encoded], axis=1,
                                ignore_index=True)
        df_features.columns = columns

        df_features = train.join(df_features.set_index("client_id"))

        df_features = df_features[~df_features.age.isna()]

        # Use the purchase list to take the extra features and define network
        purchases = pd.read_csv(f"{self.raw_dir}/{purchases_file}")
        purchases = purchases[[
            "client_id",
            "transaction_id",
            "transaction_datetime",
            "purchase_sum",
            "store_id",
            "product_id",
            "product_quantity",
        ]]
        purchases["transaction_datetime"] = pd.to_datetime(
            purchases["transaction_datetime"])
        purchases["transaction_abs_time"] = (
            purchases["transaction_datetime"] -
            pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        # remove users that are not part of the experiment
        dictionary = dict(
            zip(df_features.index, df_features["first_redeem_date"]))
        purchases["first_redeem_date"] = purchases["client_id"].map(dictionary)
        purchases = purchases[~purchases["first_redeem_date"].isna()]
        dictionary = dict(zip(df_features.index, df_features["treatment_flg"]))
        purchases["treatment_flg"] = purchases["client_id"].map(dictionary)

        client_map = {j: i for i, j in enumerate(purchases.client_id.unique())}
        product_map = {
            j: i
            for i, j in enumerate(purchases.product_id.unique())
        }
        store_map = {j: i for i, j in enumerate(purchases.store_id.unique())}

        # separate before and after treatment redeem
        ind = purchases["transaction_datetime"] < purchases[
            "first_redeem_date"]
        purchases_before = purchases[ind]
        purchases_after = purchases[~ind]

        # calculate metrics on average over client and transactions
        features_purchases_before = (
            purchases_before.groupby("transaction_id").agg({
                "client_id":
                "first",
                "purchase_sum":
                "first",
                "transaction_datetime":
                "first",
            }).reset_index())
        features_purchases_before.columns = [
            "transaction_id",
            "client_id",
            "purchase_sum",
            "transaction_datetime",
        ]
        features_purchases_before = features_purchases_before.groupby(
            "client_id").agg({
                "purchase_sum": "mean",
                "transaction_id": "count",
                "transaction_datetime": ["max", "min"],
            })
        features_purchases_before.columns = [
            "avg_money_before",
            "total_count_before",
            "last_purchase_before",
            "first_purchase_before",
        ]
        features_purchases_before[
            "avg_count_before"] = features_purchases_before[
                "total_count_before"] / (
                    (features_purchases_before["last_purchase_before"] -
                     features_purchases_before["first_purchase_before"]
                     ).dt.days + 1)
        features_purchases_before = features_purchases_before[[
            "avg_money_before", "avg_count_before"
        ]]

        labels_purchases_after = (
            purchases_after.groupby("transaction_id").agg({
                "client_id":
                "first",
                "purchase_sum":
                "first",
                "transaction_datetime":
                "first",
            }).reset_index())
        labels_purchases_after.columns = [
            "transaction_id",
            "client_id",
            "purchase_sum",
            "transaction_datetime",
        ]
        labels_purchases_after = labels_purchases_after.groupby(
            "client_id").agg({
                "purchase_sum": "mean",
                "transaction_id": "count",
                "transaction_datetime": ["max", "min"],
            })
        labels_purchases_after.columns = [
            "avg_money_after",
            "total_count_after",
            "last_purchase_after",
            "first_purchase_after",
        ]
        labels_purchases_after["avg_count_after"] = labels_purchases_after[
            "total_count_after"] / (
                (labels_purchases_after["last_purchase_after"] -
                 labels_purchases_after["first_purchase_after"]).dt.days + 1)
        labels_purchases_after = labels_purchases_after[[
            "avg_money_after", "avg_count_after"
        ]]

        purchases_before["client_id"] = purchases_before["client_id"].map(
            client_map)
        purchases_before["product_id"] = purchases_before["product_id"].map(
            product_map)
        purchases_before["store_id"] = purchases_before["store_id"].map(
            store_map)
        purchases_after["client_id"] = purchases_after["client_id"].map(
            client_map)
        purchases_after["product_id"] = purchases_after["product_id"].map(
            product_map)
        purchases_after["store_id"] = purchases_after["store_id"].map(
            store_map)

        purchases_before["label"] = 0
        purchases_after["label"] = 1

        purchases_before = (purchases_before.groupby(
            ["client_id", "product_id",
             "label"]).sum("product_quantity").reset_index())
        purchases_after = (purchases_after.groupby(
            ["client_id", "product_id",
             "label"]).sum("product_quantity").reset_index())

        purchase_processed = pd.concat([purchases_before, purchases_after])
        purchase_processed = purchase_processed[[
            "client_id", "product_id", "label", "product_quantity"
        ]]
        purchase_processed.columns = ["user", "product", "T", "weight"]
        purchase_processed.to_csv(f"{self.processed_dir}/{edge_index_file}",
                                  index=False)

        degrees = (purchase_processed[(purchase_processed["T"] == 0
                                       )].groupby("user").size().reset_index())
        degrees = dict(zip(degrees["user"], degrees[0]))

        weighted_degrees = (purchase_processed[(purchase_processed["T"] == 0)].
                            groupby("user").sum("weight").reset_index())

        weighted_degrees = dict(
            zip(weighted_degrees["user"], weighted_degrees["weight"]))

        # add targets
        data = (df_features.join(features_purchases_before).join(
            labels_purchases_after).fillna(0))

        data["avg_money_change"] = data["avg_money_after"] - data[
            "avg_money_before"]
        data["avg_count_change"] = data["avg_count_after"] - data[
            "avg_count_before"]
        data = data[data.index.isin(
            purchases.client_id.unique())].reset_index()

        data["client_id"] = data["client_id"].map(client_map)

        data["degree_before"] = data["client_id"].map(degrees).fillna(0)
        data["weighted_degree_before"] = (
            data["client_id"].map(weighted_degrees).fillna(0))

        treatment = ["treatment_flg"]
        labels = [
            "target",
            "avg_money_change",
            "avg_count_change",
            "avg_money_after",
            "avg_count_after",
        ]
        features = [
            "age",
            "F",
            "M",
            "U",
            "first_issue_abs_time",
            "first_redeem_abs_time",
            "redeem_delay",
            "avg_money_before",
            "avg_count_before",
            "degree_before",
            "weighted_degree_before",
        ]

        data = data[treatment + labels + features]

        data.to_csv(f"{self.processed_dir}/{features_file}", index=False)
        edge_index_df = pd.read_csv(f"{self.processed_dir}/{edge_index_file}")
        features = pd.read_csv(f"{self.processed_dir}/{features_file}")

        columns_to_norm = [
            "age",
            "first_issue_abs_time",
            "first_redeem_abs_time",
            "redeem_delay",
            "degree_before",
            "weighted_degree_before",
        ]
        if len(columns_to_norm) > 0:
            normalized_data = StandardScaler().fit_transform(
                features[columns_to_norm])
            features[columns_to_norm] = normalized_data

        data = HeteroData()
        data["user", "buys", "product"] = {
            "edge_index":
            torch.tensor(edge_index_df[["user", "product"
                                        ]].values).type(torch.LongTensor).T,
            "treatment":
            torch.tensor(edge_index_df["T"].values).type(torch.BoolTensor),
        }

        data["user"] = {
            "x":
            torch.tensor(features[[
                "age",
                "F",
                "M",
                "U",
                "first_issue_abs_time",
                "first_redeem_abs_time",
                "redeem_delay",
            ]].values).type(torch.FloatTensor),
            "t":
            torch.tensor(features["treatment_flg"].values).type(
                torch.LongTensor),
            "y":
            torch.tensor(features["avg_money_change"].values).type(
                torch.FloatTensor),
        }
        data["products"] = {
            "num_products": len(edge_index_df["product"].unique())
        }

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
