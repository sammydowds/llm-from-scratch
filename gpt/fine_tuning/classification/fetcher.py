import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

class SMSSpamFetcher():
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        self.zip_path = "sms_spam_collection.zip"
        self.extracted_path = "sms_spam_collection"
        self.data_file_path = Path(self.extracted_path) / "SMSSpamCollection.tsv"

    def download(self):
        if self.data_file_path.exists():
            print(f"{self.data_file_path} already exists. Skipping download and extractions.")
            return

        with urllib.request.urlopen(self.url) as response:
            with open(self.zip_path, "wb") as out_file:
                out_file.write(response.read())

        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.extracted_path)
        
        original_file_path = Path(self.extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, self.data_file_path)
        print(f"File downloaded and saved as {self.data_file_path}")

    def read(self):
        return pd.read_csv(
            self.data_file_path, sep="\t", header=None, names=["Label", "Text"] 
        )

    def balance(self, df):
        num_spam = df[df["Label"] == "spam"].shape[0]
        ham_subset = df[df["Label"] == "ham"].sample(
            num_spam, random_state=123
        )
        balanced_df = pd.concat([
            ham_subset, df[df["Label"] == "spam"]
        ])
        return balanced_df
    
    def random_split(self, df, train_frac = 0.7, validation_frac = 0.1):
        df = df.sample(
            frac=1, random_state=123
        ).reset_index(drop=True)

        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)
        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]

        return train_df, validation_df, test_df
    
    def fetch_and_process(self):
        self.download()
        df = self.read()
        balanced_df = self.balance(df)
        balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

        train_df, validation_df, test_df = self.random_split(balanced_df, 0.7, 0.1)
        train_df.to_csv("train.csv", index=None)
        validation_df.to_csv("validation.csv", index=None)
        test_df.to_csv("test.csv", index=None)
