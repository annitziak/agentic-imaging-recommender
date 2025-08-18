import json
import re
import pickle
import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class PostFilteringAgent:
    """
    A class to perform post-filtering on medical abstracts based on the GRADE scale. The features used are inspired
    by other studies and we use a RandomForest as our ML algorithm.
    """
    def __init__(self, input_path="data/abstracts_dataset_post_filtering.json", sjr_path="data/SJR2024.csv"):
        self.input_path = input_path #where the supervised dataset is stored (contains the abstracts and information and also their SOE)
        self.sjr_path = sjr_path
        self.df = None # we will use a df for easier processing
        self.model = None
        self.X = None
        self.y = None
        self.mlb = MultiLabelBinarizer() # for the publication_type feature

    def _extract_grade_features(self, paper):
        """
        Extracts binary simple features from the abstract of a medical paper that are relevant for GRADE-style evaluation.
        Each feature represents the presence or absence of specific evidence-related phrases (binary), except the samples size.
        """
        features = {}
        abstract = paper.get("abstract", "").lower() if isinstance(paper.get("abstract", ""), str) else ""

        # Binary feature for patient outcomes (e.g., mortality, morbidity)
        features["mentions_patient_outcomes"] = int(any(w in abstract for w in [
            "mortality", "morbidity", "clinical outcome", "patient outcome"
        ]))

        # Binary feature for diagnostic accuracy metrics
        features["mentions_accuracy_metrics"] = int(any(w in abstract for w in [
            "sensitivity", "specificity", "auc", "roc", "positive predictive value",
            "negative predictive value", "likelihood ratio", "diagnostic accuracy"
        ]))

        # Binary feature for mentioning comparator/control
        features["mentions_comparator"] = int(any(w in abstract for w in [
            "compared with", "versus", "vs.", "reference standard", "gold standard",
            "comparison", "control group"
        ]))

        # Binary feature for mentioning of treatment or impact
        features["mentions_effective_treatment"] = int(any(w in abstract for w in [
            "treated with", "treatment", "therapy", "managed with",
            "impact on management", "impact on treatment"
        ]))

        # Binary feature for mentioning of blinding or masking
        features["mentions_blinding"] = int(any(w in abstract for w in [
            "blinded", "blind", "masked", "masking"
        ]))

        # Binary feature for mentioning of randomization
        features["mentions_randomization"] = int(any(w in abstract for w in [
            "randomized", "randomised", "random allocation", "randomly assigned"
        ]))

        # Binary feature for mentioning confidence intervals
        features["mentions_confidence_interval"] = int("confidence interval" in abstract or "ci" in abstract)

        # Binary feature for mentioning meta-analyses or systematic reviews
        features["mentions_meta_analysis"] = int(any(w in abstract for w in [
            "meta-analysis", "systematic review", "pooled analysis"
        ]))

        # Binary feature for mentioning funding
        features["mentions_funding"] = int(any(w in abstract for w in [
            "funded by", "supported by", "grant", "sponsored by"
        ]))

        # Extract sample size if reported, default to 0 if not found
        match = re.search(r"(n\s*=\s*|sample size of\s*)(\d+)", abstract)
        features["sample_size_reported"] = int(match.group(2)) if match else 0

        return features


    def _add_sjr_feature(self, feature_df):
        """
        Adds the SJR score to the input DataFrame by matching cleaned journal names
        using exact and fuzzy matching against the preprocessed SJR list.
        """

        def clean_journal_name(name):
            """ Cleans the journal name by removing subtitles and parenthetical phrases. """
            if pd.isna(name):
                return None
            name = name.lower()
            name = name.split(":")[0].strip()  # remove subtitles
            name = re.sub(r"\s*\(.*?\)", "", name).strip()  # remove parentheticals
            return name
        

        # Load cleaned SJR index 
        sjr = pd.read_csv(self.sjr_path)[["journal", "SJR"]]
        sjr_journals = sjr["journal"].tolist()

        # Clean journal names in the input
        feature_df["journal"] = feature_df["journal"].apply(clean_journal_name)

        # Try initial exact merge
        merged = pd.merge(feature_df, sjr, on="journal", how="left")
        # Identify unmatched journals BEFORE dropping 'journal'
        missing_mask = merged["SJR"].isna()
        missing_journals = merged.loc[missing_mask, "journal"].unique().tolist()

        # Fuzzy match the missing journals
        fuzzy_map = {
            j: process.extractOne(j, sjr_journals, scorer=fuzz.token_sort_ratio, score_cutoff=85)[0]
            if process.extractOne(j, sjr_journals, scorer=fuzz.token_sort_ratio, score_cutoff=85) else None
            for j in missing_journals
        }
        merged.loc[missing_mask, "journal_fuzzy"] = merged.loc[missing_mask, "journal"].map(fuzzy_map)

        # Merge fuzzy matches
        fuzzy_matched = pd.merge(
            merged[missing_mask],
            sjr.rename(columns={"journal": "journal_fuzzy"}),
            on="journal_fuzzy",
            how="left"
        )
        # Update original with fuzzy SJR where available
        merged.loc[missing_mask, "SJR"] = fuzzy_matched["SJR_y"].fillna(0).values
        merged.drop(columns=["journal", "journal_fuzzy"], inplace=True, errors="ignore")
        return merged


    def load_and_prepare_training_data(self):
        """ Loads and prepares the data from all the JSON files in the input path.
        Extracts relevant features and merges with the SJR data to create a DataFrame suitable for training.
        """
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        rows = []
        for paper in data:
            # All these columns need to be present in our dataset
            if not all(k in paper for k in ("study_quality", "journal", "publication_types", "year")):
                continue
            row = {
                "study_quality": paper["study_quality"],
                "journal": paper["journal"],
                "publication_types": paper["publication_types"],
                "year": paper["year"]
            }
            row.update(self._extract_grade_features(paper))
            rows.append(row)

        self.df = pd.DataFrame(rows).dropna().reset_index(drop=True)
        self.df = self._add_sjr_feature(self.df)
        self.df["SJR"] = self.df["SJR"].astype(str).str.replace(",", "").astype(float) #make sure SJR is float

    def train_model(self):
        """ Trains a Random Forest model on the prepared data.
        The model predicts study quality based on extracted GRADE features and SJR score.
        """

        # Convert publication types to binary features (multi-label binarization)
        pub_type_df = pd.DataFrame(self.mlb.fit_transform(self.df["publication_types"]), columns=self.mlb.classes_)

        # Concatenate the SJR score and binary publication types with the GRADE features for the model input and target as "study_quality"
        self.X = pd.concat([pub_type_df, self.df.drop(columns=["study_quality", "publication_types"])], axis=1)
        self.y = self.df["study_quality"].values


        # Train-test split of 0.2, stratified by study quality
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=30)
        self.model.fit(X_train, y_train)

        # Save the models to a file -> will be used in predict time
        with open("saved_models/soe_predictor_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        with open("saved_models/mlb.pkl", "wb") as f:
            pickle.dump(self.mlb, f)

        # Evaluate the model and print classification report
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    def predict_quality(self, papers: list[dict]):
        """
        Predicts study quality scores for new papers. Returns a list of dictionaries with paper details and predicted quality scores.
        This will be used by the Medical Review Agent for filtering."""

        rows = []
        valid_indices = []

        for i, paper in enumerate(papers):
            if not all(k in paper for k in ("abstract", "journal", "publication_types")):
                continue  # skip incomplete entries
            features = self._extract_grade_features(paper)
            row = {
            "journal": paper["journal"],
            "publication_types": paper["publication_types"],
            "year": paper.get("year", None), 
            **features
        }
            rows.append(row)
            valid_indices.append(i)

        # make df for easy processing
        df = pd.DataFrame(rows)
        df = self._add_sjr_feature(df)
        df["SJR"] = df["SJR"].astype(str).str.replace(",", "").astype(float)

        # Encode publication types -> load the mlb and just transform
        with open("saved_models/mlb.pkl", "rb") as f:
            self.mlb = pickle.load(f)

        pub_type_df = pd.DataFrame(self.mlb.transform(df["publication_types"]), columns=self.mlb.classes_)

        # Concatenate with features (excluding 'journal' and 'publication_types')
        input_df = pd.concat([pub_type_df, df.drop(columns=["publication_types"])], axis=1)

        # load the model and make predictions
        if not self.model:
            with open("saved_models/soe_predictor_model.pkl", "rb") as f:
                self.model = pickle.load(f)

        # make the prediction
        preds = self.model.predict(input_df)

        # Return aligned with input
        results = []
        for idx, pred in zip(valid_indices, preds):
            result = {
                "paper": papers[idx],
                "predicted_quality": pred #new in each json entry
            }
            results.append(result)

        return results

#maybe need to change main
if __name__ == "__main__":
    agent = PostFilteringAgent()
    agent.load_and_prepare_training_data()
    agent.train_model()