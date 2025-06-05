import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import joblib

from rdkit import Chem
from RxnPred.model import RxnPredModel, RxnPredDataset
from train import getModelInputs
from RxnPred.getTfRecord import GetTfRecord


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def run_prediction(input_csv, output_csv, model_dir, isotonic_model_path, batch_size=64, rm_tfrecord=True):
    tf.get_logger().setLevel('ERROR')
    time_start = datetime.now()

    # Load input CSV
    df = pd.read_csv(input_csv)
    tfrecord_path = os.path.splitext(input_csv)[0] + ".tfrecord"

    # Validate input data
    invalid_rows = []
    for i, row in df.iterrows():
        smi1, smi2 = row["SMILES1"], row["SMILES2"]
        if not (is_valid_smiles(smi1) and is_valid_smiles(smi2)):
            invalid_rows.append(i)
    if invalid_rows:
        print(f"Found {len(invalid_rows)} invalid SMILES pairs. Marked as ERROR.")
    df["score"] = "PENDING"
    df.loc[invalid_rows, "score"] = "ERROR"
    df_valid = df.drop(index=invalid_rows).copy().reset_index(drop=True)
    df_error = df.loc[invalid_rows].copy().reset_index(drop=True)
    if df_valid.empty:
        print("No valid SMILES found. Exiting.")
        df.to_csv(output_csv, index=False)
        return

    # Generate TFRecord if not exist
    if os.path.exists(tfrecord_path):
        print("TFRecord exists.")
    else:
        GetTfRecord(dataframe=df_valid, save_name=os.path.splitext(input_csv)[0], is_label=False, is_preset=False)

    print("Load dataset...")
    dataset = RxnPredDataset(filenames=tfrecord_path, batch_size=batch_size, training=False).get_iterator()
    print("Load dataset OK!")

    print("Load model...")
    models = glob.glob(os.path.join(model_dir, "*.json"))
    for model_str in models:
        model_name = os.path.splitext(os.path.basename(model_str))[0].split("Performance_")[1]

        with open(model_str, "r") as json_file:
            parameters = json.load(json_file)

        params = {
            "gconv_units": [parameters["num_gconv_units"]] * parameters["num_gconv_layers"],
            "gconv_regularizer": tf.keras.regularizers.L2(parameters["weight_decay"]),
            "dense_units": [parameters["num_dense_units"]] * parameters["num_dense_layers"],
            "dense_dropout": parameters["dense_dropout"],
        }

        model = RxnPredModel(**params)
        model_weights = os.path.join(model_dir, f"model_{model_name}.ckpt")
        model.load_weights(model_weights)

        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        isotonic_model = joblib.load(isotonic_model_path)

        print("Predicting...")
        num_batches = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        pbar = tqdm(total=num_batches, bar_format="{l_bar}{bar:50}{r_bar}", dynamic_ncols=True)

        y_preds = []
        for batch in dataset:
            inputs = getModelInputs(batch, is_structure=True, is_reaction=True)
            probability = probability_model(inputs, training=False)[:, 1].numpy()
            probability = isotonic_model.transform(probability)
            probability[np.isnan(probability)] = 1
            y_preds.append(probability)
            pbar.update()

        y_preds = np.concatenate(y_preds).round(4)
        df_valid["score"] = y_preds
        print("\nPrediction complete.")

    if rm_tfrecord and os.path.exists(tfrecord_path):
        os.remove(tfrecord_path)

    df = pd.concat([df_valid, df_error], ignore_index=True)
    df.to_csv(output_csv, index=False)
    time_end = datetime.now()
    print("Time:", time_end - time_start)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reaction prediction on input CSV.")
    parser.add_argument('--input', default='demo.csv', help='Input CSV file with SMILES1 and SMILES2')
    parser.add_argument('--output', default='demo_out.csv', help='Output CSV file path')
    parser.add_argument('--model_dir', default='./Model/graph_structure_reaction/', help='Directory containing model files (.json and .ckpt)')
    parser.add_argument('--isotonic_model', default='./Model/graph_structure_reaction/isotonic_model.joblib', help='Path to isotonic regression model (.joblib)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction')
    parser.add_argument('--keep_tfrecord', action='store_true', help='Keep temporary TFRecord file')

    args = parser.parse_args()

    run_prediction(
        input_csv=args.input,
        output_csv=args.output,
        model_dir=args.model_dir,
        isotonic_model_path=args.isotonic_model,
        batch_size=args.batch_size,
        rm_tfrecord=not args.keep_tfrecord,
    )
