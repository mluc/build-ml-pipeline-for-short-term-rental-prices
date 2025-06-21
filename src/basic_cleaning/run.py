#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import wandb
import pandas as pd
from pathlib import Path
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run_name = f"basic_cleaning_{uuid.uuid4().hex[:8]}"
    logger.info(f"Running basic_cleaning with run_name: {run_name}")
    run = wandb.init(job_type="basic_cleaning", name=run_name)
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading input artifact %s", artifact_local_path)
    df = pd.read_csv(artifact_local_path)

    # Drop outliers according to the price column
    logger.info("Filtering rows with price not in [%s, %s]", args.min_price, args.max_price)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting 'last_review' column to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save cleaned data to a csv file
    df.to_csv("clean_sample.csv", index=False)

    # Create an artifact and log it to W&B
    logger.info("Uploading cleaned data as artifact %s", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    # Ensure the artifact is uploaded before finishing the run
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully qualified name of the raw dataset artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the newly created, cleaned dataset artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type for the cleaned dataset artifact.",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A short description for the cleaned dataset artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price allowed (rows with smaller price will be dropped)",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price allowed (rows with larger price will be dropped)",
        required=True
    )


    args = parser.parse_args()

    go(args)
