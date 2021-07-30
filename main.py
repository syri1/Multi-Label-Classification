import argparse
from os import path
from load_dataset.preprocess import transform_df, load_data
from src.run_experiment import train, test
from sklearn.model_selection import train_test_split
from src.evaluate import evaluate


def main(args):
    # Loading the data into a DataFrame

    df = load_data(args.path, args.mode)

    if args.mode == "train-test":
        train_set, test_set = train_test_split(
            df, test_size=args.test_size, random_state=42)
        # training
        X_train, y_train = transform_df(
            train_set, "train", args.update_hostname_category_map)

        train(X_train, y_train)

        # testing
        y_test = test_set[['target']]
        X_test = test_set.drop('target', axis=1)
        X_test, _ = transform_df(
            X_test, "test", args.update_hostname_category_map)

        y_pred = test(X_test, args.prediction_threshold)
        evaluate(y_test, y_pred)
    else:
        X, y = transform_df(df, args.mode, args.update_hostname_category_map)

        # train and test
        if args.mode == "train":
            train(X, y)
        elif args.mode == "test":
            test(X, args.prediction_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",  choices=["train", "test", "train-test"], default="train")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="test-size between 0 and  1 if mode is train-test")
    parser.add_argument("--prediction_threshold", type=float, default=0.27)
    parser.add_argument("--path", type=str, default='./',
                        help="path to the directory containing the parquet file")
    parser.add_argument("--update_hostname_category_map", type=int, default=-1)
    args = parser.parse_args()
    main(args)
