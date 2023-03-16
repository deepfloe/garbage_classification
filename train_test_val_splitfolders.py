import splitfolders
import pathlib

if __name__ == "__main__":
    input_path = pathlib.Path("data/original_data")
    output_path = pathlib.Path("data/train_val_test_split")
    splitfolders.ratio(input_path, ratio=(0.5, 0.25, 0.25),output=output_path)