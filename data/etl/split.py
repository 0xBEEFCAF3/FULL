#!/usr/bin/env python3
import json

FILE_PATH = './clean/clean_data.json'
OUTPUT_PATH = './interim/split_data.json'


def main():
    with open(OUTPUT_PATH, 'w') as split_file:
        with open(FILE_PATH, 'r') as raw_data_file:
            raw_data = json.loads(raw_data_file.read())
            split_file.write(json.dumps(raw_data[int(len(raw_data) * .8):]))


if __name__ == "__main__":
    main()
