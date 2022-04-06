#!/usr/bin/env python3

import json
import rocksdb
import glob
import argparse


def convert_to_json_file(db_path, db_glob, output_path):

    accumulated_list = []
    file_list = glob.glob(db_path + db_glob)
    for rocks_db_path in file_list:
        opts = rocksdb.Options()

        opts.create_if_missing = False
        opts.max_open_files = 300000
        opts.write_buffer_size = 67108864
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 67108864

        opts.table_factory = rocksdb.BlockBasedTableFactory(
            filter_policy=rocksdb.BloomFilterPolicy(10),
            block_cache=rocksdb.LRUCache(2 * (1024 ** 3)),
            block_cache_compressed=rocksdb.LRUCache(500 * (1024 ** 2)))
        db = rocksdb.DB(rocks_db_path, opts)
        it = db.iterkeys()
        it.seek_to_first()
        tx_ids = list(it)
        for _index, txid in enumerate(tx_ids):
            accumulated_list.append(json.loads(db.get(txid)))

    with open(output_path, 'w') as f:
        json.dump(accumulated_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbpath", type=str,
                        help="Path to data base objects")

    parser.add_argument("--dbglob", type=str,
                        help="regex pattern for objects to inside db path")

    parser.add_argument("--outputpath", type=str,
                        help="output path")
    args = parser.parse_args()

    print(args.dbpath)
    convert_to_json_file(args.dbpath, args.dbglob, args.outputpath)
