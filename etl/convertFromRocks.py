#!/usr/bin/env python3

import json
import rocksdb
import glob

DB_PATH = '../data/'
DB_GLOB = '/*.rocks'
OUTPUT_PATH = './raw_data.json'


def convert_to_json_file():

    accumulated_list = []
    file_list = glob.glob(DB_PATH + DB_GLOB)
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

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(accumulated_list, f)


if __name__ == "__main__":
    convert_to_json_file()
