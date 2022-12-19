<img src="https://user-images.githubusercontent.com/24356537/157892419-413b35ab-6302-4255-8145-0915edf3442a.png" width="100" height="100" /> 

# FULL: BTC Mempool Predictor 

### Data-driven Mempool Prediction Model

#### In short
It is essential for simple sender-recipient transactions to have on-chain success metrics, such as fees, to ensure their success. For more complex interactive transactions and smart contracts, on-chain success metrics for the near future, up to 12,000 blocks, are also necessary. This is particularly relevant for scenarios such as a multi-sig transaction requiring signatures from multiple hardware wallets, or the use of protocols such as coinjoin, whirlpool, or BIP78, which involve collaboration between multiple parties over an extended period of time to construct private transactions. Additionally, the collaborative closing of lightning channels and the use of commitment transactions also require the consideration of on-chain success metrics.

Project FULL aims to utilize past mempool and Bitcoin network data to accurately predict on-chain success metrics for the current and near-future state of the bitcoin mempool.

#### What Determines On-Chain Success? 

Transactions (tx) enter the mempool contending for space on the next block. There are select features that determine a successful tx. Mainly the fee, mempool size, hash rate distribution, and hash rate. Txs that fail to provide sufficient fees at a time of high network activities get “stuck” in the mempool [[0]](https://bitlaunch.io/blog/bitcoin-transaction-stuck-on-mempool-heres-why-and-what-you-can-do/).



### Getting Started

#### Start virtual env

- `source .venv/bin/activate`
- `pip3 install -r requirments.txt`

#### Pull Data

- <a href="https://git-lfs.github.com/"> install git lfs </a>
- `git lfs pull`
