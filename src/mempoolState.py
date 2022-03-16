import asyncio
import requests
from enum import Enum
import datetime as dt
import json


class BLOCK_RESOURCE(Enum):
    MEMPOOL_SIZE = 'https://api.blockchain.info/charts/mempool-size?timespan=5days&sampled=true&metadata=false&cors=true&format=json'
    MEMPOOL_GROWTH_RATE = 'https://api.blockchain.info/charts/mempool-growth?timespan=5days&sampled=true&metadata=false&cors=true&format=json'
    NETWORK_DIFFICULTY = 'https://api.blockchain.info/charts/difficulty?timespan=5days&sampled=true&metadata=false&cors=true&format=json'
    AVERAGE_CONFIRMATION_TIME = 'https://api.blockchain.info/charts/avg-confirmation-time?timespan=5days&sampled=true&metadata=false&cors=true&format=json'
    MEDIAN_CONFIRMATION_TIME = 'https://api.blockchain.info/charts/median-confirmation-time?timespan=5days&sampled=true&metadata=false&cors=true&format=json'
    MINER_REVENUE = 'https://api.blockchain.info/charts/miners-revenue?timespan=30days&sampled=true&metadata=false&cors=true&format=json'
    TOTAL_HASH_RATE = 'https://api.blockchain.info/charts/hash-rate?daysAverageString=7D&timespan=1year&sampled=true&metadata=false&cors=true&format=json'
    MARKET_PRICE = 'https://api.blockchain.info/charts/market-price?timespan=30days&sampled=true&metadata=false&cors=true&format=json'
    DAY_OF_WEEK = 'foo',
    HOUR_OF_DAY = 'bar',


class MempoolState():
    def __init__(self, logging):
        self.logging = logging
        self.state = {}
        self.growth_rate = None
        self.network_difficulty = None
        self.average_confirmation_time = None
        self.median_confirmation_time = None
        self.day_of_week = None
        self.hour_of_day = None
        self.loop = asyncio.get_event_loop()

    def get_resource(self, resource_uri):
        return requests.get(resource_uri).json()["values"][-1]["y"]

    def get_resources(self):
        for resource in BLOCK_RESOURCE:
            if isinstance(resource.value, str):
                self.state[resource.name] = self.get_resource(resource.value)
        self.state[BLOCK_RESOURCE.DAY_OF_WEEK.name] = dt.datetime.today().weekday()
        self.state[BLOCK_RESOURCE.HOUR_OF_DAY.name] = dt.datetime.now().hour

    async def handle(self):
        self.logging.info('[Mempool State]: Starting gather mempool status')
        self.get_resources()
        self.logging.info(
            '[Mempool State]: Current mempool state ' + json.dumps(self.state))
        await asyncio.sleep(20)
        asyncio.ensure_future(self.handle())

    def start(self):
        print('Starting event loop')
        self.loop.create_task(self.handle())

    def stop(self):
        self.loop.stop()
