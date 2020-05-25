#!/usr/bin/env python3

import sys
import functools
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import List, Optional, Callable

import click
# from coinbase.wallet.client import Client
import coinbasepro as cbpro
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

WINDOW_LEN = 5

INITIAL_WALLET_EUR = 1.0


def create_model() -> keras.Sequential:
    model = keras.Sequential([
        layers.Dense(5, activation=tf.nn.tanh, input_shape=[WINDOW_LEN]),
        layers.Dense(5, activation=tf.nn.elu),
        layers.Dense(3, activation=tf.nn.elu),
        layers.Dense(3, activation=tf.nn.elu),
        layers.Dense(3, activation=tf.nn.elu),
        layers.Dense(1, activation=tf.nn.relu),
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


class OracleException(Exception):
    pass


class Oracle(ABC):
    @abstractmethod
    def feed_point(self, price: float) -> None:
        pass

    @abstractmethod
    def can_predict(self) -> bool:
        pass

    def predict(self) -> float:
        '''Return the estimated price of the coin in the nest step.

        @raise OracleException if there's not enought data to make a prediction
        '''
        if not self.can_predict():
            raise OracleException(
                "Doen't have enought data to make a prediction")

        return self._predict()

    @abstractmethod
    def _predict(self) -> float:
        '''Where the actual logic of the prediction lies. Only called after verifying
        if a prediction can be made

        '''


class OracleModel(Oracle):
    def __init__(self, model: keras.Model):
        self._hist = np.empty((1, WINDOW_LEN))
        self._hist_idx = 0
        self._model = model

    # Override
    def feed_point(self, price: float) -> None:
        if self._hist_idx < WINDOW_LEN:
            self._hist[0][self._hist_idx] = price

        else:
            self._hist[0][0:WINDOW_LEN - 1] = self._hist[0][1:WINDOW_LEN]
            self._hist[0][WINDOW_LEN-1] = price

        self._hist_idx += 1

    # Override
    def can_predict(self) -> bool:
        return self._hist_idx >= WINDOW_LEN

    # Override
    def _predict(self) -> Optional[float]:
        ret = self._model.predict(self._hist)[0]
        return ret


class OracleSimPerfect(Oracle):
    def __init__(self, hist: List[float]):
        self._hist = hist
        self._step_index = 0

    # Override
    def feed_point(self, price: float) -> None:
        self._step_index += 1

    # Override
    def can_predict(self) -> bool:
        return True

    # Override
    def _predict(self) -> Optional[float]:
        if self._step_index >= len(self._hist):
            return 0.0

        return self._hist[self._step_index]


class TradeActor():
    def __init__(self, oracle: Oracle, eur_funds=0.0):
        self._wallet_eth = 0.0
        self._wallet_eur = eur_funds
        self._oracle = oracle
        self._price_eth2eur = 0

    def step(self, eth2eur_price: float):
        self._price_eth2eur = eth2eur_price
        self._oracle.feed_point(eth2eur_price)

        if not self._oracle.can_predict():
            # click.echo(f'skip!    not enough data')
            return

        price_current = eth2eur_price
        price_prediction = self._oracle.predict()
        price_should_rise = price_current < price_prediction
        if price_should_rise:
            self._wallet_eth += self._wallet_eur * (1 / price_current)
            self._wallet_eur = 0.0
            # click.echo(f'buying!  Net Worth: {self.value}€')

        else:
            self._wallet_eur += self._wallet_eth * price_current
            self._wallet_eth = 0.0
            # click.echo(f'selling! Net Worth: {self.value}€')

    @property
    def value(self) -> float:
        '''Value in euros of all wallets'''
        return self._wallet_eur + (self._wallet_eth * self._price_eth2eur)


@click.group()
def cli():
    pass


def rates_preprocess(rates) -> pd.DataFrame:
    '''
    @returns A DataFrame with fields: time, low, high, open, close and volume
    '''

    rates = [(datetime.fromtimestamp(time), low, high, open, close, volume)
             for time, low, high, open, close, volume in rates]

    SERIES = ['time', 'low', 'high', 'open', 'close', 'volume']
    series = {
        name: pd.Series((row[i] for row in rates))
        for i, name in enumerate(SERIES)
    }

    rates = pd.DataFrame(series)
    return rates


def rates_save(rates: pd.DataFrame, path: str):
    rates.to_pickle(path)


def rates_load(path: str) -> pd.DataFrame:
    rates = pd.read_pickle(path)

    return rates


def rates2data(rates: pd.DataFrame):
    """Transforms the historical rates into data ready for training or simulation

    @returns A tupple of the data and labels

    """

    rates['mid'] = (rates['high'] + rates['low']) / 2

    prices = rates.pop('open')

    data = np.array(
        [prices[i:i + WINDOW_LEN] for i in range(len(prices) - WINDOW_LEN)])

    labels = prices[WINDOW_LEN:]

    return data, labels


@cli.command()
def download_data():
    """Downloads coinbase market history data"""

    client = cbpro.PublicClient()

    delta = timedelta(seconds=60 * 300)
    now = datetime.now()
    start = datetime(2019, 1, 1)
    end = start + delta
    rates = []
    while end < now:
        time_start = datetime.now()

        click.echo(f'Downloading: {start} -> {end}')
        rates_partial = client.get_product_historic_rates(
            'ETH-USD',
            start=start,
            end=end,
            granularity=60,
        )

        if isinstance(rates_partial, dict):
            message = rates_partial['message']

            if message == 'Slow rate limit exceeded':
                time.sleep(10)
                client = cbpro.PublicClient()
                continue

            click.echo(f"Unkown error: {message}")
            sys.exit(1)

        rates_partial.reverse()
        rates.extend(rates_partial)

        start = end
        end = start + delta

        time_end = datetime.now()
        time_diff = time_end - time_start
        time_to_sleep = 0.334 - time_diff.total_seconds()
        # NOTE: to avoid rate limits
        if time_to_sleep > 0.0:
            time.sleep(time_to_sleep)

    rates = rates_preprocess(rates)
    rates_save(rates, './data/ETH-EUR.pkl')
    click.echo("Done")


@cli.command()
def train():
    """Trains a simple model using the downloaded data"""

    rates = rates_load('./data/ETH-EUR.pkl')
    data, labels = rates2data(rates)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    model = create_model()
    model.fit(
        data,
        labels,
        epochs=500,
        validation_split=0.2,
        verbose=2,
        batch_size=512,
        callbacks=[early_stop],
    )

    model.save_weights('./model.ckpt')


def run_simulation_step(actor_construtor: Callable[[], TradeActor], prices):
    actor = actor_construtor(prices)

    initial_value = actor.value
    for i, price in enumerate(prices):
        actor.step(price)

    day_profit = actor.value - initial_value
    day_profit = round(day_profit, 4)

    return day_profit


def run_simulation(
        actor_construtor: Callable[[], TradeActor],
        prices,
        step_len=24 * 60,  # default is number of prices per day
):

    step_num = len(prices) // step_len
    results = [None] * step_num
    profits = np.empty(len(results))
    with Pool() as pool:
        for step_idx in range(step_num):
            prices_start = step_idx * step_len
            prices_end = step_idx * step_len + step_len
            prices_step = prices[prices_start:prices_end].reset_index(
                drop=True)

            results[step_idx] = pool.apply_async(
                run_simulation_step,
                (actor_construtor, prices_step),
            )

        for i, result in enumerate(results):
            day_profit = result.get()
            profits[i] = day_profit

            click.echo(f"Simulation progress: {i:3}/{len(results):3} days")

    profits = pd.Series(profits)
    return profits


def _actor_model_construtor(prices) -> TradeActor:
    model = create_model()
    model.load_weights('./model.ckpt')
    oracle = OracleModel(model)
    actor = TradeActor(oracle, INITIAL_WALLET_EUR)
    return actor


def _actor_perfect_construtor(prices) -> TradeActor:
    return TradeActor(OracleSimPerfect(prices), INITIAL_WALLET_EUR)


@cli.command()
def simulate():
    rates = rates_load('./data/ETH-EUR.pkl')
    rates['mid'] = (rates['high'] + rates['low']) / 2
    prices = rates.pop('open')

    profits = run_simulation(_actor_model_construtor, prices.copy())
    model_profit = profits.mean()

    profits = run_simulation(_actor_perfect_construtor, prices.copy())
    validation_profit = profits.mean()

    model_effectiveness = round((model_profit / validation_profit) * 100, 3)

    click.echo()
    click.echo(f"Simulation Results (Day avg of {len(profits)} days)")
    click.echo(f"Model profit: {model_profit}€")
    click.echo(f"Perfect profit: {validation_profit}€")
    click.echo(f"Model effectiveness: {model_effectiveness:3.6}%")


if __name__ == '__main__':
    cli()
