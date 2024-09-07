from collections import deque
import gc

import matplotlib.pyplot as plt
import numpy as np


class Environment:
    def __init__(
        self,
        agent,
        data,
        commission_for_train=0.5,
        commission_for_val=0,
        start_money=100_000,
        start_actives_count=0,
    ):
        self.start_money = start_money
        self.agent = agent
        self.start_actives_count = start_actives_count
        self.cur_actives_count = start_actives_count
        self.commission_for_train = commission_for_train
        self.commission_for_val = commission_for_val
        self.cur_money = start_money
        self.data = data
        self.money_memory = deque(maxlen=502)

    def make_move(self, move, price, commission):
        if move[0] and self.cur_actives_count > 0:
            self.cur_money += (
                price * self.cur_actives_count * (1 - commission / 100)
            )
            self.cur_actives_count = 0
        elif move[2]:
            to_buy = self.cur_money // (4 * price)
            self.cur_money -= to_buy * price * (1 + commission / 100)
            self.cur_actives_count += to_buy

    def calculate_reward(self, cur_price, first_price):
        cur_actives_growth = (
            self.cur_money + self.cur_actives_count * cur_price
        ) / self.start_money

        money_500_steps_ago = self.start_money
        if len(self.money_memory) > 500:
            money_500_steps_ago = self.money_memory[-500]
        cur_actives_growth_last_500_steps = (
            self.cur_money + self.cur_actives_count * cur_price
        ) / money_500_steps_ago
        cur_actives_growth_div_price = (
            (self.cur_money + self.cur_actives_count * cur_price) * first_price
        ) / (self.start_money * cur_price)
        return np.log(
            0.35 * cur_actives_growth
            + 0.2 * cur_actives_growth_last_500_steps
            + 0.45 * cur_actives_growth_div_price
        )

    def init_state_for_batch(self):
        self.cur_money = self.start_money
        self.cur_actives_count = self.start_actives_count
        self.money_memory = deque(maxlen=502)
        gc.collect()

    def train_agent(self, batch_size=10_000):
        batch_n = 1
        total_batches = self.data.train_batches_count(
            batch_size=batch_size
        )
        for batch, open_prices in self.data.train_batches_generator(
            batch_size=batch_size
        ):

            self.init_state_for_batch()

            first_price = float(open_prices[0])
            state_old = self.agent.get_state(
                batch[0],
                first_price,
                self.cur_money,
                self.cur_actives_count,
                first_price,
            )

            for i in range(1, len(batch)):
                cur_price = float(open_prices[i])

                move = self.agent.get_action(state_old)
                self.make_move(move, cur_price, self.commission_for_train)

                self.money_memory.append(self.cur_money)
                reward = self.calculate_reward(cur_price, first_price)

                state_new = self.agent.get_state(
                    batch[i],
                    cur_price,
                    self.cur_money,
                    self.cur_actives_count,
                    first_price,
                )
                self.agent.remember(state_old, move, reward, state_new)
                state_old = state_new.detach().clone()
                if i % 1000 == 0:
                    gc.collect()
                    print_list = [
                        f"\rTraining...Batch№:{batch_n}",
                        f"out of {total_batches}",
                        f"done {batch_n * 100 / total_batches: .2f}%",
                        f"Rows {i * 100 / len(batch):.2f}%",
                        f"Money growth: {(self.cur_money + self.cur_actives_count * cur_price) / self.start_money: .2f}",
                        f"Money growth div price: {((self.cur_money + self.cur_actives_count * cur_price) * first_price)/ (self.start_money * cur_price): .2f}",
                    ]
                    print(
                        "   ".join(print_list),
                        end="",
                        flush=True,
                    )

            self.agent.train_short_memory(batch_size=batch_size)
            self.agent.train_long_memory()
            batch_n += 1

    def val_agent(self):
        plt.ion()
        fig, ax = plt.subplots()
        (line1,) = ax.plot(
            [], [], lw=2, label="Money Growth", color="blue"
        )  # First line (sin curve)
        (line2,) = ax.plot(
            [], [], lw=2, label="Price Growth", color="red"
        )  # Second line (cos curve)
        plt.legend(loc="best")
        plt.show()
        batch_n = 1
        total_batches = self.data.val_batches_count()
        all_money_growth = []
        all_price_growth = []
        for batch, open_prices in self.data.val_batches_generator():
            money_growth = []
            price_growth = []
            cur_num = []
            self.init_state_for_batch()

            first_price = float(open_prices[0])

            for i in range(len(batch) - 1):
                cur_price = float(open_prices[i])
                next_price = float(open_prices[i + 1])
                state = self.agent.get_state(
                    batch[i],
                    cur_price,
                    self.cur_money,
                    self.cur_actives_count,
                    first_price,
                )
                move = self.agent.get_action(state, add_randomness=False)
                self.make_move(move, next_price, self.commission_for_val)
                if i % 1000 == 0:
                    gc.collect()
                    print_list = [
                        f"\rValidating.. Batch№:{batch_n}",
                        f"out of {total_batches}",
                        f"done {batch_n * 100 / total_batches: .2f}%",
                        f"Rows {i * 100 /len(batch):.2f}%",
                        f"Money growth:{(self.cur_money + self.cur_actives_count * cur_price) / self.start_money:.2f}"
                        f"Money growth div price:{((self.cur_money + self.cur_actives_count * cur_price) * first_price)/ (self.start_money * cur_price): .2f}",
                    ]
                    print(
                        "   ".join(print_list),
                        end="",
                        flush=True,
                    )
                    money_growth.append(
                        (self.cur_money + self.cur_actives_count * cur_price)
                        / self.start_money
                    )
                    price_growth.append(cur_price / first_price)
                    cur_num.append(i / 1000)
                    line1.set_xdata(cur_num)
                    line1.set_ydata(money_growth)

                    line2.set_xdata(cur_num)
                    line2.set_ydata(price_growth)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()  #
            all_money_growth.append(money_growth)
            all_price_growth.append(price_growth)
            batch_n += 1
        return all_money_growth, all_price_growth
