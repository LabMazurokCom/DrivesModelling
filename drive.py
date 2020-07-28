import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib


class DriveCore:
    def __init__(self, id, epoch):
        self.size = self.set_size()
        self.drive_id = id
        self.epoch = epoch
        self.required_replicators = self.set_required_replicators()

    def set_size(self):
        return int(np.round(np.random.triangular(10, 50, 1000)))

    def set_required_replicators(self):
        return int(np.round(np.random.triangular(4, 10, 20)))


class Drive:
    def __init__(self, driveCore: DriveCore):
        self.drive_core = driveCore
        #self.used_replicators = 0
        self.replicators = set()

    def get_replicators_left(self):
        return self.drive_core.required_replicators - self.get_used_replicators()

    def get_rewards(self):
        if self.get_used_replicators() < self.get_min_replicators():
            return 0
        return self.drive_core.size * self.drive_core.required_replicators / self.get_used_replicators()

    def is_full(self):
        return self.drive_core.required_replicators == self.get_used_replicators()

    def get_min_replicators(self):
        return int(np.round(2 * self.drive_core.required_replicators / 3))

    def get_required_replicators(self):
        return self.drive_core.required_replicators

    def get_used_replicators(self):
        return len(self.replicators)

    def add_replicator(self, replicator_id):
        self.replicators.add(replicator_id)

    def has_min_replicators(self):
        return 3 * self.get_used_replicators() > 2 * self.drive_core.required_replicators

    def get_size(self):
        return self.drive_core.size

    def get_epoch(self):
        return self.drive_core.epoch

    def get_drive_id(self):
        return self.drive_core.drive_id

    def get_replicators(self):
        return self.replicators

    def clear_replicators(self):
        self.replicators.clear()

class DrivesFactory:
    def __init__(self, channels):
        self.drives_created = 0
        self.drive_channels = [dict() for _ in range(channels)]
        self.drive_channels_infinite = [dict() for _ in range(channels)]

    def generate_drives(self, epoch):
        m = 100
        for i in range(np.random.poisson(m)):
            drive_core = DriveCore(self.drives_created, epoch)
            for channel, channel_infinite in zip(self.drive_channels, self.drive_channels_infinite):
                drive = Drive(drive_core)
                channel[self.drives_created] = drive
                channel_infinite[self.drives_created] = drive
            self.drives_created += 1

    def remove_drive(self, channel, drive_id):
        del self.drive_channels[channel][drive_id]

    def get_drive_channels(self):
        return self.drive_channels

    def get_drives_infinite(self, channel):
        return self.drive_channels_infinite[channel]


class ReplicatorCore:
    def __init__(self, id):
        self.space = self.set_space()
        self.id = id

    def set_space(self):
        return int(np.round(np.random.uniform(500, 4000)))


class Replicator:
    def __init__(self, replicator_core: ReplicatorCore):
        self.drives = set()
        self.created_requests = 0
        self.accepted_requests = 0
        self.replicator_core = replicator_core
        self.used_space = 0
        self.rewards = 0

    def get_space(self):
        return self.replicator_core.space

    def get_id(self):
        return self.replicator_core.id

    def get_rewards(self):
        return self.rewards

    def generate_dynamic_requests(self, drives):
        drive_keys, dp, take = self.dynamic(drives)
        drives_left = len(drive_keys)
        w = self.get_space() - self.used_space
        space_left = w
        requests = []
        while drives_left > 0 and space_left > 0:
            if take[drives_left][space_left]:
                requests.append(drive_keys[drives_left - 1])
                space_left -= drives[drive_keys[drives_left - 1]].get_size()
            drives_left -= 1
        self.created_requests += len(requests)
        return requests

    def get_dynamic_rewards(self, drive):
        r_min = drive.get_min_replicators()
        r_max = drive.get_required_replicators()
        r = drive.get_used_replicators()
        return drive.get_size() * min((r + 1) / r_min, 1) * r_max / max(r_min, r + 1)

    def dynamic(self, drives):
        w = self.get_space() - self.used_space
        drive_keys = self.filter_drives(drives)
        dp = np.zeros((len(drive_keys) + 1, w + 1))
        take = np.zeros(dp.shape, dtype=bool)
        for h, drive_id in enumerate(drive_keys):
            drive = drives[drive_id]
            i = h + 1
            cur_size = drive.get_size()
            for j in range(w + 1):
                if j < cur_size:
                    dp[i][j] = dp[i - 1][j]
                    take[i][j] = False
                else:
                    add = self.get_weight_value(drive)
                    if dp[i - 1][j] > dp[i - 1][j - cur_size] + add:
                        dp[i][j] = dp[i - 1][j]
                        take[i][j] = False
                    else:
                        dp[i][j] = dp[i - 1][j - cur_size] + add
                        take[i][j] = True
        return drive_keys, dp, take

    def generate_dynamic_random_requests(self, drives):
        drive_keys, dp, take = self.dynamic(drives)
        drives_left = len(drive_keys)
        space_left = self.get_space() - self.used_space
        sizes = []
        while drives_left > 0 and space_left > 0:
            if take[drives_left][space_left]:
                sizes.append(drives[drive_keys[drives_left - 1]].get_size())
                space_left -= drives[drive_keys[drives_left - 1]].get_size()
            drives_left -= 1
        unique_sizes, count_sizes = np.unique(sizes, return_counts=True)
        clusters = [[] for _ in range(len(unique_sizes))]
        for drive_id in drive_keys:
            drive = drives[drive_id]
            cluster = np.searchsorted(unique_sizes, drive.get_size(), 'left')
            if cluster < len(clusters):
                clusters[cluster].append(drive_id)
        requests = []
        for cluster, count in zip(clusters, count_sizes):
            weights = []
            for drive_id in cluster:
                weights.append(self.get_weight_value(drives[drive_id]))
            weights = np.array(weights)
            if np.allclose(weights, 0):
                weights = np.ones(len(weights))
            chosen_drives = np.random.choice(cluster, min(count, len(cluster)), replace=False,
                                             p=weights / np.sum(weights))
            requests += list(chosen_drives)
        self.created_requests += len(requests)
        return requests

    def generate_random_requests(self, drives):
        drive_keys = self.filter_drives(drives)
        volume = self.replicator_core.space - self.used_space
        requests = []
        while len(drive_keys) > 0:
            candidate_drive_keys = []
            for drive_key in drive_keys:
                drive = drives[drive_key]
                if drive.get_size() <= volume:
                    candidate_drive_keys.append(drive_key)
            if len(candidate_drive_keys) > 0:
                weights = np.ones(len(candidate_drive_keys))
                request_arg = np.random.choice(len(weights), 1, p=weights / np.sum(weights)).item()
                requests.append(candidate_drive_keys[request_arg])
                volume -= drives[candidate_drive_keys[request_arg]].get_size()
                del candidate_drive_keys[request_arg]
            drive_keys = candidate_drive_keys
        self.created_requests += len(requests)
        return requests

    def generate_random_weighted_requests(self, drives):
        drive_keys = self.filter_drives(drives)
        volume = self.replicator_core.space - self.used_space
        requests = []
        while len(drive_keys) > 0:
            candidate_drive_keys = []
            for drive_key in drive_keys:
                drive = drives[drive_key]
                if drive.get_size() <= volume:
                    candidate_drive_keys.append(drive_key)
            if len(candidate_drive_keys) > 0:
                weights = np.array([self.get_weight_value(drives[drive_key]) for drive_key in candidate_drive_keys])
                if np.allclose(weights, 0):
                    weights = np.ones(len(weights))
                request_arg = np.random.choice(len(weights), 1, p=weights / np.sum(weights)).item()
                requests.append(candidate_drive_keys[request_arg])
                volume -= drives[candidate_drive_keys[request_arg]].get_size()
                del candidate_drive_keys[request_arg]
            drive_keys = candidate_drive_keys
        self.created_requests += len(requests)
        return requests

    def generate_requests(self, drives, mode):
        if mode == 'dynamic':
            return self.generate_dynamic_requests(drives)
        elif mode == 'dynamic_random':
            return self.generate_dynamic_random_requests(drives)
        elif mode == 'random':
            return self.generate_random_requests(drives)
        else:
            return self.generate_random_weighted_requests(drives)

    def filter_drives(self, drives):
        drive_keys = []
        for drive_key, drive_value in drives.items():
            if drive_key not in self.drives:
                drive_keys.append(drive_key)
        return drive_keys

    def accept_drive(self, drive):
        self.drives.add(drive.get_drive_id())
        self.used_space += drive.get_size()
        self.accepted_requests += 1

    def update_rewards(self, drives):
        self.rewards = 0
        for drive_id in self.drives:
            self.rewards += drives[drive_id].get_rewards()
        return self.rewards

    def get_weight_value(self, drive: Drive):
        r_max = drive.get_required_replicators()
        r_min = drive.get_min_replicators()
        r = drive.get_used_replicators()
        if r < r_min:
            return (r + 1) / r_min
        else:
            return (r_max - r) / (r_min * (r_max - r_min))

    def remove_drive(self, drive):
        drive_id = drive.get_drive_id()
        if drive_id in self.drives:
            self.used_space -= drive.get_size()
            self.drives.remove(drive_id)

class ReplicatorsFactory:
    def __init__(self, channels, n):
        self.replicators_channels = [[] for _ in range(channels)]
        for i in range(n):
            core = ReplicatorCore(i)
            for channel in self.replicators_channels:
                channel.append(Replicator(core))

    def get_replicators(self):
        return self.replicators_channels

class Statistics:
    def __init__(self, pref):
        # self.epochs_for_min_replicators = []
        # self.epochs_for_required_replicators = []
        self.replicators_accepted_ratio = []
        # self.replicators_rewards = []
        self.drives_with_min_replicators = []
        self.drives_with_required_replicators = []
        # self.requests_accept_ratio = []
        self.mean_rewards = []
        self.occupied_space_ratio = []
        self.useful_occupied_space_ratio = []
        # self.requests_made = 0
        # self.requests_accepted = 0
        # self.drives_number = 0
        self.pref = pref

    # def add_epochs_for_min_replicators(self, epochs):
    #     self.epochs_for_min_replicators.append(epochs)
    #
    # def add_epochs_for_required_replicators(self, epochs):
    #     self.epochs_for_required_replicators.append(epochs)

    # def add_replicators_stats(self, replicators):
    #     for replicator in replicators:
    #         self.replicators_accepted_ratio.append(replicator.accepted_requests / replicator.created_requests)
    #         self.replicators_rewards.append(replicator.get_rewards())

    def finish_epoch(self, replicators, drives):
        # self.drives_with_min_replicators.append(len(self.epochs_for_min_replicators))
        # self.drives_with_required_replicators.append(len(self.epochs_for_required_replicators))
        # self.requests_accept_ratio.append(self.requests_accepted / self.requests_made if self.requests_made != 0 else 0)
        # self.mean_rewards.append(np.mean(self.replicators_rewards))

        min_replicators = 0
        max_replicators = 0
        rewards = 0
        for drive_id, drive in drives.items():
            if drive.has_min_replicators():
                min_replicators += 1
            if drive.is_full():
                max_replicators += 1
            rewards += drive.get_rewards() * drive.get_used_replicators()

        made_requests = 0
        accepted_requests = 0
        occupied_space_ratio = []
        useful_occupied_space_ratio = []
        for replicator in replicators:
            accepted_requests += replicator.accepted_requests
            made_requests += replicator.created_requests
            useful_space = 0
            for drive_id in replicator.drives:
                drive = drives[drive_id]
                if drive.has_min_replicators():
                    useful_space += drive.get_size()
            occupied_space_ratio.append(replicator.used_space / replicator.get_space())
            if replicator.used_space > 0:
                useful_occupied_space_ratio.append(useful_space / replicator.used_space)

        self.drives_with_min_replicators.append(min_replicators)
        self.drives_with_required_replicators.append(max_replicators)
        self.replicators_accepted_ratio.append(accepted_requests / made_requests)
        self.mean_rewards.append(rewards / len(replicators))
        self.occupied_space_ratio.append(np.mean(occupied_space_ratio))
        self.useful_occupied_space_ratio.append(np.mean(useful_occupied_space_ratio))

        # self.requests_made = 0
        # self.requests_accepted = 0
        # self.replicators_accepted_ratio.clear()
        # self.replicators_rewards.clear()

    @staticmethod
    def plt_bar(ax, array, y_range=None):
        if y_range is not None:
            ax.set_ylim(y_range[0], y_range[1])
        ax.bar(range(len(array)), array)

    @staticmethod
    def plt_plots(ax, arrays, labels, y_range=None):
        if y_range is not None:
            ax.set_ylim(y_range[0], y_range[1])
        x = range(len(arrays[0]))
        for array, label in zip(arrays, labels):
            if len(x) != len(array):
                print('LOL')
            ax.plot(x, array, 'o-', label=label)
        ax.legend()

    @staticmethod
    def finish_statistics(statistics_channels, epoch):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        matplotlib.rc('font', **font)
        labels = [statistics.pref for statistics in statistics_channels]
        fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(25, 25))
        Statistics.plt_plots(axs[0], [statistics.drives_with_min_replicators for statistics in statistics_channels], labels)
        Statistics.plt_plots(axs[1], [statistics.drives_with_required_replicators for statistics in statistics_channels], labels)
        Statistics.plt_plots(axs[2], [statistics.replicators_accepted_ratio for statistics in statistics_channels], labels)
        Statistics.plt_plots(axs[3], [statistics.mean_rewards for statistics in statistics_channels], labels)
        Statistics.plt_plots(axs[4], [statistics.occupied_space_ratio for statistics in statistics_channels], labels)
        Statistics.plt_plots(axs[5], [statistics.useful_occupied_space_ratio for statistics in statistics_channels], labels)
        # for j, statistics in enumerate(statistics_channels):
        #     Statistics.plt_bar(axs[0][j], statistics.drives_with_min_replicators)
        #     Statistics.plt_bar(axs[1][j], statistics.drives_with_required_replicators)
        #     Statistics.plt_bar(axs[2][j], statistics.requests_accept_ratio, y_range=(0, 1))
        #     Statistics.plt_bar(axs[3][j], statistics.mean_rewards)
        rows = ['Have Min Replicators', 'Have Max Replicators', 'Request Accept Ratio', 'Mean Rewards',
                'Space Occupied Ratio', 'Useful Space Ratio']
        # for ax, col in zip(axs[0], cols):
        #     ax.set_title(col)
        for ax, row in zip(axs, rows):
            ax.set_ylabel(row.replace(' ', '\n'), rotation=0, size='large')
        # for ax, col in zip(axs[0], cols):
        #     ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
        #                 xycoords='axes fraction', textcoords='offset points',
        #                 size='large', ha='center', va='baseline')
        #
        # for ax, row in zip(axs[:, 0], rows):
        #     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
        #                 xycoords=ax.yaxis.label, textcoords='offset points',
        #                 size='large', ha='right', va='center')
        plt.savefig('res/stats_epoch_{}.png'.format(epoch))
        plt.close(fig)

    def update_drives_number(self, number):
        self.drives_number = number

    # def notify_request_made(self):
    #     self.requests_made += 1
    #
    # def notify_request_accepted(self):
    #     self.requests_accepted += 1
    #
    # def draw_min_replicators(self, epoch):
    #     self.draw_replicators(self.epochs_for_min_replicators, 'Min', len(self.epochs_for_min_replicators), epoch)
    #
    # def draw_required_replicators(self, epoch):
    #     self.draw_replicators(self.epochs_for_required_replicators, 'Required', len(self.epochs_for_required_replicators), epoch)

    # def draw_replicators(self, hist_data, mode, number, epoch):
    #     fig, ax = plt.subplots()
    #     ax.hist(hist_data, bins=100, range=(1, 100))
    #     ax.text(0.5, 0.4, 'Drives In Network {}'.format(self.drives_number), color='red', fontsize=20,
    #             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #     ax.text(0.5, 0.6, 'Drives With {} Replicators {}'.format(mode, number), color='red', fontsize=20,
    #             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #     plt.savefig('res/{}/{}Replicators/block {}'.format(self.pref, mode, epoch))
    #     plt.close(fig)

    # def draw_requests_ratio(self, epoch):
    #     fig, ax = plt.subplots()
    #     ax.hist(self.replicators_accepted_ratio, range=(0, 1))
    #     ax.text(0.5, 0.4, 'Made Requests {}'.format(self.requests_made), color='red', fontsize=20,
    #             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #     ax.text(0.5, 0.6, 'Accepted Requests {}'.format(self.requests_accepted), color='red', fontsize=20,
    #             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #     # plt.text(0.5, 0.5, 'New Requests {}'.format(made_requests), horizontalalignment='center', verticalalignment = 'center')
    #     plt.savefig('res/{}/Requests/block {}'.format(self.pref, epoch))
    #     plt.close(fig)

    # def draw_rewards(self, epoch):
    #     fig, ax = plt.subplots()
    #     ax.hist(self.replicators_rewards)
    #     ax.text(0.5, 0.5, 'Mean Rewards {}'.format(np.mean(self.replicators_rewards)), color='red', fontsize=20,
    #             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    #     # plt.text(0.5, 0.5, 'New Requests {}'.format(made_requests), horizontalalignment='center', verticalalignment = 'center')
    #     plt.savefig('res/{}/Rewards/block {}'.format(self.pref, epoch))
    #     plt.close(fig)

    # def draw_statistics(self, epoch):
    #     self.draw_min_replicators(epoch)
    #     self.draw_required_replicators(epoch)
    #     self.draw_requests_ratio(epoch)
    #     self.draw_rewards(epoch)

def accept_drive(replicator: Replicator, drive: Drive, epoch, statistics: Statistics):
    replicator.accept_drive(drive)
    #had_min_replicators = drive.has_min_replicators()
    #was_full = drive.is_full()
    drive.add_replicator(replicator.get_id())
    #has_min_replicators = drive.has_min_replicators()
    #is_full = drive.is_full()
    #if not had_min_replicators and has_min_replicators:
    #statistics.add_epochs_for_min_replicators(drive.get_epoch() - epoch + 1)
    #if not was_full and is_full:
    #statistics.add_epochs_for_required_replicators(drive.get_epoch() - epoch + 1)


def are_valid(replicator, drives, requests):
    sum = 0
    for drive_id in requests:
        sum += drives[drive_id].get_size()
    return replicator.get_space() >= replicator.used_space + sum

def should_del_drive(drive_id, drives, replicators, epoch):
    drive = drives[drive_id]
    age = epoch - drive.get_epoch() + 1
    epochs_to_live = 5
    if not drive.has_min_replicators() and age >= epochs_to_live:
        for replicator_id in drive.get_replicators():
            replicators[replicator_id].remove_drive(drive)
        return True


def main():
    np.random.seed(18)
    steps = 20
    replicators_n = 100
    request_functions = ['dynamic', 'dynamic_random', 'random', 'random_weighted']
    replicators_factory = ReplicatorsFactory(len(request_functions), replicators_n)
    drives_factory = DrivesFactory(len(request_functions))
    statistics_channels = [Statistics(function) for function in request_functions]
    for step in range(steps):
        print(step)
        drives_factory.generate_drives(step)
        for channel, (replicators, drives, statistics, request_function) in enumerate(zip(replicators_factory.get_replicators(),
            drives_factory.get_drive_channels(), statistics_channels, request_functions)):
            if step == 0 and channel == 0:
                need = 0
                for drive_id, drive in drives.items():
                    need += drive.get_size() * drive.get_required_replicators()
                have = 0
                for replicator in replicators:
                    have += replicator.get_space()
                # print('need {}'.format(need), 'have {}'.format(have))
            print(request_function)
            all_requests = dict()
            statistics.update_drives_number(drives_factory.drives_created)
            for replicator_i, replicator in enumerate(replicators):
                # print(replicator_i)
                requests = replicator.generate_requests(drives, request_function)
                for drive_id in requests:
                    if drive_id not in all_requests:
                        all_requests[drive_id] = []
                    all_requests[drive_id].append(replicator.get_id())
                    # statistics.notify_request_made()
            print(request_function, len(all_requests))
            for drive_id, drive in all_requests.items():
                need = drives[drive_id].get_replicators_left()
                random.shuffle(drive)
                for i in range(min(need, len(drive))):
                    accept_drive(replicators[drive[i]], drives[drive_id], step, statistics)
                    # statistics.notify_request_accepted()
                if drives[drive_id].is_full():
                    drives_factory.remove_drive(channel, drive_id)
            for replicator in replicators:
                replicator.update_rewards(drives_factory.get_drives_infinite(channel))
            #statistics.add_replicators_stats(replicators)
            # statistics.draw_statistics(step)
            statistics.finish_epoch(replicators, drives_factory.get_drives_infinite(channel))
            drives_to_del = []
            for drive_id, drive in drives.items():
                if should_del_drive(drive_id, drives, replicators, step):
                    drives_to_del.append(drive_id)
            for drive_id in drives_to_del:
                drives_factory.remove_drive(channel, drive_id)
        Statistics.finish_statistics(statistics_channels, step)
    return drives_factory, replicators_factory, statistics_channels


drives_factory, replicators_factory, statistics_channels = main()