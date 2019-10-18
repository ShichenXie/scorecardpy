# -*- coding: utf-8 -*-

from .monotonic import (_FeatureMonotone,_BinBadRate,_BadRateMonotone,_AssignBin)
import numpy as np

def monotone_merge(df, target, col):
    '''
    :return:
    将数据集df中，不满足坏样本率单调性的变量col进行合并，使得合并后的新的变量中，坏样本率单调.

    '''
    def _merge_matrix(m, i,j,k):
        '''
        :param m: 需要合并行的矩阵
        :param i,j: 合并第i和j行
        :param k: 删除第k行
        :return: 合并后的矩阵
        '''
        m[i, :] = m[i, :] + m[j, :]
        m = np.delete(m, k, axis=0)
        return m

    def _merge_adjacent_rows(i, bad_by_bin_current, bins_list_current, not_monotone_count_current):
        '''
        :param i: 需要将第i行与前、后的行分别进行合并，比较哪种合并方案最佳。判断准则是，合并后非单调性程度减轻.
        :param bad_by_bin_current:合并前的分箱矩阵，包括每一箱的样本个数、坏样本个数和坏样本率
        :param bins_list_current: 合并前的分箱方案
        :param not_monotone_count_current:合并前的非单调性元素个数
        :return:分箱后的分箱矩阵、分箱方案、非单调性元素个数和衡量均匀性的指标balance
        '''
        i_prev = i - 1
        i_next = i + 1
        bins_list = bins_list_current.copy()
        bad_by_bin = bad_by_bin_current.copy()
        #合并方案a：将第i箱与前一箱进行合并
        bad_by_bin2a = _merge_matrix(bad_by_bin.copy(), i_prev, i, i)
        bad_by_bin2a[i_prev, -1] = bad_by_bin2a[i_prev, -2] / bad_by_bin2a[i_prev, -3]
        not_monotone_count2a = _FeatureMonotone(bad_by_bin2a[:, -1])['count_of_nonmonotone']
        # 合并方案b：将第i行与后一行进行合并
        bad_by_bin2b = _merge_matrix(bad_by_bin.copy(), i, i_next, i_next)
        bad_by_bin2b[i, -1] = bad_by_bin2b[i, -2] / bad_by_bin2b[i, -3]
        not_monotone_count2b = _FeatureMonotone(bad_by_bin2b[:, -1])['count_of_nonmonotone']
        # balance = ((bad_by_bin[:, 1] / N).T * (bad_by_bin[:, 1] / N))[0, 0]
        balance_a = ((bad_by_bin2a[:, 1] / N).T * (bad_by_bin2a[:, 1] / N))[0, 0]
        balance_b = ((bad_by_bin2b[:, 1] / N).T * (bad_by_bin2b[:, 1] / N))[0, 0]
        # 满足下述2种情况时返回方案a：（1）方案a能减轻非单调性而方案b不能；（2）方案a和b都能减轻非单调性，但是方案a的样本均匀性优于方案b
        if not_monotone_count2a < not_monotone_count_current and not_monotone_count2b >= not_monotone_count_current or \
                                        not_monotone_count2a < not_monotone_count_current and not_monotone_count2b < not_monotone_count_current and balance_a < balance_b:
            bins_list[i_prev] = bins_list[i_prev] + bins_list[i]
            bins_list.remove(bins_list[i])
            bad_by_bin = bad_by_bin2a
            not_monotone_count = not_monotone_count2a
            balance = balance_a
        # 同样地，满足下述2种情况时返回方案b：（1）方案b能减轻非单调性而方案a不能；（2）方案a和b都能减轻非单调性，但是方案b的样本均匀性优于方案a
        elif not_monotone_count2a >= not_monotone_count_current and not_monotone_count2b < not_monotone_count_current or \
                                        not_monotone_count2a < not_monotone_count_current and not_monotone_count2b < not_monotone_count_current and balance_a > balance_b:
            bins_list[i] = bins_list[i] + bins_list[i_next]
            bins_list.remove(bins_list[i_next])
            bad_by_bin = bad_by_bin2b
            not_monotone_count = not_monotone_count2b
            balance = balance_b
        # 如果方案a和b都不能减轻非单调性，返回均匀性更优的合并方案
        else:
            if balance_a< balance_b:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
            else:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
        return {'bins_list': bins_list, 'bad_by_bin': bad_by_bin, 'not_monotone_count': not_monotone_count,
                'balance': balance}


    N = df.shape[0]
    [badrate_bin, bad_by_bin] = _BinBadRate(df, col, target)
    bins = list(bad_by_bin[col])
    bins_list = [[i] for i in bins]
    badRate = sorted(badrate_bin.items(), key=lambda x: x[0])
    badRate = [i[1] for i in badRate]
    not_monotone_count, not_monotone_position = _FeatureMonotone(badRate)['count_of_nonmonotone'], _FeatureMonotone(badRate)['index_of_nonmonotone']
    # 迭代地寻找最优合并方案，终止条件是:当前的坏样本率已经单调，或者当前只有2箱
    while (not_monotone_count > 0 and len(bins_list)>2):
        # 当非单调的箱的个数超过1个时，每一次迭代中都尝试每一个箱的最优合并方案
        all_possible_merging = []
        for i in not_monotone_position:
            merge_adjacent_rows = _merge_adjacent_rows(i, np.mat(bad_by_bin), bins_list, not_monotone_count)
            all_possible_merging.append(merge_adjacent_rows)
        balance_list = [i['balance'] for i in all_possible_merging]
        not_monotone_count_new = [i['not_monotone_count'] for i in all_possible_merging]
        # 如果所有的合并方案都不能减轻当前的非单调性，就选择更加均匀的合并方案
        if min(not_monotone_count_new) >= not_monotone_count:
            best_merging_position = balance_list.index(min(balance_list))
        # 如果有多个合并方案都能减轻当前的非单调性，也选择更加均匀的合并方案
        else:
            better_merging_index = [i for i in range(len(not_monotone_count_new)) if not_monotone_count_new[i] < not_monotone_count]
            better_balance = [balance_list[i] for i in better_merging_index]
            best_balance_index = better_balance.index(min(better_balance))
            best_merging_position = better_merging_index[best_balance_index]
        bins_list = all_possible_merging[best_merging_position]['bins_list']
        bad_by_bin = all_possible_merging[best_merging_position]['bad_by_bin']
        not_monotone_count = all_possible_merging[best_merging_position]['not_monotone_count']
        not_monotone_position = _FeatureMonotone(bad_by_bin[:, 3])['index_of_nonmonotone']
    return bins_list


def monotonous_bin(df,col,cutOffPoints,target,special_values):

    #单调性检验
    var_cutoff = {}
    col1 = col + '_Bin'  # 检验单调性
    df[col1] = df[col].map(lambda x: _AssignBin(x, cutOffPoints=cutOffPoints,
                                                     special_attribute=special_values))
    BRM = _BadRateMonotone(df, col1, target, special_attribute=special_values)  # 是否单调
    if not BRM:
        # 合并方案
        if special_values == []:
            bin_merged = monotone_merge(df, target, col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin) > 1:
                    indices = [int(b.replace('Bin ', '')) for b in bin]
                    removed_index = removed_index + indices[0:-1]
            removed_point = [cutOffPoints[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints.remove(p)
            var_cutoff[col] = cutOffPoints
        else:
            cutOffPoints2 = [i for i in cutOffPoints if i not in special_values]
            temp = df.loc[~df[col].isin(special_values)]
            bin_merged = monotone_merge(temp, target, col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin) > 1:
                    indices = [int(b.replace('Bin ', '')) for b in bin]
                    removed_index = removed_index + indices[0:-1]
            removed_point = [cutOffPoints2[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints2.remove(p)
            cutOffPoints2 = cutOffPoints2 + special_values
            var_cutoff[col] = cutOffPoints2  # 单调性检验结果
    return var_cutoff
