
from math import log2


def tree_generate(data):
    node = {}

    label_set = []
    for instance in data:
        label_set.append(instance[-1])
    if len(set(label_set)) == 1:
        node = label_set[0]
    elif len(data[0]) == 1:
        node = majority(label_set)
    else:
        best_feature = argmax_info_gain(data)
        best_feature_value_set = set()
        for instance in data:
            best_feature_value_set.add(instance[best_feature])

        for value in best_feature_value_set:

            sub_data = []
            for instance in data:
                if instance[best_feature] == value:
                    new_instance = instance[:best_feature]
                    new_instance.extend(instance[best_feature + 1:])
                    sub_data.append(new_instance)
            node[value] = tree_generate(sub_data)
    return node


def majority(lt: list):
    count = {}
    for i in lt:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
    value_max = max(count.values())
    key_max = None
    for i in count:
        if count[i] == value_max:
            key_max = i
    return key_max


def argmax_info_gain(data):
    gain_dict = {}

    feature_list = range(len(data[0][:-1]))
    for feature in feature_list:
        gain_dict[feature] = info_gain(data, feature)

    value_max = max(gain_dict.values())
    key_max = None
    for i in feature_list:
        if gain_dict[i] == value_max:
            key_max = i
            break
    return key_max


def info_gain(data, column):
    gain = info_entropy(data)

    feature_value_set = set()
    for instance in data:
        feature_value_set.add(instance[column])

    feature_value_dict = {}

    for feature_value in feature_value_set:
        feature_value_dict[feature_value] = 0
        for instance in data:
            if instance[column] == feature_value:
                feature_value_dict[feature_value] += 1

    num = sum(feature_value_dict.values())

    for feature_value in feature_value_set:
        sub_data = []
        for instance in data:
            if instance[column] == feature_value:
                sub_data.append(instance)
        gain -= feature_value_dict[feature_value] / num * info_entropy(sub_data)
    return gain


def info_entropy(data):
    label_set = []
    for instance in data:
        label_set.append(instance[-1])

    count = {}
    for i in label_set:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1

    num = sum(count.values())
    ent = 0
    for key in count.keys():
        p = count[key] / num
        ent -= p*log2(p)
    return ent
