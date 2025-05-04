import json
from collections import defaultdict


def count_frequencies_with_custom_equal(lst, equal_func, weights=None, **kwargs):
    if weights is None:
        weights = [1 for i in range(len(lst))]
    assert len(weights) == len(lst)
    frequency_dict = defaultdict(float)
    
    for item, weight in zip(lst, weights):
        found = False
        for key in frequency_dict.keys():
            if equal_func(item, key, **kwargs):
                frequency_dict[key] += weight
                found = True
                break
        if not found:
            frequency_dict[item] += weight
    return sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)    

if __name__ == "__main__":
    def custom_equal(a, b):
        return a.lower() == b.lower()

    my_list = ['apple', 'banana', 'Apple', 'orange', 'Banana', 'banana']
    weights = [0.2, 0.3, 0.1, 0.6, 0.2, 0.1]
    frequencies = count_frequencies_with_custom_equal(my_list, custom_equal, weights)

    print(frequencies)
    print(json.dumps(frequencies))

