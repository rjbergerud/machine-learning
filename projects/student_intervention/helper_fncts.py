import pandas as pd
# import matplotlib.pyplot as plt

# category_name is a string that is the label of a column in data
# data is a pandas dataframe
def plot_success_by_category(category_name, data, plt, index):
    # plt.subplot(2,2,index)
    number_features = len(list(data.columns[:-1]))
    labels = data[category_name].unique()
    labels.sort()
    totals_by_label = [len(data[data[category_name] == label]) for label in labels]
    passed_by_label = [len(data[(data[category_name] == label) & (data["passed"] == "yes")]) for label in labels]

    perc_passed = [i/float(j)*100.0 for i,j in zip(passed_by_label, totals_by_label)]

    # Figure with single subplot
    # f, ax = plt.subplots()
    # f, ax = plt.subplots(1, figsize=(10,len(labels)))
    bar_width = 1

    # positions of the left bar-boundaries
    bar_l = [i for i in range(len(labels))]

    plt.bar(bar_l,
       # using pre_rel data
       perc_passed,
       # labeled
       label='Pre Score',
       # with alpha
       alpha=0.9,
       # with color
       color='#019600',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

    tick_pos = [i+(bar_width/2) for i in bar_l]
    plt.xticks(tick_pos, labels)
    plt.xlabel(category_name)

    plt.ylabel("Percentage who passed")
    plt.ylim(-10, 110)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
