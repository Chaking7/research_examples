def evaluate_CDS(dataset_sizes, pearsons, crit_fraction=0.9, max_p=None,
                 directory=None):
    '''Automatic evluation of model Pearson score vs dataset size.

    dataset_sizes - array of data count values
    pearsons - corresponding Pearson score
    crit_fraction - [0, 1] value specifying threshold Pearson
        relative to best-in-class
    max_p - if specified, best Pearson seen so far,
        otherwise take the max value of pearsons
    directory - if specified, target dir to save the plot. Otherwise, call plt.show()
    '''
    if max_p == None:
        max_p = max(pearsons)
    crit_fraction = crit_fraction * max_p
    x = plt.plot(dataset_sizes, pearsons, label = 'pearson/datasize')
    for c in range(0, len(dataset_sizes)):
        if crit_fraction < pearsons[c]:
            slope = (pearsons[c] + pearsons[c - 1]) / (dataset_sizes[c] + dataset_sizes[c - 1])
            intersection = crit_fraction / slope
    plt.title("CDS " + str(intersection))
    plt.xlabel('Datapoints')
    plt.ylabel('Pearson_Scores')
    y = plt.axhline(y = crit_fraction, color = 'red', linestyle= 'dashed',
                    label= 'crit_fract')
    plt.legend()
    if directory:
        plt.savefig(os.path.join(directory, filename))
    else:
        plt.show()
    i = 0
    for c in range(0, len(pearsons)):
        if pearsons[c] >= crit_fraction:
            return dataset_sizes[i]
        else:
            i = i + 1

def create_range(length, num_points=10):
    '''Return int-valued logarithmic range from 2 to length.'''
    x = np.logspace(np.log10(2), np.log10(length), num_points-1, endpoint=False)
    return np.append(x, length).astype(int)