import numpy as np
from numpy.testing import assert_array_almost_equal
from .cifar_c import CIFAR_CORRUPTIONS
from .tiny_imagenet_c import TI_CORRUPTIONS

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise


from PIL import Image, ImageFilter
# def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
#     if noise_type == 'pairflip':
#         train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
#     if noise_type == 'symmetric':
#         train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
#     return train_noisy_labels, actual_noise_rate

def noisify_y(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels

def add_trigger(trainset, targets, trigger_size, trigger_rate=0, random_state=0):
    mask = np.zeros([32, 32, 3], dtype=np.uint8)
    trigger = np.zeros([32, 32, 3], dtype=np.uint8)
    position_x = np.random.randint(0, 33-trigger_size)
    position_y = np.random.randint(0, 33-trigger_size)
    color = np.random.randint(0, 256, [trigger_size, trigger_size, 3])
    print('position:', position_x, position_y, 'color:', color)
    
    mask[position_x:position_x+trigger_size, position_y:position_y+trigger_size, :] = 1
    trigger[position_x:position_x+trigger_size, position_y:position_y+trigger_size, :] = color
    seed = np.random.RandomState(random_state)
    selected = seed.binomial(1, trigger_rate, size=(trainset.shape[0],))
    for i in range(trainset.shape[0]):
        if selected[i] == 0:
            continue 
        trainset[i] = (1 - mask) * trainset[i] + mask * trigger
        targets[i] = 0
    
    # im = Image.fromarray(mask * 255)
    # im.save(os.path.join(args.model_dir, 'mask.png'))

    # im = Image.fromarray(trigger)
    # im.save(os.path.join(args.model_dir, 'trigger.png'))

    # num_poison = int(len(trainset.targets) * trigger_ratio)

    # index = np.where(np.asarray(trainset.targets) != target)[0]
    # np.random.shuffle(index)
    # selected_index = index[:num_poison]
    
    # for idx in selected_index:
    #     trainset.data[idx] = (1 - mask) * trainset.data[idx] + mask * trigger
    #     trainset.targets[idx] = target

    # for i in range(10):
    #     print('number of images belonging to label', i, 'is', np.sum(np.asarray(trainset.targets) == i))

    return trainset, targets, selected

def noisify_x(dataset, data, noise_type, noise_arg, noise_rate=0, random_state=0):
    if dataset == 'cifar':
        CORRUPTIONS = CIFAR_CORRUPTIONS
    else:
        CORRUPTIONS = TI_CORRUPTIONS

    if not noise_type in CORRUPTIONS:
        print('Unsupport Noise Type!')
        return data, np.zeros(data.shape[0])
    if noise_arg < 1 or noise_arg > 7:
        print('noise arg out of bounds!')
        return data, np.zeros(data.shape[0])
    seed = np.random.RandomState(random_state)
    selected = seed.binomial(1, noise_rate, size=(data.shape[0],))
    for i in range(data.shape[0]):
        if selected[i] == 0:
            continue 
        data[i] = np.uint8(CORRUPTIONS[noise_type](data[i], int(noise_arg)))
    # print('xnoise selected: ', (selected ==1).sum())
    return data, selected

def noisify_overlap(data, noise_rate, noise_arg, random_state):
    import copy
    new_data = copy.deepcopy(data)
    N = data.shape[0]
    seed = np.random.RandomState(random_state)
    selected = seed.binomial(1, noise_rate, size=(N,))
    for i in range(N):
        if selected[i] == 0: continue 
        bg_idx = seed.randint(0, N)
        while bg_idx == i: 
            bg_idx = seed.randint(0, N)
        new_data[i] = np.maximum(data[i], noise_arg * data[bg_idx])
    #print('xnoise selected: ', (selected ==1).sum())
    return new_data, selected

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std
