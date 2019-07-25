import torch

class HammingLoss(object):
    def __init__(self):
        self._num_samples = 0
        self._num_labels = 0
        self._wrong_pred = 0

    def update(self, predicted, target):
        if not self._num_labels:
            self._num_labels = target.size(1)
        assert(target.size(1) == predicted.size(1) == self._num_labels)
        assert(target.size(0) == predicted.size(0))
        self._num_samples += target.size(0)
        cur_wrong_pred = (target.byte() ^ predicted.byte()).sum().item()
        self._wrong_pred += cur_wrong_pred
        return cur_wrong_pred/(self._num_labels * target.size(0))    # loss for current batch

    @property
    def loss(self):
        return (self._wrong_pred/(self._num_labels*self._num_samples))

    @loss.setter
    def loss(self, val):
        raise NotImplementedError('Modifying hamming loss value is not supported.')

    @property
    def inverseloss(self):
        return (1 - (self._wrong_pred/(self._num_labels*self._num_samples)))

    @inverseloss.setter
    def inverseloss(self, val):
        raise NotImplementedError('Modifying inverse hamming loss value is not supported.')
