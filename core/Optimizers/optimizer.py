import time
class optimizer(object):
    def __init__(self, lr=1e-4):
        self.lr = lr
        self.batch_sz = 1

    def set_lr(self, lr):
        # set learning rate
        self.lr = lr

    def optimize(self, model):
        raise NotImplementedError

    def fit(self, model, train_inputs, train_labels, train_iterator, epoch=1):
        start_t = time.time()

        print('[ num of inputs ]: ', len(train_inputs))
        print('[ num of epoch ]: ', epoch)

        if model.gpu_backend:
            print('[ info ]: Using GPU backend')
            train_iterator.cuda()
        else:
            print('[ info ]: Using CPU backend')
            train_iterator.cpu()

        self.batch_sz = train_iterator.batch_sz

        for loop in range(epoch):
            tot_loss, cnt = 0, 0
            batch_idx = 0
            for train_data in train_iterator(train_inputs, train_labels):
                output = model.forward_prop(train_data["inputs"])
                delta_loss = model.backward_prop(output, train_data["labels"])
                tot_loss += delta_loss
                self.optimize(model)
                cnt += 1
                batch_idx += 1

            print('[ current epoch ]: ', loop + 1,
                  '; [ num_of_batch ]: ', batch_idx + 1,
                  '; [ fps ]: ', cnt * train_iterator.batch_sz / (time.time() - start_t),
                  ';')

            print('[ average loss ]: ', tot_loss / cnt)
            start_t = time.time()
