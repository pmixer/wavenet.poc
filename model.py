# in reference to https://github.com/tomlepaine/fast-wavenet
# omg, I forgot to add relu between layers...
import time
import torch
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt


def parse_wav(path):
    rate, wav = scipy.io.wavfile.read(path)
    data = wav[:, 0] # 2 sound channels, take 1st one

    # normalize to [-1, 1]
    data_ = np.float32(data) - np.min(data)
    data_ = (data_ / np.max(data_) - 0.5) * 2

    # or use mu law for better normalization:
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    # quantization, gen input seqs, N, C, T for PyTorch
    bins = np.linspace(-1, 1, 256)
    bin_indices = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[bin_indices][None, None, :] # shape of (T) to (N=1, C=1, T)

    # quantization, gen target integers(next-item bin index)
    targets = (np.digitize(data_[1:], bins, right=False) - 1)[None, :]
    return inputs, targets


# tf implementation looks more complex
class WaveNet(torch.nn.Module):
    def __init__(self, n_classes=256, n_layers=26, n_hidden=128):
        super(WaveNet, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = torch.nn.ModuleList()
        for l in range(self.n_layers):
            d = 2**(l%14) # max dialation = 8192
            if l == 0:
                conv_op = torch.nn.Conv1d(1, n_hidden, 2, dilation=d, bias=False)
            else:
                conv_op = torch.nn.Conv1d(n_hidden, n_hidden, 2, dilation=d, bias=False)
            torch.nn.init.xavier_uniform_(conv_op.weight)
            self.layers.append(conv_op)
        self.classifier = torch.nn.Conv1d(n_hidden, n_classes, 1)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.001)

    def forward(self, seq):
        for l in range(self.n_layers):
            seq = torch.nn.functional.pad(seq, (2**(l%14), 0))
            seq = self.layers[l](seq)
            seq = torch.nn.functional.relu(seq)
        seq = self.classifier(seq)

        return seq

    def predict(self, first_frame_input, dev, n_steps=32000):
        dN, dC = first_frame_input.shape
        bins = np.linspace(-1, 1, 256)
        labels = []
        dp_queues = []
        dp_queue_indices = []
        for l in range(self.n_layers):
            dp_queue_indices.append(0)
            d = 2**(l%14)
            dp_queues.append(torch.zeros((dN, self.n_hidden, d)).to(dev))

        res = torch.zeros((dN, dC, n_steps)).to(dev); res[:, :, 1] = first_frame_input
        for s in range(2, n_steps):
            emb = res[:, :, s-1] # last_frame
            rem = res[:, :, s-2]
            for l in range(self.n_layers):
                if l > 0:
                    rem = dp_queues[l][:, :, dp_queue_indices[l]].clone()
                    dp_queues[l][:, :, dp_queue_indices[l]] = emb
                W = self.layers[l].weight.data
                W_r, W_e = W[:, :, 0], W[:, :, 1]
                emb = torch.matmul(rem, W_r.T) + torch.matmul(emb, W_e.T)
                emb = torch.nn.functional.relu(emb)

                qlen = dp_queues[l].shape[-1]
                assert qlen == self.layers[l].dilation[0]
                dp_queue_indices[l] = (dp_queue_indices[l] + 1) % qlen
                
            print("generated frame: " + str(s))
            emb = torch.matmul(emb, self.classifier.weight.data.squeeze(-1).T)
            emb += self.classifier.bias.data

            label = torch.argmax(emb, axis=1).item()
            # if label != 89:
                # import pdb; pdb.set_trace()
            labels.append(label)
            res[:, :, s] = bins[label]

        return res, labels


if __name__ == '__main__':
    input_seqs, next_items = parse_wav('voice.wav')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_seqs = torch.FloatTensor(input_seqs).to(dev)
    next_items = torch.LongTensor(next_items).to(dev)
    # initializer, layers, loss, optimizer
    model = WaveNet().to(dev)
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    
    try:
        model.load_state_dict(torch.load('model.ckpt', map_location=torch.device(dev)))
    except:
        print('no model.ckpt found, training from scratch')

    # training, trying to fit one audio clip by gradient descent
    if True: # train or not
        for i in range(2333):
            adam_optimizer.zero_grad()
            next_item_preds = model(input_seqs)
            loss = ce_criterion(next_item_preds, next_items)
            print("iteration %d, loss: %.4f" % (i, loss.item()))
            if loss.item() < 0.1: break
            loss.backward()
            adam_optimizer.step()
        torch.save(model.state_dict(), 'model.ckpt')

    model.load_state_dict(torch.load('model.ckpt', map_location=torch.device(dev)))

    # test fitting result
    logits = model(input_seqs)
    ref_labels = torch.argmax(logits, axis=1).cpu().numpy().flatten().tolist()
    bins = np.linspace(-1, 1, 256)
    audio_norm = bins[ref_labels]
    audio = (audio_norm * 32768.0).astype('int16')
    scipy.io.wavfile.write('expected.wav', 44100, audio.flatten())

    # start online/causal prediction
    input_ = input_seqs[:, :, 0]
    audio_norm, labels = model.predict(input_, dev)

    # save generated audio file
    audio = (audio_norm.squeeze().cpu().numpy() * 32768.0).astype('int16')
    scipy.io.wavfile.write('generated.wav', 44100, audio)
