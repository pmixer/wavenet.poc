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
    def __init__(self, n_classes=256, n_blocks=2, n_layers=14, n_hidden=128):
        super(WaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.blocks = torch.nn.ModuleList()
        for b in range(self.n_blocks):
            layers = torch.nn.ModuleList()
            for l in range(self.n_layers):
                d = 2**(l%10) # max dialation = 512
                if b == 0 and l == 0:
                    conv_op = torch.nn.Conv1d(1, n_hidden, 2, dilation=d, bias=False)
                else:
                    conv_op = torch.nn.Conv1d(n_hidden, n_hidden, 2, dilation=d, bias=False)
                torch.nn.init.xavier_uniform_(conv_op.weight)
                layers.append(conv_op)
            self.blocks.append(layers)
        self.classifier = torch.nn.Conv1d(n_hidden, n_classes, 1)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.001)

    def forward(self, seq):
        for b in range(self.n_blocks):
            for l in range(self.n_layers):
                seq = torch.nn.functional.pad(seq, (2**(l%10), 0))
                seq = self.blocks[b][l](seq)
                seq = torch.nn.functional.relu(seq)
        seq = self.classifier(seq)

        return seq

    def predict(self, first_frame_input, n_steps=32000):
        dN, dC, dT = first_frame_input.shape
        bins = np.linspace(-1, 1, 256)
        dp_queues = []
        dp_queue_indices = []
        for b in range(self.n_blocks):
            for l in range(self.n_layers):
                dp_queue_indices.append(0)
                d = 2**(l%10) # max dialation = 1024
                dp_queues.append(torch.zeros((dN, self.n_hidden, d)))

        res = torch.zeros((dN, dC, n_steps)); res[:, :, :2] = first_frame_input
        for s in range(2, n_steps):
            emb = res[:, :, s-1] # last_frame
            rem = res[:, :, s-2]
            for lid in range(len(dp_queues)):
                b, l = lid // self.n_layers, lid % self.n_layers
                W = self.blocks[b][l].weight.data
                W_r, W_e = W[:, :, 0], W[:, :, 1]
                emb = torch.matmul(rem, W_r.T) + torch.matmul(emb, W_e.T)
                emb = torch.nn.functional.relu(emb)

                rem = dp_queues[lid][:, :, dp_queue_indices[lid]]
                qlen = self.blocks[b][l].dilation[0]
                dp_queue_indices[lid] = (dp_queue_indices[lid] + 1) % qlen
                dp_queues[lid][:, :, dp_queue_indices[lid]] = emb
            print("generated frame: " + str(s))
            emb = torch.matmul(emb, self.classifier.weight.data.squeeze(-1).T)
            emb += self.classifier.bias.data

            label = torch.argmax(emb, axis=1)
            emb = bins[label.item()]
            res[:, :, s] = emb

        return res


if __name__ == '__main__':
    input_seqs, next_items = parse_wav('voice.wav')
    dev = 'cpu' # cuda:0
    input_seqs = torch.FloatTensor(input_seqs).to(dev)
    next_items = torch.LongTensor(next_items).to(dev)
    # initializer, layers, loss, optimizer
    model = WaveNet().to(dev)
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))

    # training, trying to fit one audio clip by gradient descent
    if False: # train or not
        for i in range(2333):
            adam_optimizer.zero_grad()
            next_item_preds = model(input_seqs)
            loss = ce_criterion(next_item_preds, next_items)
            print("iteration %d, loss: %.4f" % (i, loss.item()))
            if loss.item() < 0.1: break
            loss.backward()
            adam_optimizer.step()
        torch.save(model.state_dict(), 'model.ckpt')

    dev = 'cpu'
    model.load_state_dict(torch.load('model.ckpt', map_location=torch.device(dev)))

    # start online/causal prediction
    input_ = input_seqs[:, :, :2]
    audio_norm = model.predict(input_)

    # save generated audio file
    audio = (audio_norm.squeeze().numpy() * 32768.0).astype('int16')
    scipy.io.wavfile.write('generated.wav', 44100, audio)