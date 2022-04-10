import utils
import torch
import model
import torch.nn as nn
import argparse
import torch.optim as optim
import csv
import os
parser = argparse.ArgumentParser()
parser.add_argument('--gpu-num', default=0, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu_num}')
result_path = '../result'
isDebug = True

def output2csv(pred_y, file_name=os.path.join(result_path, 'sent_class.pred.csv')):
    os.makedirs(result_path, exist_ok=True)
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'pred'])
        for i, p in enumerate(pred_y):
            y_id = str(i + 1)
            if len(y_id) < 3:
                y_id = '0' * (3 - len(y_id)) + y_id
            writer.writerow(['S'+y_id, p[0].item()])
    print('file saved.')


def train(train_x, train_y, epoch):
    tr_loss = 0.
    correct = 0
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    batch_size = args.batch_size
    iteration = (len(train_x) // batch_size) + 1
    cur_i = 0
    for i in range(1, iteration + 1):
        if cur_i >= len(train_x):
            break
        if i < iteration:
            data, targets = train_x[cur_i:cur_i + batch_size].to(device), train_y[cur_i:cur_i + batch_size].to(device)
            cur_i += batch_size
        if i == iteration:
            data, targets = train_x[cur_i:].to(device), train_y[cur_i:].to(device)
            cur_i = len(train_x)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, targets)
        tr_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

        print("\r[epoch {:3d}/{:3d}] loss: {:.6f}".format(epoch, args.epochs, loss), end=' ')

    tr_loss /= iteration
    tr_acc = correct / len(train_x)
    return tr_loss, tr_acc


def test(test_x, targets=None):
    net.eval()
    correct = 0.
    ts_loss = 0.
    with torch.no_grad():
        data = test_x.to(device)
        output = net(data)

        pred = output.argmax(dim=1, keepdim=True)
        if targets != None:
            targets = targets.to(device)
            ts_loss = criterion(output, targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            ts_acc = correct / len(test_x)
            return ts_loss, ts_acc
        else:
            return pred

if __name__ == '__main__':
    # load data
    tr_sents, tr_labels = utils.load_data(file_path='../data/sent_class.train.csv')
    ts_sents, ts_labels = utils.load_data(file_path='../data/sent_class.test.csv')
    if isDebug: print('load_data:', tr_sents[:3])

    # # tokenization
    tr_tokens = utils.tokenization(tr_sents)
    ts_tokens = utils.tokenization(ts_sents)
    if isDebug: print('tokenization:', tr_tokens[:3])
    #
    # # lemmatization
    tr_lemmas = utils.lemmatization(tr_tokens)
    ts_lemmas = utils.lemmatization(ts_tokens)
    if isDebug: print('lemmatization:', tr_lemmas[:3])
    #
    # # character one-hot representation
    char_dict = utils.make_char_dict(tr_lemmas)
    vocab_size = char_dict.__len__()
    if isDebug: print('char_dict:', char_dict)
    tr_char_onehot = utils.char_onehot(tr_lemmas, char_dict).to(device)
    ts_char_onehot = utils.char_onehot(ts_lemmas, char_dict).to(device)
    if isDebug: print('tr_char_onehot:', tr_char_onehot.shape)
    dim = 100
    word_emb_matrix = nn.Parameter(torch.Tensor(vocab_size, dim)).to(device)
    nn.init.kaiming_uniform_(word_emb_matrix)

    train_x = torch.matmul(tr_char_onehot, word_emb_matrix)
    train_y = torch.tensor(tr_labels)
    test_x = torch.matmul(ts_char_onehot, word_emb_matrix)

    if isDebug: print('train_x:', train_x.shape)
    net = model.CNN(6).to(device)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train(train_x, train_y, epoch)
        ts_loss, ts_acc = test(train_x, train_y)
        print("loss: {:.4f}, acc: {:.4f} ts_loss: {:.4f}, ts_acc: {:.4f}".format(tr_loss, tr_acc, ts_loss, ts_acc))

    pred_y = test(test_x)
    output2csv(pred_y)