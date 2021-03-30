import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from tqdm import tqdm, tqdm_notebook

from .utilities import device


def classification_performance(pred, true):

    loss = F.cross_entropy(pred, true, reduction = 'mean')

    pred = pred.max(1)[1]

    corrects = pred.eq(true).sum().item()

    total = len(true)

    return loss, corrects, total


def fit_classification(epoch, dataloader, model, pad_id, optimizer, pbar, save_every = None, save_path = None, phase = 'training', clip = 2, notebook = False):

    if phase == 'training':

        model.train()

    if phase == 'validation':

        model.eval()

    total_loss, n_items_total, n_items_correct = 0, 0, 0


    if notebook:
        progress = tqdm_notebook(total = len(dataloader), leave = False, desc = "Dataloader")
    else:
        progress = tqdm(total = len(dataloader), leave = False, desc = "Dataloader")

    for ix, batch in enumerate(dataloader):

        src, trg, label = Variable(batch['src'].to(device())), Variable(batch['trg'].to(device())), Variable(batch['label'].to(device()))

        if phase == 'training':

            optimizer.zero_grad()

        pred = model(src, trg)

        loss, corrects, total = classification_performance(pred, label)

        if phase == 'training':

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        n_items_total += total
        n_items_correct += corrects
        total_loss += loss.item()

        progress.update()

    loss_per_item = total_loss / n_items_total
    accuracy = n_items_correct / n_items_total

    t = f'''{phase.upper} EPOCH: {epoch} | Loss: {loss_per_item} | Accuracy: {accuracy}'''

    if save_every:
        if epoch % save_every == 0:
            if not save_path:
                return f'Got results {t}, please provide a path to `save_path` argument to save your model after every {save_every} epochs as chosen'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            save_path_ = os.path.join(save_path, f'classification_transformer_training_epoch_{epoch}.pt')
            save_path_dict = os.path.join(save_path, f'classification_transformer_training_epoch_{epoch}_state_dict.pth')

            torch.save(model, save_path_)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_per_item,
                'acc': accuracy
            }, save_path_dict)

    pbar.write(t)
    pbar.update(1)
    return loss_per_item, accuracy, t



