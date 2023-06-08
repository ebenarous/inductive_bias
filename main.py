'''Storage used in /Users/elior/.cache/'''
import logging
import time
import pandas as pd
import numpy as np
from PIL import Image
import copy
import os
import matplotlib.pyplot as plt
import wandb
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.spatial.distance import pdist

from sot_torchvision_models import resnet18, resnet50
from sot_modif_resnet import modify_resnet_model
from models import *
from vit_models import ViT

from data import load_geirhos_transfer_pre, load_data, MyDataset, load_noise, load_fractal, load_geirhos_edge_silhouette
from geirhos.probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
from loss import info_nce_loss
from utils import create_logger, visualize, norm_calc, find_overlap, remove_int


def pretext_train(epoch, pre_type='supervised',  train_loader=DataLoader, model=nn.Module, pre_lr=0.001,
                  log_interval=100, save_models=False, save_path=None, verbose=False, log=True, logger=None, device='cuda:0'):
    """
    Modify loss and optimizer inside function because that's not something we need to modify easily. Not project focus.
    """
    logger.info('')
    if pre_type=='supervised':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=pre_lr)
        train_correct = 0
        
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()

            if i % log_interval == 0:
                msg = '[Epoch %d] Batch [%d], Loss: %.3f' % (epoch + 1, i + 1, loss.item())
                if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
                if verbose: print(msg)
        
        train_acc = 100. * train_correct / len(train_loader.dataset)

    elif pre_type=='contrastive':
        """ TODO:
        Choose loss: InfoNCE, NT-Xent, Contrastive softmax
        Choose optimizer: SimCLR use LARS
        """
        optimizer = optim.Adam(model.parameters(), lr=pre_lr)
        train_acc = 0
        model.train()
        for i, ((im_x, im_y), _) in enumerate(train_loader): # adapted to loaders that include labels, if loader does not have labels, replace with "i, (im_x, im_y)"
            optimizer.zero_grad()
            inputs = torch.cat([im_x, im_y], dim=0).to(device)
            out = model(inputs)
            # need to separate again with .chunk(2) ?

            # Apply InfoNCE loss
            loss, acc = info_nce_loss(out=out, temperature=0.5, log=log, logger=logger)
            loss.backward()
            optimizer.step()
            train_acc += acc.item()

            if i % log_interval == 0:
                msg = '[Epoch %d] Batch [%d], Loss: %.3f' % (epoch + 1, i + 1, loss.item())
                if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
                if verbose: print(msg)
        
        # TODO: Is this correct? is the acc computed in InfoNCE logical?
        train_acc /= len(train_loader.dataset)
        train_acc *= 100

    msg = '[Epoch %d] Pre-training complete, Acc: %.3f%%' % (epoch + 1, train_acc)
    if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
    if verbose: print(msg)
    if save_models and save_path is not None: 
        torch.save(model.state_dict(), save_path)
        msg = 'Model saved to {}'.format(save_path)
        if verbose: print(msg)
        if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)

    return model
        
def test(epoch, pre_type='supervised', model=nn.Module, is_vit=False, classifier=nn.Module, test_loader=DataLoader, stage='Pre',
         save_models=False, save_path=None, verbose=False, log=True, logger=None, jig=False, device='cuda:0'):
    # if pretext objective was contrastive, modify model to adapt to downstream objective
    
    if pre_type == 'contrastive' and stage == 'Pre':
        # Remain in contrastive framework
        test_acc = 0
        test_loss = 0
        avg_dist = 0

        model.eval()
        with torch.no_grad():
            for i, ((im_x, im_y), _) in enumerate(test_loader): # adapted to loaders that include labels, if loader does not have labels, replace with "i, (im_x, im_y)"
                inputs = torch.cat([im_x, im_y], dim=0).to(device)
                h, out = model(inputs, return_both=True) # used to be out = model(inputs)

                # Compute distance between embeddings of same batch
                dist_vec = pdist(h.detach().cpu().numpy(), metric='cosine')
                avg_dist += np.sum(dist_vec) # TODO: need to divide by 2 because 2 views?

                # Standard test classification procedure
                loss, acc = info_nce_loss(out=out, temperature=0.5, mode='test', log=log, logger=logger)
                test_acc += acc.item()
                test_loss += loss.item()
            
        # TODO: Is this correct? is the acc computed in InfoNCE logical?
        test_acc = 100. * test_acc / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        avg_dist /= len(test_loader.dataset) # * 2 because of 2 views processed ? doesnt scale linearly though
        
        msg = '[Epoch %d] %s-testing complete, Avg Loss: %.3f, Acc: %.3f%%' % (epoch + 1, stage, test_loss, test_acc)
        if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
        if verbose: print(msg)
        if save_models and save_path is not None: 
            torch.save(model.state_dict(), save_path)
            msg = 'Model saved to {}'.format(save_path)
            if verbose: print(msg)
            if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)

      
    else:
        criterion = nn.CrossEntropyLoss()
        test_correct = 0
        test_loss = 0
        avg_dist = 0

        # If downstream testing, the nb_classes should be different from the pre_training, 
        # regardless of pre_type and finetune
        if stage == 'Down':
            '''if not is_vit: model.fc = nn.Identity()
            else: model.head = nn.Identity()'''
            classifier.eval()
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if stage == 'Pre':
                    h, out = model(inputs, return_both=True)
                else:
                    h = model(inputs, return_embed=True)
                    out = classifier(h)
                
                # Compute distance between embeddings of same batch
                dist_vec = pdist(h.detach().cpu().numpy(), metric='cosine')
                avg_dist += np.sum(dist_vec)

                # Standard test classification procedure
                test_loss += criterion(out, labels)
                pred = out.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(labels.view_as(pred)).sum().item() 
        test_acc = 100. * test_correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        avg_dist /= len(test_loader.dataset)

        msg = '[Epoch %d] %s-testing complete, Avg Loss: %.3f, Acc: %.3f%%' % (epoch + 1, stage, test_loss, test_acc)
        if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
        if verbose: print(msg)
        if save_models and save_path is not None: 
            torch.save(model.state_dict(), save_path)
            msg = 'Model saved to {}'.format(save_path)
            if verbose: print(msg)
            if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)

    if not jig: wandb.log({'{}_acc'.format(stage):test_acc, '{}_loss'.format(stage):test_loss})
    return test_acc, avg_dist

def down_finetune(epoch, finetune_epochs=5, pre_type='supervised', train_loader=DataLoader, 
                  model=nn.Module, is_vit=False, classifier=nn.Module, down_lr=0.001,
                  log_interval=100, save_models=False, save_path=None, verbose=False, log=True, logger=None, device='cuda:0'):
    '''if pre_type=='supervised':
        # modify fc dimensions and finetune with standard training procedure
        if not is_vit: model.fc = classifier
        else: model.head = classifier
        model.train()
    
    elif pre_type=='contrastive': 
        # freeze encoder and train a head
        if not is_vit: model.fc = nn.Identity()
        else: model.head = nn.Identity()
        model.eval()
        classifier.train()''' # TODO: can fix the identity pb with return_embed=True but how to finetune encoder?
    
    model.eval()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=down_lr)
    for e in range(finetune_epochs):
        train_correct = 0
        train_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            '''if pre_type=='supervised':
                out = model(inputs)
            elif pre_type=='contrastive':
                out = classifier(model(inputs))'''# TODO: can fix the identity pb with return_embed=True but how to finetune encoder?
            
            out = classifier(model(inputs, return_embed=True))
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
            train_loss += loss

            if i % log_interval == 0:
                msg = '[Finetuning Epoch %d] Batch [%d], Loss: %.3f' % (e + 1, i + 1, loss.item())
                if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
                if verbose: print(msg)

    train_acc = 100. * train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    
    msg = '[Epoch %d] Finetuning complete, Avg Loss: %.3f, Acc: %.3f%%' % (epoch + 1, train_loss, train_acc)
    if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
    if verbose: print(msg)
    if save_models and save_path is not None: 
        torch.save(model.state_dict(), save_path)
        msg = 'Model saved to {}'.format(save_path)
        if verbose: print(msg)
        if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)

    return model

def eval_bias(model, loader, mapping, 
                  log=True, verbose=False, logger=None, epoch=0, device='cuda:0'):

    model.eval()
    results = []
    with torch.no_grad():
        for img, path in loader:
            labels = pd.Series(path)
            labels = labels.str.split("/").str.get(-1).str.split("-")
            labels = pd.DataFrame({'shape': labels.str.get(0), 
                                   'texture': labels.str.get(1)})
            labels['texture'] = labels['texture'].str.replace(r'\.png$', '', regex=True)
            labels['shape'], labels['texture'] = labels['shape'].apply(remove_int), labels['texture'].apply(remove_int)

            out = model(img.to(device))
            out = torch.nn.Softmax(dim=1)(out)
            out = torch.squeeze(out).detach().cpu().numpy()
            map = mapping.probabilities_to_decision(out)

            labels['Map'] = map
            results.append(labels.to_numpy())
        
    results = pd.DataFrame(np.vstack(results), columns=["Shape", "Texture", "Map"])
    correct_shape = results.loc[results['Map'] == results['Shape']]
    correct_texture = results.loc[results['Map'] == results['Texture']]
    
    nb_shape = len(correct_shape.index)
    nb_texture = len(correct_texture.index)
    shape_bias = nb_shape / (nb_shape + nb_texture)

    overlap = find_overlap(correct_shape.index, correct_texture.index)
    accuracy = (nb_shape + nb_texture - overlap) / len(loader.dataset)
    
    msg = '[Epoch %d] Standard Bias Eval complete, Shape Bias: %.3f%% Acc: %.3f%%' % (epoch + 1, shape_bias*100, accuracy*100)
    if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
    if verbose: print(msg)

    wandb.log({'shape bias':shape_bias})
    return shape_bias, accuracy

def eval_bias_embed(model, loader, nb_neigh=5, metric='cosine', is_vit=False,
                        log=True, verbose=False, logger=None, epoch=0, device='cuda:0'):
    embed_map = {
        "knife"    : 0, 
        "keyboard" : 1, 
        "elephant" : 2, 
        "bicycle"  : 3, 
        "airplane" : 4,
        "clock"    : 5, 
        "oven"     : 6, 
        "chair"    : 7, 
        "bear"     : 8, 
        "boat"     : 9, 
        "cat"      : 10, 
        "bottle"   : 11,
        "truck"    : 12, 
        "car"      : 13,
        "bird"     : 14, 
        "dog"      : 15
    }
    
    model_results = np.ones(shape=(nb_neigh, 2))
    model.eval()
    for it in range(1, nb_neigh+1):
        results = []
        gt = []
        embeddings = []
        with torch.no_grad():
            # collect all embeddings
            for img, path in loader:
                labels = pd.Series(path)
                labels = labels.str.split("/").str.get(-1).str.split("-")
                labels = pd.DataFrame({'shape': labels.str.get(0), 
                                       'texture': labels.str.get(1)})
                labels['texture'] = labels['texture'].str.replace(r'\.png$', '', regex=True)
                labels['shape'], labels['texture'] = labels['shape'].apply(remove_int), labels['texture'].apply(remove_int)
                gt.append(labels.replace(embed_map).to_numpy())

                embeddings.append(torch.squeeze(model(img.to(device), return_embed=True)).detach().cpu().numpy())

            embeddings = np.vstack(embeddings)
            gt = np.vstack(gt)

            # eval with KNN
            for idx in range(len(embeddings)):
                train_embeddings = np.delete(np.asarray(embeddings), idx, axis=0)
                test_embedding = np.expand_dims(embeddings[idx], axis=0)
                train_gt = np.delete(gt, idx, axis=0)
                test_gt = gt[idx]

                # Fit for all but this embedding
                # TODO: which metric to use, using cosine <=> normalizing
                neigh_shape = KNN(n_neighbors=it, metric=metric)
                neigh_shape.fit(X=train_embeddings, y=train_gt[:, 0])
                neigh_texture = KNN(n_neighbors=it, metric=metric)
                neigh_texture.fit(X=train_embeddings, y=train_gt[:, 1])

                # Predict on this embedding
                pred_shape = np.squeeze(neigh_shape.predict(test_embedding))
                pred_texture = np.squeeze(neigh_texture.predict(test_embedding))

                results.append(np.array([pred_shape, pred_texture, test_gt[0], test_gt[1]]))

        results = pd.DataFrame(results, columns=["Predicted Shape", "Predicted Texture", "Shape", "Texture"])
        correct_shape = results.loc[results['Predicted Shape'] == results['Shape']]
        correct_texture = results.loc[results['Predicted Texture'] == results['Texture']]
        
        nb_shape = len(correct_shape.index)
        nb_texture = len(correct_texture.index)
        shape_bias = nb_shape / (nb_shape + nb_texture)
        
        overlap = find_overlap(correct_shape.index, correct_texture.index)
        accuracy = (nb_shape + nb_texture - overlap) / len(loader.dataset)

        model_results[it-1, 0], model_results[it-1, 1] = shape_bias, accuracy

    model_bias_avg, model_acc_avg = np.average(model_results, axis=0, keepdims=True)[0]

    msg = '[Epoch %d] Embedding Bias Eval complete, with %d neighbors - Shape Bias: %.3f%% Acc: %.3f%%' % (epoch + 1, nb_neigh, model_bias_avg*100, model_acc_avg*100)
    if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
    if verbose: print(msg)
    
    wandb.log({'shape bias knn':shape_bias})
    return model_bias_avg, model_acc_avg

def eval_edge_sil_embed(model, loader, nb_neigh=5, metric='cosine', is_vit=False,
                        log=True, verbose=False, logger=None, epoch=0, device='cuda:0', type='Edge'):
    embed_map = {
        "knife"    : 0, 
        "keyboard" : 1, 
        "elephant" : 2, 
        "bicycle"  : 3, 
        "airplane" : 4,
        "clock"    : 5, 
        "oven"     : 6, 
        "chair"    : 7, 
        "bear"     : 8, 
        "boat"     : 9, 
        "cat"      : 10, 
        "bottle"   : 11,
        "truck"    : 12, 
        "car"      : 13,
        "bird"     : 14, 
        "dog"      : 15
    }
    
    model_results = np.ones(shape=(nb_neigh, 1))
    model.eval()
    for it in range(1, nb_neigh+1):
        results = []
        gt = []
        embeddings = []
        with torch.no_grad():
            # collect all embeddings
            for img, path in loader:
                labels = pd.Series(path)
                labels = labels.str.split("/").str.get(-1)
                labels = pd.DataFrame({'labels': labels})
                labels['labels'] = labels['labels'].str.replace(r'\.png$', '', regex=True).apply(remove_int)
                
                gt.append(labels.replace(embed_map).to_numpy())
                embeddings.append(torch.squeeze(model(img.to(device), return_embed=True)).detach().cpu().numpy())

            embeddings = np.vstack(embeddings)
            gt = np.vstack(gt)

            # eval with KNN
            for idx in range(len(embeddings)):
                train_embeddings = np.delete(np.asarray(embeddings), idx, axis=0)
                test_embedding = np.expand_dims(embeddings[idx], axis=0)
                train_gt = np.delete(gt, idx, axis=0)
                test_gt = gt[idx]

                # Fit for all but this embedding
                neigh = KNN(n_neighbors=it, metric=metric)
                neigh.fit(X=train_embeddings, y=train_gt.ravel())
                
                # Predict on this embedding
                pred = np.squeeze(neigh.predict(test_embedding))

                results.append(np.array([pred, test_gt.item()]))

        results = pd.DataFrame(results, columns=["Predicted", "Ground Truth"])
        correct = results.loc[results['Predicted'] == results['Ground Truth']]
        accuracy = len(correct.index) / len(loader.dataset)

        model_results[it-1] = accuracy

    model_acc_avg = np.average(model_results, axis=0, keepdims=True).item()

    msg = '[Epoch %d] %s Embedding Eval complete, with %d neighbors - Acc: %.3f%%' % (epoch + 1, type, nb_neigh, model_acc_avg*100)
    if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
    if verbose: print(msg)
    
    wandb.log({'{} knn'.format(type) : accuracy})
    return model_acc_avg

def eval_views_embed_dist(epoch, model=nn.Module, is_vit=False, test_loader=DataLoader, 
                        save_models=False, save_path=None, verbose=False, log=True, logger=None, device='cuda:0'):
    # Testing the encoder with embedding distances
    
    '''if not is_vit: model.fc = nn.Identity()
    else: model.head = nn.Identity()'''
    
    model.eval()
    avg_dist = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            nb_views = len(inputs)
            batch_size = len(inputs[0])
            inputs = torch.cat([view for view in inputs], dim=0).to(device) # won't work if loader does not load 2+ views
            out = model(inputs, return_embed=True)

            for h in out.reshape(nb_views, batch_size, -1).transpose(0, 1):
                dist_vec = pdist(h.detach().cpu().numpy(), metric='cosine')
                avg_dist += np.sum(dist_vec)
    avg_dist /= len(test_loader.dataset)

    msg = '[Epoch %d] Pair embeddings distance testing complete, Avg Distance: %.3f' % (epoch + 1, avg_dist)
    if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)
    if verbose: print(msg)
    if save_models and save_path is not None: 
        torch.save(model.state_dict(), save_path)
        msg = 'Model saved to {}'.format(save_path)
        if verbose: print(msg)
        if log: logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + msg)

    return avg_dist


def main(models2compare=[], train_epochs=10, pre_type='supervised', pre_dataset='CIFAR10', aug_pre=False,
            test_interval=5, finetune=False, down_dataset='CIFAR10', 
            save_models=False, experiment_id='test1_elior', modelnames=[], checkpoint_names=None):
        
        """
        Returns: score_table: shape = (nb of models to compare, nb of logged epochs)
                    score_table indices: name of models
                    score_table columns: logged epochs
                    inside each element: [pretext_test, bias_percentage, downstream_test, embed_pair_dist]
                    access a score: score_table.loc['model name', 'epoch nb'][idx]
        """

        assert len(models2compare) == len(modelnames), 'Provide a list of model names with same length as list of models to be tested'
        if checkpoint_names is not None: assert len(models2compare) == len(checkpoint_names), 'Provide a list of model checkpoint names with same length as list of models to be tested'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Pre and Down Dataloader
        pre_train, pre_test = load_data(dataset=pre_dataset, stage='pre', finetune=finetune, aug=aug_pre) # TODO: cifar train with simCLR aug
        down_train, down_test = load_data(dataset=down_dataset, stage='down', finetune=finetune)
        
        # Custom Geirhos Dataloaders
        bias_data_path = load_geirhos_transfer_pre(conflict_only=True)
        edge_data_path = load_geirhos_edge_silhouette(type='edge')
        sil_data_path = load_geirhos_edge_silhouette(type='sil')
        geirhos_bs = 256 if device=='cuda:0' else 1
        geirhos_bias_ds = MyDataset(bias_data_path, transforms=transforms.Compose([transforms.Resize(32),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        geirhos_edge_ds = MyDataset(edge_data_path, transforms=transforms.Compose([transforms.Resize(32),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        geirhos_sil_ds = MyDataset(sil_data_path, transforms=transforms.Compose([transforms.Resize(32),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        geirhos_loader = torch.utils.data.DataLoader(geirhos_bias_ds, batch_size=geirhos_bs, num_workers=0, shuffle=False, pin_memory=False)
        geirhos_edge_loader = torch.utils.data.DataLoader(geirhos_edge_ds, batch_size=geirhos_bs, num_workers=0, shuffle=False, pin_memory=False)
        geirhos_sil_loader = torch.utils.data.DataLoader(geirhos_sil_ds, batch_size=geirhos_bs, num_workers=0, shuffle=False, pin_memory=False)

        # Pair Embedding Distances Dataloader
        _ , views_dist_loader = load_data(dataset=down_dataset, stage='down', finetune=False, n_views=3)
        _ , gray_pair_dist_loader = load_data(down_dataset, ('gray', 1), stage='down', finetune=False, n_views=2, spe_pair=True) # ensure other sample is augmented
        _ , hflip_pair_dist_loader = load_data(down_dataset, ('hflip', 1), stage='down', finetune=False, n_views=2, spe_pair=True) # ensure other sample is augmented
        _ , rdm_rcrop_pair_dist_loader = load_data(down_dataset, ('crop', 1), stage='down', finetune=False, n_views=2, spe_pair=True)

        # Jigsaw Dataloader
        _ , jig_loader_16 = load_data(down_dataset, stage='down', jigsaw_ps=16)
        _ , jig_loader_8 = load_data(down_dataset, stage='down', jigsaw_ps=8)

        # Init logger
        logger = create_logger(experiment_id)
        logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + 'Starting {} pre-learning on {}, testing on {}'.format(pre_type, pre_dataset, down_dataset))

        class_name_2_nb_classes = {
            'ImageNet' : 1000,
            'CIFAR10'  : 10,
            'CIFAR100' : 100,
            'STL10'    : 10,
            'tiny'     : 200,
        }
        scores = []
        scores_idx = 0
        scores_epochs = []
        for model in models2compare:
            scores.append([])
            distances = []
            start_epoch = 0

            if checkpoint_names is not None:
                model.load_state_dict(torch.load(os.path.join('model', checkpoint_names[scores_idx] + '.pth')))
                str_epoch = r'_e(\d+)_'
                match = re.search(str_epoch, checkpoint_names[scores_idx])
                if match:
                    start_epoch = int(match.group(1))
                logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + 'Loaded model: {} at epoch: {}'.format(checkpoint_names[scores_idx], start_epoch))

            model.to(device)
            try: model.fc # for model.fc / model.head 
            except: is_vit = True
            else: is_vit = False
            
            # init wandb log
            wandb.init(entity='eliorb', project=experiment_id, name=modelnames[scores_idx])
            logger.info(time.strftime('%Y-%m-%d-%H-%M') + ' - ' + 'Start of {}'.format(modelnames[scores_idx]))
            for epoch in range(start_epoch, train_epochs+1):
                
                save_path = os.path.join('model', '{}_{}_pre.pth'.format(modelnames[scores_idx], epoch+1))
                
                # Train on pretext task
                model = pretext_train(pre_type=pre_type, train_loader=pre_train, model=model, pre_lr=0.001,
                                      log_interval=100, save_models=save_models, save_path=save_path, verbose=False, log=True, logger=logger, epoch=epoch, device=device)
                
                # Test
                if epoch % test_interval == 0:
                    
                    # Pretext test
                    if pre_dataset not in ['noise', 'fractal']:
                        result_pre, dist_pre = test(pre_type=pre_type, model=model, test_loader=pre_test, is_vit=is_vit, stage='Pre',
                                                    save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)

                    
                    # Shape bias with Geirhos method (1200 images in his custom set)
                    if pre_dataset == 'ImageNet': # standard classification
                        result_bias, result_acc = eval_bias(model=model, loader=geirhos_loader, mapping=ImageNetProbabilitiesTo16ClassesMapping(),
                                                            log=True, verbose=False, logger=logger, epoch=epoch, device=device)
                    else: # KNN classification of embeddings
                        result_bias, result_acc = eval_bias_embed(model=model, loader=geirhos_loader, nb_neigh=5, metric='cosine',
                                                                  is_vit=is_vit, log=True, verbose=False, logger=logger, epoch=epoch, device=device)
                        edge_acc = eval_edge_sil_embed(model=model, loader=geirhos_edge_loader, nb_neigh=5, metric='cosine', 
                                                       is_vit=is_vit, log=True, verbose=False, logger=logger, epoch=epoch, device=device, type='Edge')
                        sil_acc = eval_edge_sil_embed(model=model, loader=geirhos_sil_loader, nb_neigh=5, metric='cosine', 
                                                       is_vit=is_vit, log=True, verbose=False, logger=logger, epoch=epoch, device=device, type='Sil')

                    # Downstream
                    out_features = class_name_2_nb_classes[down_dataset]
                    if not is_vit: classifier = nn.Linear(in_features=model.fc.in_features, out_features=out_features).to(device)
                    else: classifier = nn.Linear(in_features=model.head.in_features, out_features=out_features).to(device)

                    # Finetune on downstream task
                    if finetune:
                        down_finetune(model=model, finetune_epochs=5, pre_type=pre_type, train_loader=down_train, 
                                      down_lr=0.001, classifier=classifier, is_vit=is_vit,
                                      log_interval=100, save_models=False, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)
                
                    # Test on downstream task
                    result_down, dist_down = test(pre_type=pre_type, model=model, test_loader=down_test, classifier=classifier, is_vit=is_vit, stage='Down',
                                                  save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)

                    # Test on downstream jigsaw mixed data
                    result_down_jig16, dist_down_jig16 = test(pre_type=pre_type, model=model, test_loader=jig_loader_16, classifier=classifier, is_vit=is_vit, stage='Down',
                                                              save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device, jig=True)
                    result_down_jig8, dist_down_jig8 = test(pre_type=pre_type, model=model, test_loader=jig_loader_8, classifier=classifier, is_vit=is_vit, stage='Down',
                                                  save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device, jig=True)

                    # Views Embeddings distance 
                    result_3simclr_dist = eval_views_embed_dist(model=model, is_vit=is_vit, test_loader=views_dist_loader, 
                                                       save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)
                    result_pair_dist_gray = eval_views_embed_dist(model=model, is_vit=is_vit, test_loader=gray_pair_dist_loader, 
                                                       save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)
                    result_pair_dist_hflip = eval_views_embed_dist(model=model, is_vit=is_vit, test_loader=hflip_pair_dist_loader, 
                                                       save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)
                    result_pair_dist_rdm_crop = eval_views_embed_dist(model=model, is_vit=is_vit, test_loader=rdm_rcrop_pair_dist_loader, 
                                                       save_models=save_models, save_path=None, verbose=False, log=True, logger=logger, epoch=epoch, device=device)
                    
                    wandb.log({
                        "jig_16_acc":result_down_jig16,
                        "dist_jig_16":dist_down_jig16,
                        "jig_8_acc":result_down_jig8,
                        "dist_jig_8":dist_down_jig8,
                        
                        "dist_batch":dist_pre,
                        "dist_3_simclr":result_3simclr_dist,
                        "dist_gray":result_pair_dist_gray,
                        "dist_hflip":result_pair_dist_hflip,
                        "dist_rcrop":result_pair_dist_rdm_crop,
                    })
                    print('Pre: %.3f, Down: %.3f, Gray: %.3f, Hflip: %.3f, Rcrop: %.3f' % (dist_pre, dist_down, result_pair_dist_gray, result_pair_dist_hflip, result_pair_dist_rdm_crop))

                    # Save all results
                    if pre_dataset not in ['noise', 'fractal']:
                        scores[scores_idx].append([result_pre, result_bias, result_down, result_3simclr_dist])
                        distances.append([dist_pre, dist_down, result_pair_dist_gray, result_pair_dist_hflip, result_pair_dist_rdm_crop])
                    else :
                        scores[scores_idx].append([result_bias, result_down, result_3simclr_dist])
                    if scores_idx == 0: 
                        scores_epochs.append(epoch+1)
                        wandb.log({'epoch':epoch+1})

                '''scheduler.step()'''
        
            ### Specific distance analysis, temp so not in visualize
            distances = np.vstack(distances)
            labels = ['pre', 'down', 'gray', 'hflip', 'rcrop']
            fig, ax1 = plt.subplots()
            ax1.plot(scores_epochs, distances[:, 0], 'g', label='pre')
            ax1.plot(scores_epochs, distances[:, 1], 'k', label='down')
            ax2 = ax1.twinx()
            ax2.plot(scores_epochs, distances[:, 2], 'b', label='gray')
            ax2.plot(scores_epochs, distances[:, 3], 'r', label='hflip')
            ax2.plot(scores_epochs, distances[:, 4], 'orange', label='rcrop')

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='center right')
            plt.title('{}__distances_on_CIFAR10'.format(modelnames[scores_idx]))
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Distance (non augmented same batch)')
            ax2.set_ylabel('Distance (pair of aug/non-aug)')
            plt.tight_layout()
            plt.savefig(os.path.join('scores', '{}_distances_on_CIFAR10.png'.format(modelnames[scores_idx])), dpi=300)
            ### Specific distance analysis, temp so not in visualize

            scores_idx += 1
            wandb.finish()

        score_table = pd.DataFrame(scores, index=modelnames, columns=scores_epochs)

        return score_table


if __name__ == '__main__':
    # Create folders to store results and saved models
    if not os.path.isdir('scores'): os.mkdir('scores')
    if not os.path.isdir('model'): os.mkdir('model')

    # Create & Assemble models to be tested
    res18 = resnet18(num_classes=10)
    res50 = resnet50(num_classes=10)
    ResNet18_for_CIFAR = modify_resnet_model(res18)
    vit = ViT(hidden=512, mlp_hidden=512*4, img_size=32, patch=8)
    
    models_to_compare = [] # = ['ViT contrastive', 'ResNet50 supervised', ...]
    models_to_compare.append(vit)
    models_to_compare.append(ResNet18_for_CIFAR)
    model_names = ['ViT_for_CIFAR', 'ResNet18_for_CIFAR']

    scores = main(models2compare=models_to_compare, train_epochs=100, test_interval=5, 
                  save_models=False, experiment_id='test_both_aug', modelnames=model_names)
    pd.DataFrame(scores.T).to_excel(os.path.join('scores', 'FULL.xlsx'))
    visualize(scores, model_names, save=True, pre_data='not_noise')

'''
Notes:

# Loaders :
    # Custom Counterfactual Dataloader
    counter_train_sup, counter_test_sup = load_data(dataset='counterfactual', resize_image=False) # TODO: counterfactual supervised
    counter_train_cont, counter_test_cont = load_data(dataset='counterfactual', n_views=2, resize_image=False) # TODO: counterfactual contrastive

    # Pre Noise Dataloader
    contrastive_path = load_noise('stylegan-oriented', 'feature_vis-random') # 1+ folders to use in train
    pre_train_cont, pre_test_cont = load_data(dataset='noise', bs=64, n_views=2, noise_path=contrastive_path, 
                                                resize_image=False, shuffle_noise=True)
    # Call main
    scores = main(models2compare=models_to_compare, train_epochs=20, test_interval=2, pre_type='contrastive', 
                  pre_dataset='noise', finetune=True, save_models=False, experiment_id='noise', modelnames=model_names)
    
    # Fractal
    fractal_path = load_fractal()
    pre_train_cont, pre_test_cont = load_data(dataset='fractal', bs=64, n_views=2, noise_path=fractal_path, 
                                                resize_image=False, shuffle_noise=True)
    # Call main
    scores = main(models2compare=models_to_compare, train_epochs=20, test_interval=2, pre_type='contrastive', 
                  pre_dataset='fractal', finetune=True, save_models=False, experiment_id='fractal_tiny', modelnames=model_names)


# Integrate Checkpoints?

# If finetune :
    test once without any finetune? or do that in separate code runs (i.e. one with no finetune, and other with finetune)
    track test score throughout finetune (prevent having to modify script by simply saivng model at diff finetune epochs?)

# Feature vis at each test_interval?


# Model examples
    model_1 = CNN.load_resnet50(num_classes=1000, pretrained=True)
    model_2 = transformer.load_vit('vit_base_patch16_224', 1000, True)

# Performance on shape bias eval:
    ResNet50 - Shape Bias: 0.7043, Accuracy: 0.0898
    ViT - Shape Bias: 0.5431, Accuracy: 0.7336

'''
