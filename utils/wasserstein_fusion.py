import copy

import ot
import torch
import numpy as np
from utils.ground_metric import GroundMetric
import models

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c


def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy=True, float64=False):
    if activations is None:
        # returns a uniform measure
        return np.ones(cardinality) / cardinality
        if not args.unbalanced:
            print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality) / cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)


def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')

    # cumulative_T_var = None
    T_var = None
    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new_network = models.mlp.MLP256().cuda()
    # new_network = ablation_study_copy.deepcopy(networks[1])
    # avg_aligned_layers = []
    avg_aligned_layers = new_network.state_dict()

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
        # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

            aligned_wt = fc_layer0_weight_data
        else:

            # print("shape of layer: model 0", fc_layer0_weight_data.shape)
            # print("shape of layer: model 1", fc_layer1_weight_data.shape)
            # print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                # print("ground metric is ", M)

            if args.skip_last_layer and idx == (num_layers - 1):
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    # avg_aligned_layers[layer1_name] = (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight
                    avg_aligned_layers[layer1_name] = (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight
                    # avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight)
                else:
                    # avg_aligned_layers.append((aligned_wt + fc_layer1_weight) / 2)
                    avg_aligned_layers[layer1_name] = ((aligned_wt + fc_layer1_weight) / 2)
                    # avg_aligned_layers[layer1_name] = ((aligned_wt + fc_layer1_weight) / 2)

                new_network.load_state_dict(avg_aligned_layers)
                # return avg_aligned_layers
                return new_network

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            # mu = mu / mu.sum()
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            # print(mu.sum(0), nu.sum(0))
            # print(mu, nu)
            assert args.proper_marginals

        cpuM = M.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        # T = ot.emd(mu, nu, log_cpuM)

        # if args.gpu_id != -1:
        #     T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
        # else:
        #     T_var = torch.from_numpy(T).float()
        T_var = torch.from_numpy(T).cuda().float()

        # torch.set_printoptions(profile="full")
        # print("the transport map is ", T_var)
        # torch.set_printoptions(profile="default")

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                marginals = torch.ones(T_var.shape[0]).cuda() / T_var.shape[0]
                marginals = torch.diag(1.0 / (marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = (1 / (marginals_beta + eps))
                # print("shape of inverse marginals beta is ", marginals_beta.shape)
                # print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                # print(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        # if args.debug:
        #     if idx == (num_layers - 1):
        #         print("there goes the last transport map: \n ", T_var)
        #     else:
        #         print("there goes the transport map at layer {}: \n ".format(idx), T_var)
        #
        #     print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

        # print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            # print("this is past correction for weight mode")
            # print("Shape of aligned wt is ", aligned_wt.shape)
            # print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            geometric_fc = ((1 - args.ensemble_step) * t_fc0_model +
                            args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2

        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        # avg_aligned_layers.append(geometric_fc)
        avg_aligned_layers[layer1_name] = geometric_fc

    new_network.load_state_dict(avg_aligned_layers)
    return new_network
    # return avg_aligned_layers

def _get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
    print('shape of layer_weight is ', layer_weight.shape)
    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()

    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
            np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
            np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist / importance_hist.sum())
        # print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist


def geometric_ensembling_modularized(args, networks, train_loader, test_loader, activations=None):
    # if args.geom_ensemble_type == 'wts':
    #     avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, activations,
    #                                                                 test_loader=test_loader)
    # elif args.geom_ensemble_type == 'acts':
    #     avg_aligned_layers = get_acts_wassersteinized_layers_modularized(args, networks, activations,
    #                                                                      train_loader=train_loader,
    #                                                                      test_loader=test_loader)
    # avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, activations, test_loader=test_loader)
    return get_wassersteinized_layers_modularized(args, networks, activations, test_loader=test_loader)
    # return get_network_from_param_list(args, avg_aligned_layers, test_loader)

