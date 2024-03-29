import json
import os
import sys
import pickle
import signal
import argparse
import traceback

import torch
import numpy as np




import dataset.loader as loader
import train.factory_contrastive as train_utils

import train.drop_entropy_eval_contrastive_var as drop_entropy_eval_contrastive_var



def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with Distributional Signatures")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="reuters",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters, rcv1, fewrel, 20newsgroup_bert]")
    # parser.add_argument("--bertbased", action='store_true', default=False, help='whether use the bert set, only for 20news')

    parser.add_argument("--n_train_class", type=int, default=15,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=11,
                        help="number of meta-test classes")

    # load bert embeddings for sent-level datasets (optional)
    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")
    parser.add_argument("--sleep_time", type=int, default=1,
                        help="Num. of sleeping time")
    parser.add_argument("--done_queue_limit", type=int, default=100,
                        help="Num. of done_queue limit to sleep")

    parser.add_argument("--bert", default=False, action="store_true",
                        help=("set true if use bert embeddings "
                              "(only available for sent-level datasets: "
                              "huffpost, fewrel"))
    parser.add_argument("--bert_cache_dir", default=None, type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))

    # task configuration
    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    # train/test configuration
    parser.add_argument("--train_epochs", type=int, default=1000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#tasks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    # settings for finetuning baseline
    parser.add_argument("--finetune_loss_type", type=str, default="softmax",
                        help="type of loss for finetune top layer"
                        "options: [softmax, dist]")
    parser.add_argument("--finetune_maxepochs", type=int, default=5000,
                        help="number epochs to finetune each task for (inner loop)")
    parser.add_argument("--finetune_episodes", type=int, default=10,
                        help="number tasks to finetune for (outer loop)")
    parser.add_argument("--finetune_split", default=0.8, type=float,
                        help="percent of train data to allocate for val"
                             "when mode is finetune")

    # model options
    parser.add_argument("--embedding", type=str, default="avg",
                        help=("document embedding method. Options: "
                              "[avg, tfidf, meta, oracle, cnn]"))
    parser.add_argument("--classifier", type=str, default="nn",
                        help=("classifier. Options: [nn, proto, r2d2, mlp]"))
    parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                        help=("auxiliary embeddings (used for fewrel). "
                              "Options: [pos, ent]"))

    # cnn configuration
    parser.add_argument("--cnn_num_filters", type=int, default=50,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")

    # nn configuration
    parser.add_argument("--nn_distance", type=str, default="l2",
                        help=("distance for nearest neighbour. "
                              "Options: l2, cos [default: l2]"))

    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[300, 300],
                        help=("hidden dimension of the proto-net"))

    # maml configuration
    parser.add_argument("--maml", action="store_true", default=False,
                        help=("Use maml or not. "
                        "Note: maml has to be used with classifier=mlp"))
    parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[300, 5],
                        help=("hidden dimension of the proto-net"))
    parser.add_argument("--maml_innersteps", type=int, default=10)
    parser.add_argument("--maml_batchsize", type=int, default=10)
    parser.add_argument("--maml_stepsize", type=float, default=1e-1)
    parser.add_argument("--maml_firstorder", action="store_true", default=False,
                        help="truncate higher order gradient")

    # lrd2 configuration
    parser.add_argument("--lrd2_num_iters", type=int, default=5,
                        help=("num of Newton steps for LRD2"))

    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))

    # aux ebd configuration (for fewrel)
    parser.add_argument("--pos_ebd_dim", type=int, default=5,
                        help="Size of position embedding")
    parser.add_argument("--pos_max_len", type=int, default=40,
                        help="Maximum sentence length for position embedding")

    # base word embedding
    parser.add_argument("--wv_path", type=str,
                        default="./",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    # options for the distributional signatures
    parser.add_argument("--meta_idf", action="store_true", default=False,
                        help="use idf")
    parser.add_argument("--meta_iwf", action="store_true", default=False,
                        help="use iwf")
    parser.add_argument("--meta_w_target", action="store_true", default=False,
                        help="use target importance score")
    parser.add_argument("--meta_w_target_lam", type=float, default=1,
                        help="lambda for computing w_target")
    parser.add_argument("--meta_target_entropy", action="store_true", default=False,
                        help="use inverse entropy to model task-specific importance")
    parser.add_argument("--meta_ebd", action="store_true", default=False,
                        help="use word embedding into the meta model "
                        "(showing that revealing word identity harm performance)")

    # training options
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--dropout_MC", type=float, default=0.1, help="drop rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--patience", type=int, default=50, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="test",
                        help=("Running mode."
                              "Options: [train, test, finetune]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")
    parser.add_argument('--output_repr', action='store_true', default=False,
                        help='output the representation to file output_repr.txt')
    parser.add_argument('--use_idk', action='store_true', default=False,
                        help='use idk. If yes, it will show all the results from 0 to 0.4 with interval 0.05')
    parser.add_argument('--use_human_idk', action='store_true', default=False,
                        help='use human idk. If yes, it will show all the results from 0 to 0.4 assuming the uncertain part is handed over to humans')
    parser.add_argument('--idk_ratio', type=float, default=0, help='the ratio of uncertainty')
    parser.add_argument('--is_class_num_small', type=int, default=0, help='whether the class num is smaller than 2')
    parser.add_argument('--modelmode', type=str, default="clur", help='[clur]')

    # parser.add_argument('--dropout', type=float, default=0.3, help='the probability for dropout [default: 0.3]')

    # uncertainty train related
    parser.add_argument('--feature_aug_mode', type=str, default=None, help='[dropout, shuffle, cutoff]')
    parser.add_argument('--use_unequal', action='store_true', default=False, help='whether do unequal contrastive')
    parser.add_argument('--unequal_type', type=int, default=0, help='0 is cross, 1 is only last')
    parser.add_argument('--use_equal', action='store_true', default=False, help='whether use euqal contrastive train')
    parser.add_argument('--AugLevel', type=str, default='sam', help='bat (batch), sam (sample)')
    parser.add_argument('--lowb', type=float, default=0.9, help='the lower boundary of alpha [0.51, 1]')
    parser.add_argument('--larger_augmargin', action='store_true', default=False, help='use the larger margin for the training or testing')
    parser.add_argument('--augmargin', type=float, default=0.1, help='the upper boundary of sec_alpha is lowb - augmargin')
    parser.add_argument('--sec_lowb', type=float, default=0.7, help='the upper boundary of sec_alpha is lowb - augmargin')

    parser.add_argument('--contrastive', action='store_true', default=False, help='use the contrastive learning')
    parser.add_argument('--contrast_weight', type=float, default=0.1, help='weight for the contrastive loss')

    # uncertainty test related
    parser.add_argument('--use_auc_roc', action='store_true', default=False, help='whether use the auc_roc to evaluate')
    parser.add_argument('--tev_mode', type=str, default='acltev', help='the mode to evaluate the testing results!')
    parser.add_argument('--use_unequal_tev', action='store_true', default=False, help='whether use the aug_eval')

    parser.add_argument('--drop-mask', type=int, default=5,
                        help='the number of masks used for dropout bayesian method [default: 5]')
    parser.add_argument('--drop-num', type=int, default=100,
                        help='the number of the experiments used for dropout bayesian method [default: 100]')
    parser.add_argument('--use_sec_unc', action='store_true', default=False, help='whether use the acl evaluation mode')
    # parser.add_argument('--MdistTest', action='store_true', default=False, help='if test the Mdist in the ')
    parser.add_argument('--calmeanvar', action='store_true', default=False,
                        help='if use self-ensemble tev, cannot exist with emnlptev')
    parser.add_argument('--individual_eval', action='store_true', default=False,
                        help='use single socre or two scores to eval')
    parser.add_argument('--te_measure', type=str, default='1max', help='the maximum value of the softmax vector')


    # below is added by me
    parser.add_argument("--use_dynamic_classifier", action="store_true", default=False,
                        help="disable tqdm")

    parser.add_argument("--use_original_ref", action="store_true", default=False,
                        help="use original reference")
    parser.add_argument("--use_no_further_projection", action="store_true", default=False,
                        help="use further original reference, change NAACL for comparison with original model")
    parser.add_argument("--unequal_lower_boundary", type=float, default=0.75,
                        help='the parameter for the loss of metric learning [default: 0.75]')
    parser.add_argument("--embedding_dropout",  action="store_true", default=False,
                        help='whether use the embedding dropout in the embedding layer')


    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if args.classifier != "nn" and attr[:3] == "nn_":
            continue
        if args.embedding != "meta" and attr[:5] == "meta_":
            continue
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "mlp" and attr[:4] == "mlp_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if "pos" not in args.auxiliary and attr[:4] == "pos_":
            continue
        if not args.maml and attr[:5] == "maml_":
            continue
        print("\t{}={}".format(attr.upper(), value))



def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    assert (args.use_unequal and args.use_equal)==False

    if args.larger_augmargin:
        assert (args.lowb - args.augmargin - args.sec_lowb) >= 0
        assert (1 - args.sec_lowb) >= 0
        assert (1 - args.lowb + args.augmargin) >= 0


    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model0
    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)

    model = {}

    if args.modelmode == "clur" and args.contrastive:
        import embedding.factory_contrastive as ebd
        import classifier.factory_contrastive as clf
    model["ebd"] = ebd.get_embedding(vocab, args)
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)


    if args.mode == "test" and args.snapshot != "":
        print("Start of testing. Restore the best weights")

        # restore the best saved model
        model['ebd'].load_state_dict(torch.load(args.snapshot + '.ebd'))
        model['clf'].load_state_dict(torch.load(args.snapshot + '.clf'))


    if args.mode == "train":
        train_utils.train(train_data, val_data, model, args)


    elif args.mode == "finetune":
        # sample an example from each class during training
        way = args.way
        query = args.query
        shot = args.shot
        args.query = 1
        args.shot= 1
        args.way = args.n_train_class
        train_utils.train(train_data, val_data, model, args)
        args.shot = shot
        args.query = query
        args.way = way


    if args.mode == 'test' or args.mode == 'train':
        result = drop_entropy_eval_contrastive_var.few_test(test_data, model, args,
                                                            args.test_episodes)


    save_json_path = args.snapshot[:-4] + 'result.json'
    print(f"save path to {save_json_path}")

    for attr, value in sorted(result.items()):
        if isinstance(value, np.ndarray):
            print(value)
            value = value.tolist()
            new_value = []
            for ele in value:
                new_value.append(float(ele))

        else:
            value = float(value)

        result[attr] = value

    print(result)
    with open(save_json_path, "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

    exit(0)
