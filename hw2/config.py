import argparse


def str2bool(s):
    if s in ('true', 'True'):
        return True
    elif s in ('false', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--model_path', default='model.path', type=str)
    parser.add_argument('--info_path', default='result_info.pickle', type=str)
    parser.add_argument('--num_classes', default=197, type=int)
    parser.add_argument('--use_pretrained', default=True, type=str2bool)
    parser.add_argument('--feature_extract', default=False, type=str2bool)
    parser.add_argument('--train_label_info_file_name',
                        default='./data/new_cars_train/train.pickle', type=str)
    parser.add_argument('--test_label_info_file_name',
                        default='./data/new_cars_test/test.pickle', type=str)
    parser.add_argument('--train_root_dir',
                        default='./data/new_cars_train', type=str)

    parser.add_argument('--test_root_dir',
                        default='./data/new_cars_test', type=str)
    parser.add_argument('--cuda', default='0', type=str, required=True)
    return parser
