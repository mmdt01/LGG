from cross_validation import *
from prepare_data_DEAP import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    # parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='subject_4')
    parser.add_argument('--file-name', type=str, default='s4_preprocessed')
    # parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    # parser.add_argument('--label-type', type=str, default='L', choices=['A', 'V', 'D', 'L'])
    parser.add_argument('--segment', type=int, default=4)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=250) # change to 250
    parser.add_argument('--scale-coefficient', type=float, default=1)
    parser.add_argument('--input-shape', type=tuple, default=(1, 64, 751)) # change to (1, 64, 751)
    # parser.add_argument('--data-format', type=str, default='eeg')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--patient', type=int, default=20)
    parser.add_argument('--patient-cmb', type=int, default=8)
    parser.add_argument('--max-epoch-cmb', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16) # default 64
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='LGGNet')
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--graph-type', type=str, default='novel', choices=['fro', 'gen', 'hem', 'BL', 'TS', 'novel'])
    parser.add_argument('--hidden', type=int, default=32)

    # CHANGE THE PARAMETER ARGUMENTS FIRST

    ######## Reproduce the result using the saved model ######
    # parser.add_argument('--reproduce', action='store_true')

    # create an event dictionary
    event_dict = {
        'motor execution up': 1,
        'motor execution down': 2,
        'visual perception up': 3,
        'visual perception down': 4,
        'imagery up': 5,
        'imagery down': 6,
        'imagery and perception up': 7,
        'imagery and perception down': 8
    }

    ######## Run the code ########
    args = parser.parse_args()
    # sub_to_run = np.arange(args.subjects)
    pd = PrepareData(args)
    pd.load_data(event_dict, class_labels=[5,6], tmin=0, tmax=3, expand=True)

    cv = CrossValidation(args)
    seed_all(args.random_seed)

    cv.n_fold_CV(fold=3, shuffle=True)
