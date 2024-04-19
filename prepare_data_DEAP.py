# This is the processing script of DEAP dataset
import mne
import _pickle as cPickle

from train_model import *
from scipy import signal


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.file_name = args.file_name
        # self.label_type = args.label_type

        # self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
        #                        'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
        #                        'CP2', 'P4', 'P8', 'PO4', 'O2']
        
        self.original_order = ['AF3', 'FPz', 'AF4', 'F9', 'F7', 'FC4', 'F10', 'T7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'CP5', 
                  'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC6', 'FT8', 'C2', 'Cz', 'C1', 'POz', 'CP2', 'CP4', 'CP6', 'C6', 'T8', 'TP7', 'CP3',
                  'CP1', 'CPz', 'Pz', 'P4', 'P2', 'TP10', 'TP8', 'P5', 'P3', 'P1', 'PO3', 'PO10', 'P6', 'P8', 'PO4', 'P10', 'P9', 'P7', 'PO7', 'O2', 
                  'Oz', 'PO9', 'FT9', 'PO8', 'C5', 'FT10', 'TP9', 'O1']
        
        self.graph_novel = [['AF3', 'FPz', 'AF4'], ['F9', 'F7', 'F5', 'F3', 'FT9', 'FT7'], ['F4', 'F6', 'F8', 'F10', 'FT8', 'FT10'], 
                            ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2'], ['FC5', 'FC3'], ['FC4', 'FC6'], ['C5', 'C1', 'Cz', 'C2', 'C6'], 
                            ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'], ['P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz','PO4'],
                            ['T7', 'TP7', 'TP9', 'P7', 'P9'], ['T8', 'TP8', 'TP10', 'P8', 'P10'], ['PO7', 'PO9', 'O1', 'Oz', 'O2', 'PO8', 'PO10']]

        self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz', 'Cz', 'Pz', 'Oz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                               ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]
        self.TS = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3','O1',
                   'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
        self.graph_type = args.graph_type

    # function for loading the raw data and preparing it into labels and epochs
    def load_data(self, event_dict, class_labels, tmin, tmax, expand=True):
        """
        This function loads the raw data and extracts the epochs and corresponding labels
        Parameters
        ----------
        event_dict: the dictionary containing the events
        class_labels: the labels to extract
        tmin: the minimum time in window
        tmax: the maximum time in window
        expand: expand the dimensions for deep learning models
        """
        # load the preprocessed data
        raw = mne.io.read_raw_fif(self.data_path + "/" + self.file_name + ".fif", preload=True)
        # extract the event information from the raw data
        events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)
        # extract epochs from the raw data
        epochs = mne.Epochs(raw, events, event_id=class_labels, tmin=tmin, tmax=tmax, baseline=None, preload=True)
        # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
        data = epochs.get_data()*1e6 # format is in (trials, channels, samples)
        # extract and normalize the labels ensuring they start from 0
        labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        # reorder the EEG channel to build the local-global graphs
        data = self.reorder_channel(data=data, graph=self.graph_type)
        # define the number of channels and samples
        chans, samples = data.shape[1], data.shape[2]
        # expand one dimension for deep learning(CNNs)
        if expand:
            data = np.expand_dims(data, axis=-3)
        # print the shape of the epochs and labels
        print('------------------------------------------------------------------------')
        print("Shape of data after expanding dimensions: ", data.shape)
        print('------------------------------------------------------------------------')
        print("Shape of labels: ", labels.shape)
        print('------------------------------------------------------------------------')
        print("Labels: ", labels)
        print('------------------------------------------------------------------------')
        print("Data and labels prepared!")
        print('------------------------------------------------------------------------')
        # convert labels to one-hot encodings. This is required for the loss function used in the model
        # return the data
        # return data, labels, chans, samples
        self.save(data, labels)

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order
        elif graph == 'TS':
            graph_idx = self.TS
        elif graph == 'novel':  # new graph design
            graph_idx = self.graph_novel

        idx = []
        if graph in ['BL', 'TS']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
            print('idx:', idx)
        return data[:, idx, :]

    def save(self, data, label):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}'.format(self.data_path)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = self.file_name + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()
