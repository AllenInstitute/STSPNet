import numpy as np
from scipy.stats import truncexpon


class StimGenerator:
    def __init__(self,
                 image_set='A',       # image set ('A', 'B', 'C', or 'D')
                 seq_length=50000,    # length of each "trial" (ms)
                 batch_size=128,      # number of "trial" batches
                 time_step=250,       # simulation time step (ms)
                 image_pres_dur=250,  # image presentation (ms)
                 delay_dur=500,       # delay duration (ms)
                 reward_on=0,         # reward window (ms)
                 reward_off=250,
                 rep_min=4,           # minimum number of repeats (inclusive)
                 rep_max=12,          # maximum number of repeats (exclusive)
                 omit_frac=0.0,       # fraction of omitted flashes
                 seed=1):             # random seed

        self.image_set = image_set
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.time_step = time_step
        self.image_steps = image_pres_dur // time_step
        self.delay_steps = delay_dur // time_step
        self.reward_on_steps = reward_on // time_step
        self.reward_off_steps = reward_off // time_step
        self.min_repeat = rep_min
        self.max_repeat = rep_max
        self.omit_frac = omit_frac
        self.seed = seed

        # Get image features
        self.feature_dict, self.num_images = self.load_image_features()

        # Number of steps to mask loss
        self.mask_steps = 4*(self.image_steps + self.delay_steps)

    def load_image_features(self):
        """
        Loads feature dict for a given image set.
        """
        if self.image_set == 'A':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_A_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([1, 5, 2, 4, 0, 7, 6, 3])
            # image_ticks = ('077', '062', '066', '063', '065',
            #                '069', '085', '061')
        elif self.image_set == 'B':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_B_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([0, 5, 7, 1, 3, 6, 4, 2])
            # image_ticks = ('012', '057', '078', '013', '047',
            #                '036', '044', '115')
        elif self.image_set == 'C':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_C_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([3, 2, 6, 1, 5, 7, 4, 0])
            # image_ticks = ('073', '075', '031', '106', '054',
            #                '035', '045', '000')
        elif self.image_set == 'D':
            feature_dict = np.load(
                './cifar/features/image_set_cnn_D_seed_'+str(self.seed)+'.npy')
            img_ind_swap = np.array([3, 1, 5, 4, 7, 6, 2, 0])
            # image_ticks = ('072', '114', '091', '087', '034',
            #                '024', '104', '005')

        # Resort by image detectability
        feature_dict[:-1, :] = feature_dict[img_ind_swap, :]

        return feature_dict, len(feature_dict)-1

    def _generate_num_repeat(self, scale=2.0):
        """
        Helper function, randomly generate number of repetition for an image.
        Choose an integer with exponential distribution between [min, max].

        Output:
          integer between [min_repeat, max_repeat)
        """
        min_switch = truncexpon.rvs(b=(self.max_repeat - self.min_repeat) /
                                    scale, loc=self.min_repeat, scale=scale).astype(int)

        return min_switch

    def generate_batch(self):
        """
        Generate one batch of inputs

        Args:
            batch_len: number of trials per batch
            feature_dict: list of features
        """
        # Initialize output arrays
        image_array = np.zeros(
            (self.batch_size, self.seq_length // self.time_step), dtype='int')
        label_array = np.zeros(
            (self.batch_size, self.seq_length // self.time_step, 1), dtype='float32')
        mask_array = np.zeros(
            (self.batch_size, self.seq_length // self.time_step, 1), dtype='float32')
        # Mask out blank flashes
        mask_array[:, ::(self.image_steps+self.delay_steps)] = 1

        # Loop over each element in batch
        for i in range(self.batch_size):
            last_image = -1
            image = np.array([], dtype='int')
            while len(image) < (self.seq_length // self.time_step):
                # Generate random image and number of repeats
                image_num = np.random.randint(self.num_images)
                repeat_num = self._generate_num_repeat()

                image_repeat = np.tile(
                    [image_num]*self.image_steps+[self.num_images]*self.delay_steps, repeat_num)

                if image_num != last_image:
                    # Go trial
                    label_array[i, len(
                        image)+self.reward_on_steps:len(image)+self.reward_off_steps] = 1
                    last_image = image_num
                else:
                    # Catch trial
                    label_array[i, len(image)] = -1

                image = np.concatenate((image, image_repeat))

            # Use only seq_length
            image_array[i, :] = image[:(self.seq_length // self.time_step)]
            input_array = self.feature_dict[image_array, :]

        # Omitted flashes
        if self.omit_frac > 0:
            pad = self.image_steps + self.delay_steps
            omit_array = (np.random.binomial(1, self.omit_frac, size=image_array.shape)) & \
                (image_array != self.num_images) & (label_array.squeeze() == 0) & \
                (np.pad(label_array.squeeze(), ((0, 0), (0, pad)),
                        mode='constant')[:, pad:] == 0)

            # Re-assign omitted flashes here
            image_array[np.where(omit_array)[0], np.where(
                omit_array)[1]] = self.num_images
            input_array[np.where(omit_array)[0], np.where(
                omit_array)[1]] = self.feature_dict[-1, :]
        else:
            omit_array = np.zeros(
                (self.batch_size, self.seq_length // self.time_step), dtype='int')

        # Set first image to always be zero
        label_array[:, :self.reward_off_steps] = 0

        # Mask out first part of stimulus
        mask_array[:, :self.mask_steps] = 0

        return input_array, label_array, image_array, mask_array, omit_array
