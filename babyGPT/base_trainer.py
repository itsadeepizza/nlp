import torch
import random
import math
import os, datetime
from torch.utils.tensorboard import SummaryWriter
# import torchsummary
import time
import numpy.random
import numpy as np
from config import selected_config as conf
import inspect
import glob
import re

class BaseTrainer():

    def __init__(self):
        # Init random seed
        seed = conf.get('SEED')
        if seed is None:
            seed = random.randint(0, 9999999)
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        self.models = []
        # set device
        device = conf.get('DEVICE', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device

        # Get idx
        self.idx = conf.get('LOAD_IDX', 0)
        # Set learning rate
        self.update_lr()

    def count_parameters(self):
        models_n_params = []
        for model in self.models:
            # Count parameters for each model
            models_n_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        return sum(models_n_params)

    def update_lr(self):
        # Generally you need to override this method in Trainer class
        default_lr = 1e-5
        self.lr = default_lr
        print(f"Default LR is {default_lr:.3g}. Override class `update_lr` for setting custom "
              f"values")

    def init_logger(self):
        # TENSORBOARD AND LOGGING

        def generate_docker_style_name():
            """Generate name of the log folder"""
            import random
            # name has to be randomly generated even if we had chosen a random seed
            random_state = random.getstate()
            random.seed()

            # Select a random adjective from a list of common adjectives
            adjectives = ['adorable', 'beautiful', 'clean', 'drab', 'elegant', 'fancy', 'glamorous', 'handsome', 'long', 'magnificent', 'old', 'plain', 'quaint', 'sparkling', 'ugliest', 'unsightly', 'wide', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'gray', 'black', 'white', 'pink', 'brown']
            adjective = random.choice(adjectives)

            # Select a random noun from a list of common nouns
            nouns = ['person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand', 'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government', 'company', 'number', 'group', 'problem', 'fact']
            noun = random.choice(nouns)

            # Add current datetime
            now_str = datetime.datetime.now().strftime("%y-%m-%d-%HH%M")
            # Combine the adjective, noun and date to form the name
            name = now_str + "_" + adjective + '_' + noun
            random.setstate(random_state)
            return name

        # Create directories for logs

        name = generate_docker_style_name()

        self.summary_dir = conf.ROOT_RUNS + "/runs/summary/" + name
        self.models_dir = conf.ROOT_RUNS + "/runs/models/" + name
        self.test_dir = conf.ROOT_RUNS + "/runs/test/" + name
        self.img_dir = conf.ROOT_RUNS + "/runs/img/" + name

        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # Init writer
        self.writer = SummaryWriter(self.summary_dir)

        self.mean_train_loss = 0
        self.timer = 0

        # Log model info
        # model_stat = f"```{str(torchsummary.summary(self.model()))}```"
        # self.writer.add_text("Torchsummary", model_stat)
        # self.writer.add_text("Model name", str(self.model.__name__))
        # self.writer.add_text("Model code", "```  \n" + inspect.getsource(self.model) + "  \n```")

        # Log time
        self.writer.add_text("Time", datetime.datetime.now().strftime("%a %d %b %y - %H:%M"))
        # Log a formatted configuration on tensorboard (only uppercase public parameters
        config_as_table = "\n".join([f"{param:>26} = {conf.__getattribute__(param)}" for param in dir(conf) if (not param.startswith("_") and param.upper() == param)])
        self.writer.add_text("Configuration", config_as_table)
        n_parameters = self.count_parameters()
        self.writer.add_text("Parameters: ", f"{n_parameters:,}")

    def log_tensorboard(self):
        # Log always loss on train, steps per second and current learning rate
        self.writer.add_scalar("train_loss", self.mean_train_loss / conf.INTERVAL_TENSORBOARD, self.idx)
        tot_time = time.time() - self.timer
        self.timer = time.time()
        self.writer.add_scalar("steps_for_second", conf.INTERVAL_TENSORBOARD / tot_time, self.idx)
        self.writer.add_scalar("lr", self.lr, self.idx)
        self.mean_train_loss = 0

    def save_models(self):
        for model in self.models:
            name = model.__class__.__name__
            path = os.path.join(self.models_dir, name)
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(model.state_dict(), f"{path}/{name}_{self.idx}.pth")

    def remove_old_models(self):
        """Remove the second_last model in each model directory"""
        for model in self.models:
            name = model.__class__.__name__
            path = os.path.join(self.models_dir, name)
            model_list = glob.glob(path + '/*.pth')
            idx_list = [int(re.search('_([0-9]+).pth', os.path.basename(file)).groups()[0]) for file in model_list]
            if len(model_list) > 1:
                # ignore max index
                idx_list[np.argmax(idx_list)] = 0
                second_last = model_list[np.argmax(idx_list)]
                os.remove(second_last)

    @staticmethod
    def plot_to_tensorboard(fig):
        """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function

        Parameters:
            writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
            fig (matplotlib.pyplot.fig): Matplotlib figure handle.
            step (int): counter usually specifying steps/epochs/time.
        """

        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
        img = img / 255.0
        img = np.swapaxes(img, 0, 2)  # if your TensorFlow + TensorBoard version are >= 1.8
        img = np.swapaxes(img, 1, 2)  # elsewhere image is inverted
        return img

    def plot_small_module(self, module: torch.nn.Module, step: int = 0):
        """Plot weights for small modules"""
        import math
        import matplotlib.pyplot as plt
        import numpy as np
        import io

        def normalize(l):
            """extract detached weights from layer"""
            w = l.weight.detach().to("cpu").abs()
            if w.dim() == 1:
                w = w.unsqueeze(1)
            return w

        layers = [l for l in module.named_modules()][1:]  # first one is the module itself
        # parameters = [x[0] for x in self.encoder.named_parameters()] # for bias and ALL parameters in the module
        n_layers = len(layers)
        n_rows = math.ceil(n_layers / 2)
        fig, ax = plt.subplots(n_rows, 2, figsize=(14, n_rows * 7))
        for i, (name, layer) in enumerate(layers):
            ax_ = ax[i // 2][i % 2]
            ax_.imshow(normalize(layer), cmap="gray", vmin=0, vmax=0.3)
            ax_.set_title(name, fontsize=20)
        module_name = module.__class__.__name__
        fig.suptitle(module_name, fontsize=26)
        fig.canvas.draw()
        fig.savefig(os.path.join(self.img_dir, f"{module_name}_{step:06}.png"))
        img = self.plot_to_tensorboard(fig)
        plt.close(fig)
        return img
