import torch
import torch.nn as nn
from loader import train, train_dataloader, train_dataset
from model import Model

tensorize = train.tensorize

dataloader = train_dataloader
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO: fix below
vocab = {word: tensorize(word) for word, n in train.vocab.items()}
use_tensorboard = True
use_existing_model = True
save_freq = 10000
model_file = "model.pth"
model = Model(device=device, len_voc=len(vocab))
if use_existing_model: 
    print(f"loading model from file {model_file}")
    try:
        model.load_state_dict(torch.load(model_file))
    except FileNotFoundError:
        print(f"can't find {model_file}, creating a new model to train")

opt = torch.optim.SGD(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    import PIL.Image
    import io
    import matplotlib 

    writer = SummaryWriter()
    writer.add_graph(model, next(iter(dataloader))[0])
    matplotlib.use('Agg')

    def gen_plot(embedding, word1, word2, word3, step):
        with torch.no_grad():
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            (v1,v2,v3) = (embedding(tensorize(word1)).tolist(), \
                       embedding(tensorize(word2)).tolist(), \
                       embedding(tensorize(word3)).tolist())
            #only 3 dim of embedding
            v1 = [v1[0],v1[1],v1[2]]
            v2 = [v2[0],v2[1],v2[2]]
            v3 = [v3[0],v3[1],v3[2]]

            v1.insert(0,0)
            v1.insert(0,0)
            v1.insert(0,0)
            v2.insert(0,0)
            v2.insert(0,0)
            v2.insert(0,0)
            v3.insert(0,0)
            v3.insert(0,0)
            v3.insert(0,0)
            X_0,Y_0,Z_0, X,Y,Z = zip(v1,v2,v3)
            ax.quiver(X_0[0],Y_0[0],Z_0[0],X[0],Y[0],Z[0], normalize = True,color="blue")
            ax.quiver(X_0[1],Y_0[1],Z_0[1],X[1],Y[1],Z[1], normalize = True,color="red")
            ax.quiver(X_0[2],Y_0[2],Z_0[2],X[2],Y[2],Z[2], normalize = True,color="green")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            plt.title(f"{word1}(blue) - {word2}(red) - {word3}(green)")
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            plt.close('all')
            return buf

with torch.no_grad():
    print("before training")
    emb_is = model.embedding(tensorize("è"))
    emb_was = model.embedding(tensorize("era"))
    print(emb_is)
    print(emb_was)
    print(torch.dot(emb_is,emb_was))
import time
# training loop
tot_iters = 0
for epoch in range(50):
    print("epoch started: ",epoch)
    data = iter(dataloader)
    for i, (words, label) in enumerate(data):
        out = model(words)
        loss = criterion(out, label.to(device))
        if i==100:
            tic = time.time()
        if i==1100:
            toc = time.time()
            print("time in ms for batch:", (toc-tic))
        if i % 1000 == 0:
            print(f"epoch {epoch} iteration {i} loss:",loss.item())
            if use_tensorboard:
                index = epoch*tot_iters + i
                plot_buf = gen_plot(model.embedding,"è", "era", "aeroporto",index)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).unsqueeze(0)[0]
                writer.add_image('plot embedding', image, index)
                writer.add_scalar('train_loss', loss.item(), index)
        if i % save_freq == 0:
            torch.save(model.state_dict(),model_file)
            print("model saved to file")
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i>tot_iters:
            tot_iters = i
    # TODO save model / PCA prima di plottare

if use_tensorboard:
    writer.close()

with torch.no_grad():
    print("after training")
    emb_is = model.embedding(tensorize("is"))
    emb_was = model.embedding(tensorize("was"))
    print(emb_is)
    print(emb_was)
    print(torch.dot(emb_is,emb_was))

#type "tensorboard --logdir=runs" in terminal