from utils import *
print('seg')
seg = fingernailseg()
# show random example from training set
#seg.plot_example(np.random.randint(seg.X_train.__len__()))
# create U-Net model
print('create_unet')
seg.create_unet()
print('fit');
seg.fit()
seg.load_model()