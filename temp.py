
utils.attack_model(m, loss_fn, 1000, utils.valset, 400);


# In[52]:



# In[67]:


for i in [1, 2, 5, 10]:
    print(i)
    m_ = DeltaEnsemble(m, n_neighb = i, eps = 0.6)
    m_.eval()
    utils.attack_model(m_, loss_fn, int(40000 // max(1, i)), utils.valset, 400)
#     break


# In[68]:


m_ = DeltaEnsemble(m, n_neighb = 10)

import matplotlib.pyplot as plt
import torchattacks
atk = torchattacks.PGD(m_, eps=0.1, alpha=1/255, steps=400, random_start=False)


# In[69]:


from tqdm.notebook import tqdm
wrong = []

for k, (x, label) in enumerate(tqdm(utils.valset)):
    m_.eval()
    x = x.unsqueeze(0).cuda()    
    adv_images = atk(x, torch.tensor(label).unsqueeze(0).cuda())    
    if m_(x).argmax().item() != m_(adv_images).argmax().item():
        wrong.append(k)
        break


# In[70]:


# k = 35
x, label = utils.valset[k]
x = x.unsqueeze(0).cuda()
adv_images = atk(x, torch.tensor(label).unsqueeze(0).cuda())


# In[71]:


x_ = m_._get_neighb_uniform(x, 1000)
pred_ = m(x_.squeeze(1))
plt.plot(pred_.cpu().detach().T)
plt.show()


# In[72]:


adv_images_ = m_._get_neighb_uniform(adv_images, 1000)
pred_adv_images_ = m(adv_images_.squeeze(1))
plt.plot(pred_adv_images_.cpu().detach().T)
plt.show()

from collections import Counter

print(Counter(pred_.cpu().detach().argmax(1).tolist()))
print(Counter(pred_adv_images_.cpu().detach().argmax(1).tolist()))


# In[73]:


import random

fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(m(x.repeat(2, 1,1,1)).tolist()[0])
axs[0, 1].plot(m(adv_images.repeat(2, 1,1,1)).tolist()[0])
axs[1, 0].matshow(x.squeeze().cpu())
axs[1, 1].matshow(adv_images.squeeze().cpu())
axs[2, 0].matshow(x_[random.randint(0, len(adv_images_))].squeeze().cpu())
axs[2, 1].matshow(adv_images_[random.randint(0, len(adv_images_))].squeeze().cpu())
axs[3, 0].plot(m_(x.repeat(2, 1,1,1)).tolist()[0])
axs[3, 1].plot(m_(adv_images.repeat(2, 1,1,1)).tolist()[0])

