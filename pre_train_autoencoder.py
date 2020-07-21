import random
import torch


torch.manual_seed(0)
random.seed(0)

def pre_train_autoencoder(hp,
                          encoder,
                          encoder_optim,
                          decoder,
                          decoder_optim,
                          embs,
                          dev_words=None):

    if dev_words == None:
        emb_list = [embs[word] for word in embs.wv.vocab]
        random.shuffle(emb_list)
        eval_inputs = torch.split(torch.FloatTensor(emb_list[:hp.pta_dev_num]), hp.pta_batch_size)
        train_inputs = torch.split(torch.FloatTensor(emb_list[hp.pta_dev_num:]), hp.pta_batch_size)
    else:
        dev_words = [w for w in dev_words['no gender']] \
                  + [w[0] for w in dev_words['female & male']] \
                  + [w[1] for w in dev_words['female & male']] \
                  + [w for w in dev_words['stereotype']]
        eval_inputs = torch.split(torch.FloatTensor([embs[word] for word in dev_words]), hp.pta_batch_size)
        train_inputs = torch.split(torch.FloatTensor([embs[word] for word in embs.wv.vocab if word not in dev_words]), hp.pta_batch_size)
    decoder_criterion = torch.nn.MSELoss()
    def vae_loss_function(mu, logsigma, kl_weight=1.0000):
  
        latent_loss = 0.5*torch.sum(logsigma.exp() + mu.pow(2)-1-logsigma)
        vae_loss = kl_weight*latent_loss 
        #print(vae_loss)
        return vae_loss
    def sampling(z_mean,z_logsigma):
        #print(hp.hid.shape)
        #print(z_logsigma)
        batch, latent_dim = z_mean.shape
        epsilon = torch.empty(batch,latent_dim).normal_(mean=0.0,std=1.0)

        z = z_mean + torch.exp(0.5*z_logsigma)*epsilon
        #print(z.shape)
        return z
    def encoder_1(emb_1):
        encoder_output = encoder(emb_1)
        z_mean = encoder_output[:,:hp.hidden_size//2]
       
        z_logsigma = encoder_output[:,hp.hidden_size//2:]
        
        return encoder_output, z_mean, z_logsigma
       
        
    def get_latent_mu(words,batch_size=4):
        
        N = len(words)
        mu = np.zeros((N, hp.hidden_size//2))
        for start_ind in range(0, N, batch_size):
            end_ind = min(start_ind+batch_size, N+1)
            if end_ind%batch_size!=0:
                break
            batch = (words[start_ind:end_ind])
            _,batch_mu, _ = encode(batch)
            
            mu[start_ind:end_ind] = batch_mu.detach().numpy()
        return mu
    def get_training_sample_probabilities(train_words,bins=4, smoothing_fac=0.001):
        mu = get_latent_mu(train_words)
        training_sample_p = np.zeros(mu.shape[0])
        for i in range(hp.hidden_size//2):
            latent_distribution = mu[:,i]
            # generate a histogram of the latent distribution
            hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

            # find which latent bin every data sample falls in 
            bin_edges[0] = -float('inf')
            bin_edges[-1] = float('inf')

            # TODO: call the digitize function to find which bins in the latent distribution 
            #    every data sample falls in to
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
            bin_idx = np.digitize(latent_distribution, bin_edges) # TODO

            # smooth the density function
            hist_smoothed_density = hist_density + smoothing_fac
            hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

            # invert the density function 
            p = 1.0/(hist_smoothed_density[bin_idx-1])

            # TODO: normalize all probabilities
            p = p/np.sum(p)

            # TODO: update sampling probabilities by considering whether the newly
            #     computed p is greater than the existing sampling probabilities.
            training_sample_p = np.maximum(p, training_sample_p)
        
    # final normalization
        training_sample_p /= np.sum(training_sample_p)

        return training_sample_p
    def shuffle_data1(words, p_pos):
        p_pos = list(p_pos)
        perm = np.random.choice(list(range(len(words))),size=len(words),p=p_pos)
        sorted_inds = np.sort(perm)
        words = [words[idx.item()] for idx in sorted_inds]
        return words
    def run_model(inputs, mode,p_inp):
        if mode == 'train':
            encoder.train()
            decoder.train()
            #perm = torch.randperm(len(inputs))
            inputs = shuffle_data1(inputs,p_in)
        elif mode == 'eval':
            encoder.eval()
            decoder.eval()
        total_num = 0
        total_loss = 0
        for input in inputs:
            encoder.zero_grad()
            decoder.zero_grad()
            hidden, z_mean,z_logsigma = encoder_1(input)
            z = sampling(z_mean,z_logsigma)
            pre = decoder(z)
            vae_loss = vae_loss_function(z_mean,z_logsigma)
            decoder_loss = decoder_criterion(pre, emb_dummy)
            
            loss = decoder_loss + vae_loss
            if mode == 'train':
                loss.backward()
                encoder_optim.step()
                decoder_optim.step()
            total_loss += loss.item()
            total_num += len(input)

        return total_loss / total_num


    min_loss = float('inf')
    for epoch in range(1, hp.pta_epochs):
        p_inp = get_training_sample_probabilities(train_inputs)
        train_loss = run_model(train_inputs, 'train',p_inp)
        eval_loss = run_model(eval_inputs, 'eval')

        if eval_loss < min_loss:
            min_epoch = epoch
            min_loss = eval_loss
            encoder_state_dict = encoder.state_dict()
            decoder_state_dict = decoder.state_dict()
            checkpoint = {
                'encoder': encoder_state_dict,
                'decoder': decoder_state_dict,
                'hp': hp
            }
            torch.save(checkpoint,
                '{}autoencoder_checkpoint'.format(hp.save_model))

    checkpoint = torch.load('{}autoencoder_checkpoint'.format(hp.save_model))
    torch.save(checkpoint, '{}autoencoder.pt'.format(hp.save_model))

    import os
    os.remove('{}autoencoder_checkpoint'.format(hp.save_model))

    return checkpoint
