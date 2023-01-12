import jittor as jt

class Entropy_Loss:
    def __init__(self, N_rand, threshold, N_entropy):
        super(Entropy_Loss, self).__init__()
        self.N_samples = N_rand
        self.threshold = threshold
        self.N_entropy = N_entropy
        
    def compute_loss(self, sigma, acc_map): # following paper equation (4)-(8)
        sigma = sigma[self.N_samples:]
        ray_prob = sigma / (jt.sum(sigma, -1).unsqueeze(-1) + 1e-8)
        ray_entropy = -1 * ray_prob * jt.log2(ray_prob + 1e-8) # compute ray entropy
        entropy_loss = jt.sum(ray_entropy, -1)
        
        acc_map = acc_map[self.N_samples:]
        mask = (acc_map > self.threshold).detach() # compute ray mask
        
        entropy_loss *= mask
        return jt.mean(entropy_loss)