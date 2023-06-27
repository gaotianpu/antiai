# http://joschu.net/blog/kl-approx.html
import torch.distributions as dis

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)

x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)
print("true", truekl)

logr = p.log_prob(x) - q.log_prob(x)

k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr

# 三种求KL散度函数直观表示 https://www.geogebra.org/graphing/eu9dufhd
for k in (k1, k2, k3):
    print((k.mean() - truekl) / truekl, k.std() / truekl)