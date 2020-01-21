---
layout: post
title:  "Model Inferencing using CartoonGAN"
date:   2020-01-18 14:55:04 +0545
categories: jekyll blog

---
Hopefully, by the end of this blog you should be able to generate high-quality cartoon images from real-world photos. 


### Purpose of the blog:
This blog is a walk through to train your custom model based on the state of the technique in deep learning. 

### Resources
The paper can be found [here at cartoon-gan paper][cartoon-gan-paper].
Majority of the codes are taken from cartoon-gan-github-source [cartoon-gan-github-source]. 
I have merely changed anything in the source code. You could follow the instructions there as well. 

### Prerequisites
This blog would be useful for those who have very basic knowledge of Python programming language and some familiarity with unix shell programming. If you are not sure please go ahead to watch some useful tutorials


### Introduction

#### "Every adult is an obsolete children"
Cartoons are something which we have all watched while growing up. It impacts our lives in a positive way. They are entertaining, simple, and being utilized to convey very important messages.
Our objective is to learn how to use GAN in order to transform real world images into a cartoon character of our desired choice. 
While task might seem similar to painting but making cartoons are much more challenging. Cartoons have some high level simplification & abstraction, clear edges, smooth color
shading with simple textures. 


### What is GAN?
"GAN is the most interesting idea in the last ten years in machine learning"
-Yann LeCun, Director, Facebook AI 

GAN(Generative Adversarial Network) is one of the best architecture of the past decade when it comes to deep learning. These are generative models, implies, your models objective is to yield. Yields can be anything images,text, videos, the potential is enormous. 

Basic architecture consists of generator and discriminator. Generators, basically, generate some arbitraty guesses. Discriminator, on the other hand is learning to classify real v fake entities. The objective of the generator is to convince(fool) the discriminator that generated data is real enough.



Let us try to under the paper. I hope you have read it once, if not please go ahead. 


The architecture of CartoonGAN is shown below:

![GAN architecture](/images/gan_arch.png)



#### Two player game(min-max objective function)

<!-- $$
\begin{align}
  \nabla\times\vec{\mathbf{B}}-\frac{1}{c}\frac{\partial\vec{\mathbf{E}}}{\partial t} &= \frac{4\pi}{c}\vec{\mathbf{j}} \\
  \nabla\cdot\vec{\mathbf{E}} &= 4\pi\rho \\
  \nabla\times\vec{\mathbf{E}}+\frac{1}{c}\frac{\partial\vec{\mathbf{B}}}{\partial t} &= \vec{\mathbf{0}} \\
  \nabla\cdot\vec{\mathbf{B}} &= 0
\end{align}
$$
 -->
 {% raw %}

  $$ min max (D, G) = E x∼p data (x) [logD(x)] + E x∼p z (z) [log(1 − D ((G z) )))] $$	

 {% endraw %}


#### Time for action with some codes
Hope you have [signed in][colab-login] and opened the colab notebook in your browser. Paste the code snippets below and run one after the other. Check your outputs!!!

Here is the code to perform model inferencing with the trained model found in this paper.



Clone the repository from github
{%highlight ruby%}

"""Code copied from "https://github.com/mnicnc404/CartoonGan-tensorflow"  """

import os
repo = "CartoonGan-tensorflow"
!git clone https://github.com/HiteshAI/CartoonGan-tensorflow.git
os.chdir(os.path.join(repo))

{%endhighlight%}


Import necessary dependancies
{%highlight ruby%}

from IPython.display import clear_output, display, Image
!pip install tensorflow-gpu==2.0.0-alpha0
!git clone https://www.github.com/keras-team/keras-contrib.git \
    && cd keras-contrib \
    && python convert_to_tf_keras.py \
    && USE_TF_KERAS=1 python setup.py install
import tensorflow as tf
tf.__version__
clear_output()

{%endhighlight%}

Use the script to perform model infrencing on your image.
{% highlight ruby %}
!python cartoonize.py \
    --batch_size 4 \
    --all_styles \
    --comparison_view horizontal \
    --max_resized_height 800

{% endhighlight %}





<!-- Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk]. -->

If you want to contribute, please raise an [PR here][myrepo-cartoongan] 

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
[cartoon-gan-paper]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf
[cartoon-gan-github-source]:https://github.com/mnicnc404/CartoonGan-tensorflow
[myrepo-cartoongan]: https://github.com/hiteshai/CartoonGan-tensorflow
[colab-login]: https://colab.research.google.com/notebooks/welcome.ipynb#recent=true








