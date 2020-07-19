---
layout: post
title:  "Connect colab to local runtime"
date:   2020-07-19 14:55:04 +0545
categories: jekyll blog

---
Connect your colab to local runtime


### Purpose of the blog:
Connect your colab to local runtime



### Resources
Link can be found [here at cartoon-gan paper][colab-runtime-steps].

### Prerequisites
Make sure you have jupyter installed in your local machine. If not here is a [link to install jupyter][install jupyter]


### Step 1

Install and enable the jupyter_http_over_ws jupyter extension (one-time)
The jupyter_http_over_ws extension is authored by the Colaboratory team and available on GitHub.
```
pip install jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
```

### Step 2
Start server using following command
```
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
```

Once the server has started, it will print a message with the initial backend URL used for authentication. Make a copy of this URL as you'll need to provide this in the next step.

### Final step
Go to colab and select the option to connect to runtime at the top right of the browser. Paste above url and click connect

![Result](/images/colab_scrt.png)


[colab-login]: https://colab.research.google.com/notebooks/welcome.ipynb#recent=true

[colab-runtime-steps]: https://research.google.com/colaboratory/local-runtimes.html

[install jupyter]: https://jupyter.org/install






